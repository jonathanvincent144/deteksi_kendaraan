"""
Aplikasi Inference Deteksi Objek (Streamlit)

Model A : YOLOv8          -> weights/yolo_best.pt
Model B : Faster R-CNN    -> weights/frcnn_best.pth   (torchvision, ResNet50-FPN)

Jalankan:  streamlit run app.py
"""
import colorsys
import io
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ----------------------------------------------------------------------------
# Konfigurasi (samakan dengan notebook pelatihan)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / "weights"
YOLO_PATH = WEIGHTS_DIR / "yolo_best.pt"
FRCNN_PATH = WEIGHTS_DIR / "frcnn_best.pth"
CLASSES_PATH = WEIGHTS_DIR / "classes.txt"   # opsional: satu nama kelas per baris

IMG_SIZE = 512          # ukuran input saat pelatihan
YOLO_IOU = 0.6          # ambang NMS YOLO (sama dengan evaluasi di notebook)
REALTIME_MAX_SIDE = 640  # frame webcam diperkecil agar real-time lebih ringan

MODEL_A = "Model A - YOLOv8"
MODEL_B = "Model B - Faster R-CNN"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:  # webcam real-time bersifat opsional (nilai tambah)
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

    HAS_WEBRTC = True
except Exception:  # pragma: no cover
    HAS_WEBRTC = False


# ----------------------------------------------------------------------------
# Pemuatan model
# ----------------------------------------------------------------------------
@dataclass
class Detector:
    kind: str        # "yolo" atau "frcnn"
    model: object
    names: list


def _names_from_yolo_file():
    from ultralytics import YOLO

    m = YOLO(str(YOLO_PATH))
    return [m.names[i] for i in sorted(m.names)]


def _read_class_names():
    """Urutan kelas harus sama dengan saat training. Prioritas: classes.txt, lalu bobot YOLO."""
    if CLASSES_PATH.exists():
        lines = CLASSES_PATH.read_text(encoding="utf-8").splitlines()
        names = [ln.strip() for ln in lines if ln.strip()]
        if names:
            return names
    if YOLO_PATH.exists():
        return _names_from_yolo_file()
    raise FileNotFoundError(
        "Nama kelas tidak ditemukan. Letakkan weights/yolo_best.pt atau buat weights/classes.txt "
        "(satu nama kelas per baris, urutan sama dengan data.yaml)."
    )


@st.cache_resource(show_spinner="Memuat model...")
def load_detector(choice: str) -> Detector:
    if choice == MODEL_A:
        if not YOLO_PATH.exists():
            raise FileNotFoundError(f"Bobot tidak ditemukan: {YOLO_PATH}")
        from ultralytics import YOLO

        model = YOLO(str(YOLO_PATH))
        names = _read_class_names()
        return Detector("yolo", model, names)

    # Model B: Faster R-CNN (arsitektur sama dengan notebook, kelas terakhir = latar belakang)
    if not FRCNN_PATH.exists():
        raise FileNotFoundError(f"Bobot tidak ditemukan: {FRCNN_PATH}")
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    names = _read_class_names()
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=IMG_SIZE, max_size=IMG_SIZE)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(names) + 1)
    state = torch.load(str(FRCNN_PATH), map_location="cpu")
    if isinstance(state, dict) and "model" in state:   # jika yang dimuat adalah last.pth
        state = state["model"]
    model.load_state_dict(state)
    model.roi_heads.score_thresh = 0.01     # disaring ulang oleh slider
    model.roi_heads.detections_per_img = 100
    model.to(DEVICE).eval()
    return Detector("frcnn", model, names)


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def detect(det: Detector, image: Image.Image, conf: float):
    """Return boxes (xyxy, piksel gambar asli), scores, labels, waktu inferensi (ms)."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    if det.kind == "yolo":
        r = det.model.predict(
            image, imgsz=IMG_SIZE, conf=conf, iou=YOLO_IOU, verbose=False,
            device=0 if DEVICE.type == "cuda" else "cpu",
        )[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy().astype(int)
    else:
        from torchvision.transforms.functional import to_tensor

        x = to_tensor(image).to(DEVICE)
        with torch.inference_mode():
            out = det.model([x])[0]
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = (out["labels"] - 1).cpu().numpy().astype(int)   # 0 = latar belakang
        keep = scores >= conf
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    return boxes, scores, labels, ms


# ----------------------------------------------------------------------------
# Visualisasi
# ----------------------------------------------------------------------------
def _color_for(idx: int, n: int):
    r, g, b = colorsys.hsv_to_rgb((idx / max(n, 1)) % 1.0, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _load_font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_detections(image: Image.Image, boxes, scores, labels, names) -> Image.Image:
    img = image.copy().convert("RGB")
    d = ImageDraw.Draw(img)
    longest = max(img.size)
    line_w = max(2, round(longest / 300))
    font = _load_font(max(12, round(longest / 45)))

    for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
        col = _color_for(int(l), len(names))
        d.rectangle([x1, y1, x2, y2], outline=col, width=line_w)
        text = f"{names[int(l)]} {s:.2f}"
        l_, t_, r_, b_ = d.textbbox((0, 0), text, font=font)
        tw, th = r_ - l_, b_ - t_
        ty = y1 - th - 8 if y1 - th - 8 >= 0 else y1
        d.rectangle([x1, ty, x1 + tw + 8, ty + th + 8], fill=col)
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        d.text((x1 + 4, ty + 4 - t_), text, fill=(0, 0, 0) if lum > 140 else (255, 255, 255), font=font)
    return img


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def count_per_class(labels, names) -> pd.DataFrame:
    c = Counter(int(x) for x in labels)
    rows = [{"Class": names[i], "Jumlah": c[i]} for i in sorted(c)]
    return pd.DataFrame(rows, columns=["Class", "Jumlah"])


# ----------------------------------------------------------------------------
# Tampilan hasil (dipakai tab upload dan webcam foto)
# ----------------------------------------------------------------------------
def get_detector_or_stop(choice: str) -> Detector:
    try:
        return load_detector(choice)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Salin bobot hasil training ke folder `weights/` (lihat README.md).")
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(f"Gagal memuat model: {e}")
        st.stop()


def run_and_show(image: Image.Image, choice: str, conf: float, file_stem: str):
    det = get_detector_or_stop(choice)
    boxes, scores, labels, ms = detect(det, image, conf)
    annotated = draw_detections(image, boxes, scores, labels, det.names)

    col_img, col_info = st.columns([3, 2])
    with col_img:
        st.image(annotated, caption=f"Hasil deteksi - {choice} (conf >= {conf:.2f})")
        st.download_button(
            "Unduh gambar hasil deteksi", data=to_png_bytes(annotated),
            file_name=f"{file_stem}_deteksi.png", mime="image/png", key=f"dl_{file_stem}_{choice}",
        )
    with col_info:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total objek", len(scores))
        m2.metric("Waktu inferensi", f"{ms:.0f} ms")
        m3.metric("Perangkat", "GPU" if DEVICE.type == "cuda" else "CPU")

        st.markdown("**Jumlah objek per class**")
        df = count_per_class(labels, det.names)
        if df.empty:
            st.info("Tidak ada objek terdeteksi. Coba turunkan confidence threshold.")
        else:
            st.dataframe(df, hide_index=True)
            st.bar_chart(df.set_index("Class"))
            with st.expander("Detail setiap deteksi"):
                detail = pd.DataFrame({
                    "Class": [det.names[int(i)] for i in labels],
                    "Confidence": np.round(scores, 3),
                    "x1": boxes[:, 0].round(0), "y1": boxes[:, 1].round(0),
                    "x2": boxes[:, 2].round(0), "y2": boxes[:, 3].round(0),
                }).sort_values("Confidence", ascending=False)
                st.dataframe(detail, hide_index=True)


def open_image(file) -> Image.Image:
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)   # perbaiki orientasi foto dari ponsel
    return img.convert("RGB")


# ----------------------------------------------------------------------------
# Webcam real-time (streamlit-webrtc)
# ----------------------------------------------------------------------------
if HAS_WEBRTC:

    class DetectProcessor(VideoProcessorBase):
        def __init__(self):
            self.det = None
            self.conf = 0.25
            self.last_ms = 0.0
            self.last_count = 0
            self._lock = threading.Lock()

        def recv(self, frame):
            img = frame.to_image()
            img.thumbnail((REALTIME_MAX_SIDE, REALTIME_MAX_SIDE))
            with self._lock:
                det, conf = self.det, self.conf
            if det is not None:
                boxes, scores, labels, ms = detect(det, img, conf)
                img = draw_detections(img, boxes, scores, labels, det.names)
                self.last_ms, self.last_count = ms, len(scores)
            return av.VideoFrame.from_image(img)


# ----------------------------------------------------------------------------
# Halaman utama
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Deteksi Objek", page_icon="🔍", layout="wide")
    st.title("🔍 Aplikasi Deteksi Objek")
    st.caption("Bandingkan Model A (YOLOv8) dan Model B (Faster R-CNN) pada gambar upload atau webcam.")

    with st.sidebar:
        st.header("Pengaturan")
        choice = st.radio("Pilih model", [MODEL_A, MODEL_B])
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                         help="Hanya objek dengan confidence >= nilai ini yang ditampilkan.")
        st.divider()
        st.markdown(f"**Perangkat:** {'GPU (CUDA)' if DEVICE.type == 'cuda' else 'CPU'}")
        st.markdown(f"**Ukuran input model:** {IMG_SIZE} x {IMG_SIZE}")
        for label, p in (("yolo_best.pt", YOLO_PATH), ("frcnn_best.pth", FRCNN_PATH)):
            if p.exists():
                st.markdown(f"✅ `{label}` ({p.stat().st_size / 1e6:.1f} MB)")
            else:
                st.markdown(f"❌ `{label}` belum ada di `weights/`")

    tab_up, tab_cam, tab_rt = st.tabs(["📁 Upload Gambar", "📷 Foto Webcam", "🎥 Webcam Real-time"])

    with tab_up:
        files = st.file_uploader(
            "Upload atau drag and drop gambar (jpg, jpeg, png)",
            type=["jpg", "jpeg", "png"], accept_multiple_files=True,
        )
        for i, f in enumerate(files or []):
            if i:
                st.divider()
            st.subheader(f.name)
            run_and_show(open_image(f), choice, conf, Path(f.name).stem)

    with tab_cam:
        st.caption("Izinkan akses kamera pada browser. Kamera hanya bisa dipakai lewat `localhost` atau HTTPS.")
        shot = st.camera_input("Ambil foto")
        if shot is not None:
            run_and_show(open_image(shot), choice, conf, "webcam")

    with tab_rt:
        if not HAS_WEBRTC:
            st.warning("Fitur ini membutuhkan `streamlit-webrtc` dan `av`: `pip install streamlit-webrtc av`.")
        else:
            st.caption("Deteksi langsung pada video webcam. Faster R-CNN di CPU akan terasa lambat; "
                       "YOLOv8 jauh lebih ringan. Ubah slider atau model, lalu video akan mengikuti.")
            ctx = webrtc_streamer(
                key="realtime",
                video_processor_factory=DetectProcessor,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            if ctx.video_processor:
                det = get_detector_or_stop(choice)
                with ctx.video_processor._lock:
                    ctx.video_processor.det = det
                    ctx.video_processor.conf = conf


if __name__ == "__main__":
    main()
