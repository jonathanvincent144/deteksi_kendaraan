# Aplikasi Deteksi Objek (Streamlit)

Aplikasi inference untuk membandingkan dua model hasil pelatihan:

- **Model A**: YOLOv8 (Ultralytics)
- **Model B**: Faster R-CNN ResNet50-FPN (torchvision)

Kedua model dilatih dengan input 512 x 512 dan bobot awal pretrained COCO.

## Fitur

| Fitur | Keterangan |
|---|---|
| Upload / drag and drop | jpg, jpeg, png (bisa banyak gambar sekaligus) |
| Foto webcam | Tab "Foto Webcam" (`st.camera_input`) |
| Webcam real-time | Tab "Webcam Real-time" (`streamlit-webrtc`) |
| Hasil deteksi | Bounding box + nama class + confidence score |
| Jumlah per class | Tabel dan grafik batang, plus detail tiap deteksi |
| Confidence threshold | Slider di sidebar (0.05 - 0.95) |
| Pilihan model | Model A atau Model B di sidebar |
| Waktu inferensi | Ditampilkan (ms) beserta perangkat (CPU/GPU) |
| Unduh hasil | Tombol unduh gambar hasil deteksi (PNG) |

## Struktur Folder

```
project-deteksi/
├── app.py
├── requirements.txt
├── weights/
│   ├── yolo_best.pt
│   └── frcnn_best.pth
└── README.md
```

## Menyiapkan Bobot

Salin hasil dari Kaggle (`/kaggle/working/hasil`) ke folder `weights/`:

| Sumber | Tujuan |
|---|---|
| `hasil/yolo/yolov8_run/weights/best.pt` | `weights/yolo_best.pt` |
| `hasil/frcnn/best.pth` | `weights/frcnn_best.pth` |

Nama kelas dibaca otomatis dari `yolo_best.pt`, dengan urutan yang sama seperti saat training,
sehingga Model B memakai daftar yang sama. Jika ingin menjalankan Model B saja, buat
`weights/classes.txt` berisi satu nama kelas per baris (urutan sama dengan `data.yaml`).

## Instalasi dan Menjalankan

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Buka alamat yang ditampilkan (biasanya http://localhost:8501).

Jika tidak punya GPU, instal PyTorch versi CPU agar lebih ringan, mis.
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, lalu sisanya dari `requirements.txt`.

## Catatan Penggunaan

- **Webcam** hanya bisa diakses browser lewat `localhost` atau HTTPS. Izinkan akses kamera saat diminta.
- **Real-time**: YOLOv8 cukup lancar di CPU, sedangkan Faster R-CNN akan terasa patah-patah tanpa GPU.
  Frame diperkecil (maks 640 px) agar lebih ringan.
- Jika di-deploy ke server publik, koneksi webcam real-time bisa membutuhkan server TURN
  selain STUN bawaan (lihat dokumentasi `streamlit-webrtc`).
- Konfigurasi yang harus sama dengan notebook ada di bagian atas `app.py`:
  `IMG_SIZE = 512` dan `YOLO_IOU = 0.6`.
- Waktu inferensi di aplikasi mencakup pra-proses dan NMS untuk satu gambar. Angkanya bisa sedikit
  berbeda dari benchmark di notebook karena perangkat dan beban sistem berbeda.
