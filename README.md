# object_detection
Object detection model using roboflow and google colab

# Explanation_Video(ID)

```
https://youtu.be/peXuVK7Nbgo
```

# Code
```python
# Instalasi dan Import Library
!pip install ultralytics roboflow
from roboflow import Roboflow
from ultralytics import YOLO
from google.colab import files
from IPython.display import Image, display
import os
import glob

# Mengunduh Dataset dari Roboflow
rf = Roboflow(api_key="LcM7B3izc1nDg86sdSsi")
project = rf.workspace("object-detect-t3tfp").project("objectdetection-b7kzv-0gihr-l2wvv")
version = project.version(2)
dataset = version.download("yolov8")


model = YOLO("yolov8n.pt")  # Gunakan YOLOv8 versi Nano untuk kecepatan
data_path = dataset.location + "/data.yaml"  # Path file data.yaml yang disediakan oleh Roboflow

# Melatih model
model.train(data=data_path, epochs=50, imgsz=640)

# Memuat Model yang Sudah Dilatih
model = YOLO("runs/detect/train/weights/best.pt")  # Ganti dengan path model yang sudah dilatih

# Input Gambar untuk Deteksi
print("Silakan unggah gambar yang ingin dideteksi:")
uploaded = files.upload()  # Mengunggah file gambar

# Deteksi dan simpan hasilnya
for file_name in uploaded.keys():
    print(f"Gambar berhasil diunggah: {file_name}")

    # Melakukan prediksi
    result = model.predict(source=file_name, save=True)

    # Mencari folder terbaru yang berisi hasil deteksi
    result_dir_parent = "runs/detect"  # Direktori induk hasil deteksi
    result_dirs = glob.glob(os.path.join(result_dir_parent, "predict*"))  # Cari folder yang namanya diawali dengan 'predict'

    if result_dirs:
        # Menemukan folder terbaru berdasarkan waktu pembuatan
        latest_result_dir = max(result_dirs, key=os.path.getmtime)  # Menentukan folder yang paling baru
        print(f"Folder hasil deteksi terbaru: {latest_result_dir}")

        # Menemukan file gambar hasil deteksi dalam folder terbaru
        detected_files = glob.glob(os.path.join(latest_result_dir, "*"))  # Mengambil semua file di dalam folder
        detected_image_files = [f for f in detected_files if f.endswith(('.jpg', '.jpeg', '.png'))]  # Filter hanya gambar

        if detected_image_files:
            output_image_path = detected_image_files[0]  # Ambil gambar pertama yang ditemukan
            print(f"Hasil deteksi disimpan di: {output_image_path}")

            # Tampilkan gambar hasil deteksi
            display(Image(filename=output_image_path))
        else:
            print(f"Tidak ada gambar hasil deteksi di folder {latest_result_dir}.")
    else:
        print(f"Tidak ada folder hasil deteksi ditemukan di {result_dir_parent}.")

```

# Output

![download (2)](https://github.com/user-attachments/assets/a354849c-d0b1-4574-822b-bcb5503eb2b8)
