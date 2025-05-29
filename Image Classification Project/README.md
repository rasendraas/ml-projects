
---

# Deteksi Penyakit Malaria Menggunakan CNN

## Deskripsi
Proyek ini bertujuan untuk melakukan deteksi penyakit malaria menggunakan citra sel darah menggunakan arsitektur Convolutional Neural Network (CNN). Model dilatih untuk mengklasifikasikan gambar ke dalam dua kategori: Parasitized (terinfeksi malaria) dan Uninfected (tidak terinfeksi malaria).

## Langkah-langkah Proyek

### 1. Import Library
Pada tahap pertama, berbagai library penting digunakan seperti TensorFlow, Keras, NumPy, OpenCV, Matplotlib, dan scikit-learn untuk proses pelatihan, evaluasi, dan konversi model.

### 2. Data Loading
Dataset gambar dimuat dari direktori lokal menggunakan fungsi `os.listdir()` untuk membaca citra dari folder yang masing-masing mewakili kelas (Parasitized dan Uninfected).

### 3. Preprocessing
Gambar yang dimuat diproses dengan beberapa tahap:
- Menggunakan `cv2` untuk membaca gambar dan mengonversinya ke format RGB.
- Mengubah ukuran gambar menjadi 128x128 piksel.
- Normalisasi gambar dengan membagi nilai piksel dengan 255.
- Menggunakan `LabelEncoder` untuk mengubah label kategori menjadi angka.

### 4. Pembuatan Model CNN
Arsitektur model CNN dibuat menggunakan Keras dengan beberapa lapisan konvolusi (`Conv2D`), lapisan penggabungan (`MaxPooling2D`), serta dropout untuk mengurangi overfitting. Output akhir menggunakan fungsi aktivasi sigmoid untuk klasifikasi dua kelas.

### 5. Evaluasi Model
Model dievaluasi menggunakan data pelatihan, validasi, dan pengujian dengan metrik akurasi untuk memeriksa kinerja model.

### 6. Konversi Model
Setelah model dilatih, dilakukan konversi model untuk digunakan pada platform lain:
- **TensorFlow.js** (`.tfjs` format)
- **TensorFlow Lite** (`.tflite` format)
- **SavedModel** (`saved_model.pb` format)

## Instalasi
Untuk menjalankan proyek ini, pastikan Anda memiliki Python >=3.11 dan menginstal library yang diperlukan. Anda dapat menginstalnya dengan menggunakan `pip`:

```bash
pip install tensorflow opencv-python scikit-learn matplotlib tqdm
```
Atau bisa menggunakan file requirements.txt untuk install semua library yang dibutuhkan
### Instalasi dengan `requirements.txt`

File tersebut berisi semua dependensi yang diperlukan untuk menjalankan proyek ini. Anda dapat menginstal semua dependensi dengan perintah berikut:

```bash
pip install -r requirements.txt
```
### Penjelasan Library yang Digunakan

1. **tensorflow**: 
   Digunakan untuk membangun dan melatih model CNN. TensorFlow menyediakan berbagai alat untuk pengolahan data, pembuatan model, dan evaluasi.

2. **opencv-python**: 
   Digunakan untuk pemrosesan citra, seperti membaca gambar, mengonversinya menjadi format RGB, dan meresize gambar.

3. **scikit-learn**: 
   Digunakan untuk preprocessing data dan evaluasi model. Digunakan juga untuk `LabelEncoder` yang mengubah label kategori menjadi angka.

4. **matplotlib**: 
   Digunakan untuk visualisasi grafik dan gambar. Pada proyek ini, digunakan untuk menampilkan gambar dan hasil prediksi model.

5. **tqdm**: 
   Digunakan untuk menampilkan progress bar selama proses loading dan pelatihan, memberikan indikasi seberapa jauh proses tersebut berjalan.

## Penggunaan
1. Download isi semua folder ini.
2. Jalankan notebook (model.ipynb) untuk memulai pelatihan dan evaluasi model.

```bash
jupyter notebook
```

## Hasil
- **Akurasi Train**: 0.9736
- **Akurasi Validation**: 0.9572
- **Akurasi Test**: 0.9561

Model berhasil mencapai akurasi yang tinggi pada data pelatihan, validasi, dan pengujian, menunjukkan bahwa model CNN ini dapat diandalkan untuk deteksi malaria.

---