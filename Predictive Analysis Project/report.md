# Laporan Proyek Machine Learning  
## Air Quality & Pollution Assessment Quality Prediction  
**Rasendra Akbar Satyatama** — *MC004D5Y1124*

---
## Daftar Isi

- [Domain Proyek](#domain-proyek-lingkungan-dan-kesehatan-masyarakat)
  - [Referensi](#referensi)
- [Business Understanding](#business-understanding)
  - [Problem Statements](#problem-statements)
  - [Goals](#goals)
  - [Solution Statements](#solution-statements)
- [Data Understanding](#data-understanding)
  - [Sumber Data](#sumber-data)
  - [Deskripsi Variabel](#exploratory-data-analysis---deskripsi-variabel)
  - [Missing Value & Outliers](#exploratory-data-analysis---menangani-missing-value-dan-outliers)
  - [Univariate Analysis](#exploratory-data-analysis---univariate-analysis)
  - [Multivariate Analysis](#exploratory-data-analysis---multivariate-analysis)
  - [Analisis Faktor yang Memengaruhi Kualitas Udara](#analisis-faktor-yang-memengaruhi-kualitas-udara)
- [Data Preparation](#data-preparation)
  - [Label Encoding dengan Mapping pada Fitur Target](#label-encoding-dengan-mapping-pada-fitur-target)
  - [Reduksi Dimensi dengan Principal Component Analysis (PCA)](#reduksi-dimensi-dengan-principal-component-analysis-pca)
  - [Train-Test Split](#train-test-split)
  - [Data Scaling dengan RobustScaler](#data-scaling-dengan-robustscaler)
- [Model Development](#model-development)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Random Forest (RF)](#random-forest-rf)
  - [XGBoost (XGB)](#xgboost-xgb)
- [Model Evaluation](#model-evaluation)
  - [Akurasi Model & Classification Report](#akurasi-model--classification-report)
  - [Confusion Matrix](#confusion-matrix)
  - [Grafik AUC-ROC](#grafik-auc-roc)
  - [Kesimpulan Evaluasi Model Klasifikasi](#kesimpulan-evaluasi-model-klasifikasi)
- [Kesimpulan Akhir](#kesimpulan-akhir)

---
## **Domain Proyek: Lingkungan dan Kesehatan Masyarakat**

Kualitas udara merupakan aspek krusial dalam isu lingkungan dan kesehatan masyarakat. Menurut World Health Organization (WHO), polusi udara menyebabkan sekitar **7 juta kematian dini setiap tahun** di seluruh dunia karena paparan terhadap partikel halus dan gas beracun seperti PM2.5, NO₂, dan SO₂ \[1]. Wilayah dengan tingkat kepadatan penduduk tinggi atau dekat dengan kawasan industri cenderung memiliki tingkat polusi udara yang lebih tinggi, sehingga berdampak signifikan terhadap risiko penyakit pernapasan, kardiovaskular, dan berbagai gangguan kesehatan lainnya.

Dengan semakin pesatnya urbanisasi dan pertumbuhan industri, penting bagi pemerintah, organisasi lingkungan, dan masyarakat umum untuk memiliki sistem pemantauan dan prediksi kualitas udara yang dapat digunakan sebagai dasar pengambilan keputusan. Pendekatan konvensional dalam pemantauan kualitas udara, seperti penggunaan sensor stasioner, memiliki keterbatasan cakupan geografis dan biaya operasional yang tinggi. Oleh karena itu, pendekatan berbasis data dan machine learning dapat menjadi solusi yang efisien dan fleksibel dalam memperkirakan kondisi kualitas udara berdasarkan faktor lingkungan dan demografis.

Proyek ini berada dalam domain **lingkungan hidup dan kesehatan publik**, dengan fokus pada pengembangan model prediksi berbasis data untuk memetakan tingkat kualitas udara suatu wilayah secara otomatis. Dengan prediksi yang akurat, pemerintah dapat mengidentifikasi daerah risiko tinggi, mengatur kebijakan emisi, atau menyampaikan peringatan dini kepada masyarakat.

---

### Referensi:

\[1] World Health Organization (2021). *Air pollution*. Retrieved from [https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health](https://www.who.int/news-room/fact-sheets/detail/ambient-%28outdoor%29-air-quality-and-health)

---

## Business Understanding

### Problem Statements

Dalam konteks urbanisasi dan industrialisasi yang pesat, kualitas udara menjadi perhatian utama karena dampaknya terhadap kesehatan masyarakat. Organisasi lingkungan hidup, pemerintah daerah, serta badan perencana kota membutuhkan alat prediksi yang andal untuk memperkirakan kualitas udara secara proaktif. Berdasarkan hal tersebut, berikut adalah pernyataan masalah yang diangkat:

1. **Pernyataan Masalah 1:**  
   Bagaimana mengidentifikasi fitur lingkungan dan demografis yang paling berpengaruh terhadap tingkat kualitas udara di suatu wilayah?

2. **Pernyataan Masalah 2:**  
   Bagaimana membangun model prediksi yang dapat mengklasifikasikan tingkat kualitas udara (*Good, Moderate, Poor, Hazardous*) secara akurat berdasarkan data lingkungan dan kependudukan?

3. **Pernyataan Masalah 3:**  
   Bagaimana menentukan model yang dibangun mampu melakukan prediksi dengan performa optimal dan konsisten pada data baru?

---

### Goals

Untuk menjawab pernyataan masalah di atas, tujuan proyek ini dirumuskan sebagai berikut:

1. **Tujuan 1:**  
   Melakukan eksplorasi data dan analisis korelasi untuk mengetahui fitur mana yang paling signifikan terhadap kualitas udara, seperti konsentrasi PM2.5, NO₂, kepadatan penduduk, atau jarak ke kawasan industri.

2. **Tujuan 2:**  
   Membangun model machine learning klasifikasi multikelas yang dapat memprediksi kategori kualitas udara secara otomatis berdasarkan fitur numerik dan kategorikal.

3. **Tujuan 3:**  
   Melakukan evaluasi terhadap semua model yang digunakan untuk mengetahui prediksi dengan akurasi dan generalisasi terbaik.

---

### Solution Statements

Untuk mencapai tujuan-tujuan tersebut, solusi yang akan diimplementasikan meliputi:

1. **Eksperimen Beberapa Algoritma Klasifikasi:**  
   Membangun dan membandingkan performa dari beberapa algoritma klasifikasi seperti:
   - K-Nearest Neighbors (KNN)
   - Random Forest
   - XGBoost

2. **Evaluasi Berdasarkan Metrik yang Terukur:**  
   Menggunakan metrik evaluasi seperti:
   - **Accuracy** untuk mengukur kebenaran prediksi secara keseluruhan.
   - **Precision, Recall, dan F1-Score** untuk menilai performa tiap kelas.
   - **Confusion Matrix** untuk melihat distribusi kesalahan klasifikasi.

3. **Visualisasi Hasil Fitur Penting:**  
   Menggunakan *feature importance plot* untuk menjelaskan fitur mana yang paling berkontribusi terhadap prediksi model.

---

## Data Understanding

### Sumber Data
Data yang diperoleh berasal dari platform open source yaitu kaggle dan dapat diakses melalui link berikut.
[Air Quality and Pollution Assessment](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment)

### [Exploratory Data Analysis] - Deskripsi Variabel
| Nama Variabel                   | Tipe Data | Deskripsi                                                                 |
|---------------------------------|-----------|---------------------------------------------------------------------------|
| `Temperature`                   | float64   | Suhu udara lingkungan (dalam satuan derajat Celcius).                     |
| `Humidity`                      | float64   | Kelembapan udara relatif (dalam persentase).                              |
| `PM2.5`                         | float64   | Konsentrasi partikel halus dengan diameter ≤2.5µm (mikrogram/m³).         |
| `PM10`                          | float64   | Konsentrasi partikel kasar dengan diameter ≤10µm (mikrogram/m³).          |
| `NO2`                           | float64   | Konsentrasi gas nitrogen dioksida (NO₂) di udara (mikrogram/m³).          |
| `SO2`                           | float64   | Konsentrasi gas sulfur dioksida (SO₂) di udara (mikrogram/m³).            |
| `CO`                            | float64   | Konsentrasi gas karbon monoksida (CO) di udara (ppm).                     |
| `Proximity_to_Industrial_Areas`| float64   | Jarak ke area industri terdekat (dalam kilometer).                        |
| `Population_Density`           | int64     | Kepadatan penduduk di wilayah tersebut (jumlah penduduk per km²).        |
| `Air Quality`                   | object    | Label kualitas udara (kategori: Good, Moderate, Poor, Hazardous).        |

Dataset yang digunakan berisi **5000 observasi** dengan **9 fitur prediktor** numerik dan **1 fitur target** yaitu *Air Quality*. Secara umum, fitur-fitur seperti **Temperature**, **Humidity**, **PM2.5**, **PM10**, **NO2**, **SO2**, **CO**, **Proximity to Industrial Areas**, dan **Population Density** direpresentasikan dalam tipe data numerik, dengan sebaran nilai yang cukup bervariasi. Misalnya, konsentrasi **PM2.5** memiliki nilai maksimum hingga **295 µg/m³**, sedangkan variabel **SO2** bahkan menunjukkan adanya nilai negatif pada data mentah, yang mengindikasikan potensi outlier atau kesalahan pengukuran. Rata-rata kepadatan penduduk berada di angka **497 orang/km²**, dengan jarak rata-rata ke kawasan industri sekitar **8.43 km**.

| Fitur                                  | Mean   | Std Dev | Min   | 25%   | Median | 75%   | Max   |
| -------------------------------------- | ------ | ------- | ----- | ----- | ------ | ----- | ----- |
| **Temperature (°C)**                   | 30.03  | 6.72    | 13.4  | 25.1  | 29.0   | 34.0  | 58.6  |
| **Humidity (%)**                       | 70.06  | 15.86   | 36.0  | 58.3  | 69.8   | 80.3  | 128.1 |
| **PM2.5 (µg/m³)**                      | 20.14  | 24.55   | 0.0   | 4.6   | 12.0   | 26.1  | 295.0 |
| **PM10 (µg/m³)**                       | 30.22  | 27.35   | -0.2  | 12.3  | 21.7   | 38.1  | 315.8 |
| **NO2 (ppb)**                          | 26.41  | 8.90    | 7.4   | 20.1  | 25.3   | 31.9  | 64.9  |
| **SO2 (ppb)**                          | 10.01  | 6.75    | -6.2  | 5.1   | 8.0    | 13.73 | 44.9  |
| **CO (ppm)**                           | 1.50   | 0.55    | 0.65  | 1.03  | 1.41   | 1.84  | 3.72  |
| **Proximity to Industrial Areas (km)** | 8.43   | 3.61    | 2.5   | 5.4   | 7.9    | 11.1  | 25.8  |
| **Population Density (people/km²)**    | 497.42 | 152.75  | 188.0 | 381.0 | 494.0  | 600.0 | 957.0 |

### Rata-rata Fitur per Kategori Air Quality

| Air Quality   | Temperature | Humidity | PM2.5 | PM10  | NO2   | SO2   | CO   | Proximity to Industry | Population Density |
| ------------- | ----------- | -------- | ----- | ----- | ----- | ----- | ---- | --------------------- | ------------------ |
| **Good**      | 24.95       | 60.02    | 9.91  | 14.99 | 19.45 | 5.04  | 1.00 | 11.99                 | 398.94             |
| **Hazardous** | 40.35       | 89.47    | 41.92 | 61.51 | 40.60 | 20.02 | 2.49 | 4.59                  | 696.01             |
| **Moderate**  | 30.14       | 70.21    | 20.46 | 30.60 | 26.44 | 9.98  | 1.51 | 6.96                  | 497.57             |
| **Poor**      | 34.87       | 80.18    | 29.24 | 44.45 | 33.21 | 15.03 | 2.00 | 5.42                  | 594.88             |


Berdasarkan statistik deskriptif per kategori *Air Quality*, terlihat pola yang konsisten antara kenaikan polutan dan penurunan kualitas udara. Kategori **Hazardous** memiliki rata-rata tertinggi pada semua indikator polusi seperti **PM2.5 (41.92 µg/m³)**, **PM10 (61.51 µg/m³)**, **NO2 (40.60 ppb)**, **SO2 (20.02 ppb)**, dan **CO (2.49 ppm)**. Sebaliknya, kategori **Good** menunjukkan nilai terendah untuk semua polutan tersebut, sekaligus memiliki jarak terjauh ke kawasan industri (**11.99 km**) dan densitas penduduk yang relatif rendah (**398.94 orang/km²**).

Selain faktor polusi, variabel **Temperature** dan **Humidity** juga menunjukkan kecenderungan yang sejalan dengan penurunan kualitas udara. Lokasi dengan kualitas udara buruk cenderung memiliki suhu dan kelembapan lebih tinggi. Hal ini memperkuat dugaan bahwa faktor lingkungan dan aktivitas manusia, seperti industrialisasi dan kepadatan penduduk, berkontribusi signifikan terhadap degradasi kualitas udara di wilayah tertentu.

---

### [Exploratory Data Analysis] - Menangani Missing Value dan Outliers

Dalam tahap awal pembersihan data, dilakukan pengecekan terhadap **duplikasi data** dan **missing value**. Hasilnya menunjukkan bahwa **tidak terdapat duplikasi data** maupun **missing value** di seluruh kolom fitur maupun target. Hal ini mengindikasikan bahwa dataset sudah lengkap dan tidak memerlukan teknik imputasi lebih lanjut.

| Kolom                         | Jumlah Missing Value |
| ----------------------------- | -------------------- |
| Temperature                   | 0                    |
| Humidity                      | 0                    |
| PM2.5                         | 0                    |
| PM10                          | 0                    |
| NO2                           | 0                    |
| SO2                           | 0                    |
| CO                            | 0                    |
| Proximity to Industrial Areas | 0                    |
| Population Density            | 0                    |
| Air Quality                   | 0                    |

Selanjutnya, dilakukan deteksi **outlier** menggunakan metode **Interquartile Range (IQR)** untuk setiap fitur numerik. Hasil analisis menunjukkan bahwa beberapa variabel memiliki jumlah outlier yang cukup signifikan, terutama pada **PM2.5 (352 outlier)** dan **PM10 (324 outlier)**. Variabel lain seperti **SO2 (124 outlier)**, **NO2 (73 outlier)**, dan **Temperature (72 outlier)** juga menunjukkan keberadaan data pencilan dalam jumlah yang cukup banyak. Bahkan fitur-fitur lain seperti **CO**, **Proximity to Industrial Areas**, dan **Population Density** juga mengandung outlier meskipun dalam jumlah yang lebih sedikit.

| Fitur                         | Jumlah Outlier |
| ----------------------------- | -------------- |
| Temperature                   | 72             |
| Humidity                      | 19             |
| PM2.5                         | 352            |
| PM10                          | 324            |
| NO2                           | 73             |
| SO2                           | 124            |
| CO                            | 45             |
| Proximity to Industrial Areas | 16             |
| Population Density            | 7              |

![image](https://github.com/user-attachments/assets/00e0f7b0-1a90-453f-82d2-3a8b4774f1eb)

Visualisasi melalui **boxplot** semakin memperjelas sebaran data dan keberadaan outlier di setiap fitur. Polutan udara seperti **PM2.5**, **PM10**, dan **SO2** tampak memiliki sebaran yang lebar dengan banyak data berada di luar whisker (batas IQR), yang mengindikasikan variasi nilai ekstrim dalam data tersebut.

Meskipun demikian, outlier **tidak dihapus** dari dataset. Hal ini dilakukan untuk menjaga **keutuhan informasi**, mengingat data pencilan tersebut mencerminkan kondisi nyata seperti lonjakan polusi di area industri atau wilayah dengan kepadatan penduduk tinggi. Menghilangkan outlier justru berisiko menghilangkan pola penting dalam konteks analisis kualitas udara.

Sebagai langkah mitigasi terhadap pengaruh outlier, proses **scaling** akan dilakukan menggunakan teknik **Robust Scaler**. Metode ini dipilih karena lebih tahan terhadap pengaruh outlier dibandingkan teknik standardisasi konvensional (seperti Min-Max atau Standard Scaler), sehingga mampu menyesuaikan skala data dengan lebih representatif tanpa terdistorsi oleh nilai ekstrim.

---
### [Exploratory Data Analysis] - Univariate Analysis

#### Grafik 1: Distribusi Kategori Kualitas Udara
![image](https://github.com/user-attachments/assets/a0fd9186-74c0-46d1-8224-d7e069b5a67f)

Grafik pertama menampilkan distribusi **kategori kualitas udara** yang dikelompokkan ke dalam empat kelas utama:

- **Good (Baik)**: Memiliki proporsi terbesar yaitu **40%** dari total sampel.
- **Moderate (Sedang)**: Menempati urutan kedua dengan **30%** dari total sampel.
- **Poor (Buruk)**: Mencakup **20%** dari total sampel.
- **Hazardous (Berbahaya)**: Menjadi kategori paling kecil dengan proporsi **10%**.

Dari distribusi ini, dapat disimpulkan bahwa sebagian besar wilayah dalam data (**70%**) memiliki kualitas udara yang tergolong **baik hingga sedang**, sedangkan **30% sisanya** menunjukkan kualitas udara yang **buruk hingga berbahaya**. Hal ini mengindikasikan adanya potensi risiko pencemaran udara di sejumlah wilayah tertentu.


#### Grafik 2: Distribusi dan KDE Faktor Lingkungan
![image](https://github.com/user-attachments/assets/d29abab1-8364-4230-9584-af6a295da968)

Grafik kedua menampilkan distribusi dan **Kernel Density Estimation (KDE)** dari berbagai **faktor lingkungan** yang berpotensi memengaruhi kualitas udara:

- **Temperature**: Menunjukkan distribusi normal dengan rentang dominan antara **25–35°C**, puncak sekitar **30°C**.
- **Humidity**: Distribusi cenderung normal, dengan mayoritas nilai berada pada kisaran **50–90%**, dan puncak sekitar **70%**.
- **PM2.5**: Distribusi menceng ke kanan (**right-skewed**), menunjukkan mayoritas wilayah memiliki kadar PM2.5 rendah (di bawah 50), namun terdapat beberapa nilai ekstrem tinggi.
- **PM10**: Serupa dengan PM2.5, distribusinya right-skewed, mayoritas di bawah 50, namun dengan beberapa nilai tinggi yang membentuk ekor panjang.
- **NO2**: Distribusi hampir normal, dengan konsentrasi utama antara **20–45**, dan puncaknya di kisaran **30–35**.
- **SO2**: Menceng ke kanan, dengan sebagian besar nilai berada di rentang **5–25**, puncaknya sekitar **10–15**.
- **CO**: Memiliki distribusi **bimodal**, dengan dua puncak pada sekitar **1.0** dan **1.6–1.7**, kemungkinan menunjukkan dua karakteristik lingkungan yang berbeda.
- **Proximity to Industrial Areas**: Distribusinya juga **bimodal**, dengan dua kelompok dominan pada jarak sekitar **5** dan **12**, mengindikasikan lokasi-lokasi yang dekat dan jauh dari kawasan industri.
- **Population Density**: Distribusinya cukup merata, dengan dominasi nilai pada rentang **300–600**, mencerminkan variasi tingkat kepadatan penduduk.

Dari distribusi yang ditampilkan, terlihat bahwa nilai-nilai polutan seperti PM2.5, PM10, SO2, dan CO memiliki distribusi menceng ke kanan atau bimodal, yang menandakan adanya nilai ekstrem pada sebagian wilayah. Ini berpotensi berkorelasi dengan wilayah-wilayah yang memiliki kualitas udara rendah (kategori poor dan hazardous). Selain itu, distribusi bimodal pada variabel jarak ke kawasan industri menunjukkan adanya dua kelompok lokasi yang kemungkinan besar mengalami tingkat paparan polusi yang berbeda, yang turut memengaruhi kualitas udara. Variasi suhu dan kelembaban juga dapat berdampak terhadap penyebaran dan reaksi kimia dari polutan di udara, yang selanjutnya memengaruhi kualitas udara secara keseluruhan di wilayah-wilayah tersebut.

---

### [Exploratory Data Analysis] - Multivariate Analysis

#### Grafik 1: Rata-rata Parameter Lingkungan per Kategori Kualitas Udara
![image](https://github.com/user-attachments/assets/c7ff760c-7cb1-4292-8526-eb048a37f367)

Grafik ini menunjukkan bahwa kualitas udara yang **semakin buruk** (dari Good ke Hazardous) cenderung diikuti oleh:

- **Kenaikan suhu dan kelembaban**
- **Peningkatan konsentrasi polutan** seperti PM2.5, PM10, NO2, SO2, dan CO
- **Penurunan jarak ke kawasan industri** (semakin dekat)
- **Peningkatan kepadatan penduduk**

Sebaliknya, kategori **Good** memiliki karakteristik lingkungan yang lebih ideal: suhu dan kelembaban lebih rendah, kadar polutan rendah, lokasi lebih jauh dari kawasan industri, dan kepadatan penduduk lebih rendah.

#### Grafik 2: Pairplot Parameter Numerik Berdasarkan Kategori Kualitas Udara
![image](https://github.com/user-attachments/assets/c1a5c7b7-ae6d-4fd3-9c3c-9b9759e64316)

Pairplot mengungkap beberapa pola penting:

- Korelasi kuat antara **PM2.5 dan PM10**
- **Temperature dan Humidity** berkorelasi positif dengan beberapa polutan
- **Jarak ke kawasan industri** berkorelasi negatif dengan kualitas udara
- **Kepadatan penduduk** berkorelasi positif dengan peningkatan polusi

Distribusi kategori pada pairplot juga memperlihatkan pemisahan yang jelas antara kualitas udara, terutama pada parameter polutan.

Kualitas udara dipengaruhi oleh kombinasi faktor lingkungan. **Kualitas buruk** cenderung terjadi di daerah dengan suhu dan kelembaban tinggi, polusi berat, lokasi dekat industri, dan kepadatan tinggi. Sebaliknya, **kualitas baik** ditemukan di daerah yang lebih bersih, lebih jauh dari industri, dan berpenduduk lebih jarang. Faktor **kedekatan dengan kawasan industri** menjadi pembeda paling signifikan dan menunjukkan perlunya pengawasan ketat terhadap emisi industri dalam perencanaan tata kota.

---

#### Grafik 3: Heatmap Matriks Korelasi antar Variabel Numerik
![image](https://github.com/user-attachments/assets/6c652f2e-9b46-4c6e-aad7-ef049bdaa72d)

Heatmap korelasi antar variabel numerik digunakan untuk mengidentifikasi hubungan linear antara faktor-faktor yang mempengaruhi kualitas udara.

**Korelasi positif tinggi:**
- **PM2.5 dan PM10** menunjukkan korelasi sangat kuat (**0.97**), menandakan keduanya cenderung meningkat secara bersamaan.
- **CO dan NO2** memiliki korelasi kuat (**0.71**), menunjukkan hubungan erat antara keduanya.
- **CO dan Temperature** berkorelasi cukup kuat (**0.69**), menunjukkan bahwa kadar CO cenderung naik saat suhu meningkat.

**Korelasi negatif:**
- **Proximity to Industrial Areas (PIA)** berkorelasi negatif terhadap semua variabel lain, terutama terhadap **CO (-0.71)**, menandakan semakin jauh dari kawasan industri, semakin rendah kadar polusi.
- Korelasi negatif juga terlihat antara **PIA dan Temperature (-0.59)**, serta **PIA dan NO2 (-0.61)**.

**Korelasi sedang:**
- **Population Density** berkorelasi positif sedang dengan **CO (0.59)** dan **NO2 (0.51)**, mengindikasikan bahwa area dengan kepadatan penduduk tinggi cenderung memiliki tingkat polutan lebih tinggi.
- **Temperature** juga berkorelasi sedang dengan **NO2 (0.59)** dan **SO2 (0.57)**.

---

### Analisis Faktor-Faktor yang Mempengaruhi Kualitas Udara
#### 1. Hasil Uji Statistik (ANOVA: F-score dan p-value)

Uji ANOVA dilakukan untuk mengukur sejauh mana masing-masing variabel mampu membedakan kategori kualitas udara secara statistik.

| Variabel                        | F-score   | p-value |
|---------------------------------|-----------|---------|
| CO                              | 8292.52   | 0.0000  |
| Proximity to Industrial Areas   | 3714.95   | 0.0000  |
| NO2                             | 2676.22   | 0.0000  |
| Temperature                     | 2191.94   | 0.0000  |
| SO2                             | 2018.30   | 0.0000  |
| Population Density              | 1189.93   | 0.0000  |
| Humidity                        | 1071.28   | 0.0000  |
| PM10                            | 745.43    | 0.0000  |
| PM2.5                           | 354.80    | 0.0000  |

Semua variabel menunjukkan p-value < 0.0001, yang berarti semuanya memiliki perbedaan yang signifikan secara statistik antar kategori kualitas udara.
> **ANOVA** digunakan Untuk mengukur signifikansi perbedaan rata-rata antar kategori kualitas udara bagi setiap fitur. F-score tinggi menandakan perbedaan yang sangat signifikan.

---

#### 2. Feature Importance berdasarkan Random Forest
![image](https://github.com/user-attachments/assets/718e76d9-e396-4814-8d79-aabcf8679206)

Model Random Forest digunakan untuk mengukur kontribusi masing-masing fitur dalam memprediksi kualitas udara. Hasilnya divisualisasikan dalam bentuk grafik batang (bar chart).

**Rangking Feature Importance:**
- **CO (0.34)** — fitur terpenting dalam model.
- **Proximity to Industrial Areas (0.29)** — menunjukkan pentingnya lokasi terhadap kualitas udara.
- **NO2 (0.10)** — juga berkontribusi besar.

Fitur lain seperti **SO2 (0.09)** dan **Temperature (0.08)** memberikan kontribusi sedang, sementara **PM2.5 (0.01)** dan **PM10 (0.02)** memberikan kontribusi kecil meskipun secara kesehatan tetap penting.

> **Random Forest** digunakan Karena mampu menangkap hubungan non-linear dan interaksi antar fitur, serta memberikan interpretasi yang baik tentang seberapa besar setiap fitur memengaruhi output prediksi.

- **CO** adalah indikator utama kualitas udara menurut seluruh pendekatan (korelasi, ANOVA, dan Random Forest).
- **Jarak ke kawasan industri** merupakan faktor eksternal paling berpengaruh, yang secara konsisten menunjukkan hubungan negatif dengan semua polutan.
- **NO2 dan SO2** adalah polutan gas yang perlu dipantau secara ketat.
- Faktor meteorologi seperti suhu dan kelembaban memiliki dampak sedang namun penting.
- **PM2.5 dan PM10**, meskipun kurang penting secara prediktif dalam model ini, tetap perlu diperhatikan karena dampaknya terhadap kesehatan telah banyak dibuktikan dalam studi epidemiologi.

---

## Data Preparation

### 1. Label Encoding dengan Mapping pada Fitur Target

Proses encoding dilakukan secara manual untuk fitur target **Air Quality** agar label merepresentasikan tingkatan kualitas udara secara logis (ordinal). Mapping digunakan sebagai berikut:

| Kategori Air Quality | Label |
|----------------------|-------|
| Good                 | 0     |
| Moderate             | 1     |
| Poor                 | 2     |
| Hazardous            | 3     |

Mapping manual digunakan agar urutan label pada variabel target mewakili tingkatan kualitas udara secara logis. Berbeda dengan `LabelEncoder` yang mengurutkan label berdasarkan alfabet, mapping memungkinkan kita mengatur sendiri skala ordinal sesuai konteks (misal: *Good < Moderate < Poor < Hazardous*). Ini penting agar model dapat memahami bahwa kualitas udara memiliki tingkatan berurutan, bukan sekadar kategori acak.

---

### 2. Reduksi Dimensi dengan Principal Component Analysis (PCA)

#### Justifikasi Reduksi Dimensi via PCA

Dari hasil analisis korelasi, disimpulkan bahwa hanya fitur **PM2.5** dan **PM10** yang memiliki tingkat korelasi sangat tinggi (**0.97**), menunjukkan informasi yang tumpang tindih. Oleh karena itu, PCA difokuskan pada kedua variabel ini untuk merangkum variasi data menjadi satu komponen utama. Fitur lainnya tetap dipertahankan secara individual agar karakteristik spesifik masing-masing variabel tetap terjaga.

#### Hasil PCA

| Komponen | Explained Variance Ratio |
|----------|---------------------------|
| PC1      | 98.7%                     |
| PC2      | 1.3%                      |

Berdasarkan hasil PCA terhadap variabel **PM2.5** dan **PM10**, satu komponen utama mampu menjelaskan sebesar **98.7%** variasi data keduanya. Ini menunjukkan bahwa informasi dari kedua variabel sangat tumpang tindih, sehingga dapat diringkas menjadi satu fitur utama tanpa kehilangan informasi penting. Reduksi dimensi ini bertujuan mengurangi redundansi data secara efektif dan menjaga efisiensi analisis.

---

### 3. Train-Test Split

| Dataset | Jumlah Sampel |
|---------|----------------|
| Total   | 5000           |
| Train   | 4000 (80%)     |
| Test    | 1000 (20%)     |

Dengan total 5000 data, membagi 80% untuk pelatihan dan 20% untuk pengujian memberikan cukup data bagi model untuk belajar dengan baik, sekaligus menyisakan data yang memadai untuk mengukur performa model secara objektif. Proporsi ini cukup umum digunakan karena menjaga keseimbangan antara kebutuhan generalisasi model dan efektivitas pelatihan.

---

### 4. Data Scaling dengan RobustScaler

Dalam dataset ini, beberapa variabel numerik terdeteksi memiliki **outliers** yang cukup signifikan. Penggunaan teknik standarisasi berbasis rata-rata dan standar deviasi seperti `StandardScaler` menjadi kurang tepat karena mudah terpengaruh oleh outliers, yang dapat menyebabkan skala data menjadi bias.

Oleh karena itu, digunakan **RobustScaler** yang melakukan standarisasi dengan memanfaatkan **median dan interquartile range (IQR)**, sehingga lebih tahan terhadap pengaruh nilai ekstrem. Pendekatan ini memastikan skala data lebih representatif tanpa distorsi dari outliers, dan membantu meningkatkan stabilitas model saat pelatihan.

---

## Model Development

Pada tahap pengembangan model, digunakan tiga algoritma klasifikasi yang umum dan efektif, yaitu K-Nearest Neighbors (KNN), Random Forest (RF), dan XGBoost (XGB). Pemilihan ketiga model tersebut didasarkan pada karakteristik masing-masing serta tujuan untuk membandingkan performa secara empiris.

### K-Nearest Neighbors (KNN)

```python
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
````
KNN merupakan algoritma sederhana yang mengklasifikasikan data berdasarkan kedekatan dengan tetangga terdekat dalam ruang fitur. Algoritma ini cocok sebagai baseline karena tidak memerlukan asumsi distribusi data yang kompleks. Jumlah tetangga yang dipertimbangkan dalam klasifikasi ditentukan sebanyak 5 sebagai nilai awal percobaan. Parameter ini dapat dioptimasi lebih lanjut untuk memperoleh performa terbaik.

### Random Forest (RF)

````python
rf_model = RandomForestClassifier(n_estimators=100, random_state=123)
rf_model.fit(X_train_scaled, y_train)
````
Random Forest adalah metode ensemble yang menggabungkan banyak pohon keputusan untuk meningkatkan akurasi dan mengurangi risiko overfitting. Model ini efektif dalam menangani dataset dengan fitur kompleks dan variabel yang banyak.

**Parameter:**

  * `n_estimators=100`
    Jumlah pohon keputusan yang dibangun sebanyak 100, dipilih sebagai kompromi antara akurasi dan waktu komputasi.
  * `random_state=123`
    Digunakan untuk memastikan hasil yang konsisten pada setiap eksekusi.

### XGBoost (XGB)

````python
xgb_model = XGBClassifier(n_estimators=100, random_state=123, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_scaled, y_train)
````
XGBoost adalah algoritma boosting yang populer karena efisiensi dan performanya dalam menangani data besar dan kompleks. Model ini secara iteratif meningkatkan kekuatan prediksi dengan memperbaiki kesalahan dari model sebelumnya.

**Parameter:**

  * `n_estimators=100`
    Menentukan jumlah iterasi pohon boosting yang akan dibuat.
  * `random_state=123`
    Untuk memastikan reproducibility.
  * `use_label_encoder=False`
    Menonaktifkan penggunaan label encoder bawaan agar sesuai dengan versi terbaru XGBoost.
  * `eval_metric='mlogloss'`
    Fungsi evaluasi log loss untuk klasifikasi multi-kelas.

Ketiga model ini digunakan dengan pengaturan parameter awal sebagai percobaan dasar

---

## Evaluasi Model

### Akurasi Model & Classification Report

Evaluasi performa model dilakukan menggunakan metrik akurasi serta classification report yang mencakup precision, recall, dan f1-score untuk masing-masing kelas. Ketiga model yang dievaluasi adalah K-Nearest Neighbors (KNN), Random Forest, dan XGBoost. Tujuan dari evaluasi ini adalah untuk mengukur kemampuan model dalam mengklasifikasikan data secara tepat dan konsisten pada berbagai kelas.

#### Akurasi Train dan Test

| Model          | Train Accuracy | Test Accuracy |
|----------------|----------------|---------------|
| KNN            | 0.9450         | 0.9140        |
| Random Forest  | 1.0000         | 0.9490        |
| XGBoost        | 1.0000         | 0.9450        |

Berdasarkan tabel di atas, Random Forest dan XGBoost mencapai akurasi sempurna pada data latih. Meskipun demikian, keduanya tetap menunjukkan generalisasi yang baik pada data uji dengan akurasi masing-masing sebesar 94.90% dan 94.50%. Sementara itu, KNN memiliki akurasi uji yang lebih rendah, yaitu 91.40%, namun tetap kompetitif sebagai model dasar. Visualisasi berupa bar chart juga dilampirkan untuk menunjukkan perbandingan akurasi ketiga model secara lebih jelas.

#### Classification Report

Classification report menampilkan evaluasi yang lebih mendetail terhadap performa model berdasarkan masing-masing kelas target. Tabel berikut memuat nilai precision, recall, dan f1-score dari ketiga model pada tiap kelas.

| Kelas | Precision (KNN) | Recall (KNN) | F1-Score (KNN) | Support | Precision (RF) | Recall (RF) | F1-Score (RF) | Support | Precision (XGB) | Recall (XGB) | F1-Score (XGB) | Support |
|-------|------------------|--------------|----------------|---------|----------------|-------------|----------------|---------|------------------|--------------|----------------|---------|
| 0     | 0.98             | 1.00         | 0.99           | 400     | 1.00           | 1.00        | 1.00           | 400     | 1.00             | 1.00         | 1.00           | 400     |
| 1     | 0.90             | 0.95         | 0.92           | 300     | 0.95           | 0.97        | 0.96           | 300     | 0.94             | 0.96         | 0.95           | 300     |
| 2     | 0.81             | 0.81         | 0.81           | 200     | 0.86           | 0.90        | 0.88           | 200     | 0.87             | 0.87         | 0.87           | 200     |
| 3     | 0.91             | 0.67         | 0.77           | 100     | 0.93           | 0.79        | 0.85           | 100     | 0.89             | 0.83         | 0.86           | 100     |
| **Accuracy**     |      -           |      -        | **0.91**       | 1000    |      -         |     -       | **0.95**       | 1000    |       -          |     -         | **0.94**       | 1000    |
| Macro Avg | 0.90             | 0.86         | 0.87           | 1000    | 0.93           | 0.91        | 0.92           | 1000    | 0.93             | 0.92         | 0.92           | 1000    |
| Weighted Avg | 0.91         | 0.91         | 0.91           | 1000    | 0.95           | 0.95        | 0.95           | 1000    | 0.94             | 0.94         | 0.94           | 1000    |

Precision menunjukkan proporsi prediksi benar dari semua prediksi yang diberikan model untuk suatu kelas. Nilai precision yang tinggi menandakan bahwa model jarang memberikan prediksi positif yang salah. Recall mengukur sejauh mana model dapat menangkap seluruh data positif yang sebenarnya; semakin tinggi recall, semakin sedikit kasus positif yang terlewat oleh model. F1-score merupakan rata-rata harmonis dari precision dan recall, yang memberikan gambaran menyeluruh tentang keseimbangan antara keduanya.

Hasil evaluasi menunjukkan bahwa model Random Forest dan XGBoost secara umum memiliki precision, recall, dan f1-score yang tinggi dan seimbang pada semua kelas. Kinerja terbaik terlihat pada kelas 0 dan 1, yang juga memiliki jumlah data yang lebih besar. Sementara itu, kelas 3 menunjukkan nilai recall yang lebih rendah pada model KNN, yang mengindikasikan bahwa model ini sering gagal mengidentifikasi kelas tersebut. Nilai f1-score juga relatif lebih rendah pada KNN dibandingkan dua model lainnya, terutama pada kelas dengan jumlah data yang lebih sedikit.

Secara keseluruhan, Random Forest menunjukkan performa terbaik dengan akurasi tertinggi dan metrik evaluasi yang stabil di semua kelas. XGBoost mengikuti dengan performa yang hampir serupa, sementara KNN tetap layak sebagai model pembanding meskipun memiliki keterbatasan pada beberapa kelas tertentu. Evaluasi ini dapat menjadi dasar dalam memilih model yang paling sesuai untuk diterapkan secara operasional, terutama bila mempertimbangkan trade-off antara akurasi dan kompleksitas model.

---

### Confusion Matrix
![image](https://github.com/user-attachments/assets/3e34060c-d64b-4427-a386-cebebdf29380)

Confusion matrix digunakan untuk mengevaluasi sejauh mana model dapat membedakan antar kelas berdasarkan prediksi dan label aktual. Dalam studi ini, terdapat empat kelas yang merepresentasikan kualitas udara, yaitu kelas 0 (Good), 1 (Moderate), 2 (Poor), dan 3 (Hazardous).

Model K-Nearest Neighbors (KNN) menunjukkan performa yang sangat baik dalam mengklasifikasikan kelas 0, dengan seluruh 400 sampel diprediksi secara akurat. Namun, model ini mengalami kesulitan dalam membedakan kelas 2 dan 3. Kelas 2, misalnya, mengalami salah klasifikasi yang cukup signifikan ke kelas 1 dan 3. Akurasi keseluruhan model KNN mencapai 91.4%.

Model Random Forest memperlihatkan kinerja klasifikasi yang lebih merata di seluruh kelas, dengan hanya sedikit kesalahan prediksi antar kelas yang berdekatan. Model ini berhasil mengklasifikasikan 399 dari 400 sampel pada kelas 0, serta 292 dari 300 sampel pada kelas 1. Meskipun terdapat beberapa kesalahan dalam prediksi kelas 2 dan 3, jumlahnya relatif kecil dibandingkan KNN. Akurasi keseluruhan model ini tercatat sebesar 94.9%.

Model XGBoost menunjukkan pola yang mirip dengan Random Forest dan memiliki akurasi keseluruhan sebesar 94.5%. Salah satu keunggulan utama XGBoost terlihat pada kemampuannya dalam mengklasifikasikan kelas 3 (Hazardous), dengan 83 dari 100 sampel berhasil diprediksi secara akurat. Hal ini menjadikan XGBoost model yang paling sensitif terhadap kelas yang paling kritis dari perspektif kesehatan publik.

Secara umum, semua model menunjukkan kecenderungan melakukan kesalahan klasifikasi pada kelas yang secara ordinal berdekatan. Misalnya, kelas 1 dan 2 cenderung saling tertukar dalam prediksi, yang menandakan bahwa model mampu mengenali kemiripan karakteristik antar kelas meskipun belum sepenuhnya optimal dalam membedakan batasannya.

---

### Grafik AUC-ROC
![image](https://github.com/user-attachments/assets/c0bbb248-f83a-43a2-a1b0-71eb47a444b6)

Kurva ROC (Receiver Operating Characteristic) digunakan untuk menggambarkan trade-off antara True Positive Rate dan False Positive Rate pada berbagai ambang keputusan. Nilai AUC (Area Under the Curve) memberikan ukuran keseluruhan dari performa model, di mana nilai lebih tinggi mengindikasikan model yang lebih baik dalam membedakan kelas.

Model Random Forest dan XGBoost sama-sama memperoleh nilai AUC sebesar 0.99, menandakan kemampuan klasifikasi yang sangat tinggi. Keduanya menunjukkan kurva ROC yang hampir berhimpit dan menjauhi garis diagonal, yang merupakan indikasi klasifikasi acak. Model KNN memperoleh AUC sebesar 0.97, yang juga sangat baik, namun sedikit lebih rendah dibandingkan dua model lainnya.

Perbedaan yang paling mencolok terdapat pada area awal kurva (False Positive Rate < 0.1), di mana Random Forest dan XGBoost mencapai True Positive Rate yang tinggi lebih cepat dibandingkan KNN. Hal ini menunjukkan bahwa kedua model tersebut lebih efektif dalam mendeteksi kasus positif dengan tingkat kesalahan minimum, khususnya pada skenario di mana kesalahan klasifikasi berdampak besar, seperti identifikasi kualitas udara buruk.

---

### Kesimpulan Evaluasi Model Klasifikasi

#### Akurasi Model

| Model         | Train Accuracy | Test Accuracy |
| ------------- | -------------- | ------------- |
| KNN           | 0.9450         | 0.9140        |
| Random Forest | 1.0000         | 0.9490        |
| XGBoost       | 1.0000         | 0.9450        |

Model Random Forest dan XGBoost menunjukkan akurasi sempurna pada data latih, mengindikasikan kemampuan untuk mempelajari pola dalam data dengan sangat baik. Namun, hal ini juga perlu diwaspadai karena berpotensi overfitting. Pada data uji, Random Forest tetap mempertahankan performa tertinggi dengan akurasi 94.9%, diikuti XGBoost (94.5%) dan KNN (91.4%). Meskipun KNN tidak seakurat dua model lainnya, performanya masih tergolong baik.

---

#### Classification Report (Data Uji)

| Kelas          | KNN (F1-score) | Random Forest (F1-score) | XGBoost (F1-score) |
| -------------- | -------------- | ------------------------ | ------------------ |
| Good           | 0.99           | 1.00                     | 1.00               |
| Hazardous      | 0.92           | 0.96                     | 0.95               |
| Moderate       | 0.81           | 0.88                     | 0.87               |
| Poor           | 0.77           | 0.85                     | 0.86               |
| **Macro Avg**  | 0.87           | 0.92                     | 0.92               |
| **Weighted Avg** | 0.91         | 0.95                     | 0.94               |

Random Forest dan XGBoost menunjukkan kinerja superior secara konsisten di seluruh kelas, terutama pada kelas yang lebih sulit seperti *Moderate* dan *Poor*. KNN cenderung unggul hanya di kelas *Good*, namun mengalami penurunan performa signifikan pada kelas lain, khususnya *Poor*.

---

#### AUC-ROC Score

| Model         | AUC Score |
| ------------- | --------- |
| KNN           | 0.97      |
| Random Forest | 0.99      |
| XGBoost       | 0.99      |

Nilai AUC-ROC menunjukkan bahwa Random Forest dan XGBoost memiliki kemampuan klasifikasi yang sangat tinggi di berbagai threshold, dengan AUC sebesar 0.99. Model KNN memiliki AUC sebesar 0.97, yang meskipun sedikit lebih rendah, masih termasuk dalam kategori performa yang sangat baik.

---

Secara umum, model **Random Forest** tampil sebagai model terbaik berdasarkan akurasi, F1-score per kelas, dan AUC-ROC. Model ini tidak hanya unggul dalam klasifikasi kelas mayoritas (*Good*), tetapi juga menunjukkan performa tinggi pada kelas minoritas yang lebih sulit seperti *Hazardous* dan *Poor*.

Model **XGBoost** menjadi alternatif kuat dengan performa yang hampir setara, khususnya unggul dalam identifikasi kelas *Hazardous*. Perbedaan performanya dibanding Random Forest sangat tipis dan masih dapat dipertimbangkan sesuai kebutuhan implementasi.

Model **KNN**, meskipun cukup baik dari sisi akurasi keseluruhan, menunjukkan kelemahan dalam menangani kelas *Moderate* dan *Poor*. Hal ini menunjukkan bahwa model ini kurang cocok untuk konteks di mana semua kelas harus diidentifikasi secara seimbang dan akurat.

Untuk implementasi sistem klasifikasi kualitas udara yang andal, Random Forest direkomendasikan sebagai pilihan utama, dengan XGBoost sebagai cadangan yang sangat kompetitif.

---

## Kesimpulan Akhir

Berdasarkan hasil eksplorasi data dan analisis fitur, dapat disimpulkan bahwa faktor paling berpengaruh terhadap tingkat kualitas udara di suatu wilayah adalah konsentrasi **CO**, **jarak ke kawasan industri**, dan **NO₂**. Ketiganya konsisten muncul sebagai fitur paling penting baik dalam analisis korelasi, uji ANOVA, maupun ranking feature importance dari model Random Forest. Fitur lain seperti **SO₂** dan **temperature** juga berkontribusi secara sedang, sedangkan **PM2.5** dan **PM10**, meskipun memiliki kontribusi kecil secara prediktif, tetap penting untuk diperhatikan karena dampak kesehatannya yang signifikan menurut literatur ilmiah. Temuan ini menjawab **Pernyataan Masalah 1**, bahwa variabel gas polutan (khususnya CO dan NO₂) serta faktor geografis seperti jarak ke kawasan industri merupakan penentu utama kualitas udara.

Untuk menjawab **Pernyataan Masalah 2**, proyek ini telah berhasil membangun model klasifikasi multikelas menggunakan tiga algoritma utama: KNN, Random Forest, dan XGBoost. Model dikembangkan untuk mengklasifikasikan kualitas udara ke dalam kategori *Good*, *Moderate*, *Poor*, dan *Hazardous* berdasarkan kombinasi fitur lingkungan dan demografis. Hasil evaluasi menunjukkan bahwa semua model mampu melakukan klasifikasi dengan akurasi tinggi, namun **Random Forest** unggul paling konsisten dengan test accuracy 94.9%, macro F1-score 0.92, dan AUC-ROC 0.99. XGBoost menyusul dengan performa hampir setara, sementara KNN sedikit tertinggal terutama dalam mengklasifikasikan kelas *Moderate* dan *Poor*.

Menjawab **Pernyataan Masalah 3**, evaluasi menyeluruh terhadap performa model menunjukkan bahwa Random Forest tidak hanya memiliki akurasi tinggi di data latih dan uji, tetapi juga memberikan generalisasi yang baik terhadap data baru. Model ini juga memberikan interpretabilitas melalui feature importance yang bermanfaat untuk kebijakan publik. Oleh karena itu, **Random Forest direkomendasikan sebagai model utama** dalam sistem prediksi kualitas udara yang andal, dengan **XGBoost** sebagai alternatif kuat yang layak dipertimbangkan sesuai konteks aplikasi. KNN dapat digunakan sebagai baseline model, namun kurang optimal untuk deployment pada sistem nyata yang membutuhkan prediksi seimbang di semua kategori kualitas udara.
