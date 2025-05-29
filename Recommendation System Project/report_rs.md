# Laporan Proyek Machine Learning  
## Food Based Ingredient Descriptions Recommendation System
**Rasendra Akbar Satyatama** — *MC004D5Y1124*

## Daftar Isi

- [Domain Proyek](#domain-proyek-komposisi-dan-kategori-makanan)
  - [Referensi](#referensi)
- [Business Understanding](#business-understanding)
  - [Problem Statements](#problem-statements)
  - [Goals](#goals)
  - [Solution Approach](#solution-approach)
- [Data Understanding](#data-understanding)
  - [Sumber Data](#sumber-data)
  - [Deskripsi Variabel](#exploratory-data-analysis---deskripsi-variabel)
  - [Missing Value & Duplikasi Data](#exploratory-data-analysis---identifikasi-missing-value-dan-duplikasi-data)
  - [Univariate Analysis](#exploratory-data-analysis---univariate-analysis)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
  - [Cosine Similarity](#cosine-similarity)
  - [Top N Recommendation](#top-n-recommendation)
- [Evaluation](#evaluation)
- [Kesimpulan Akhir](#kesimpulan-akhir)

---
## **Domain Proyek: Komposisi dan Kategori Makanan**

Industri makanan di era digital menghadirkan beragam pilihan menu yang berlimpah, sehingga memerlukan sistem rekomendasi yang dapat membantu pengguna menemukan makanan sesuai preferensi dengan cepat dan tepat. Namun, tantangan utama muncul ketika data interaksi pengguna seperti rating atau riwayat pesanan belum tersedia, yang dikenal sebagai cold-start problem. Untuk mengatasi hal ini, pendekatan content-based filtering menjadi solusi efektif karena memanfaatkan atribut makanan itu sendiri—seperti kategori, jenis makanan, dan deskripsi bahan untuk menghitung kemiripan antar makanan dan memberikan rekomendasi relevan tanpa bergantung pada data pengguna (Cheng et al., 2017). Dengan demikian, sistem dapat memberikan rekomendasi personal yang membantu pengguna menjelajahi pilihan makanan sesuai selera mereka secara efisien, terutama pada tahap awal pengembangan aplikasi kuliner.

---

### Referensi:

Cheng, Z., Ding, Y., Zhu, L., & Kankanhalli, M. (2017). _Food Recommendation: Framework, Existing Solutions and Challenges_.

---

## Business Understanding
### **Problem Statements**

Dalam dunia kuliner, terdapat dua tantangan utama yang diangkat dalam proyek ini:

1. **Bagaimana membangun sistem rekomendasi makanan** yang dapat menyarankan menu sesuai preferensi pengguna berdasarkan atribut seperti kategori makanan, jenis makanan (veg/non-veg), dan deskripsi bahan?

2. **Bagaimana memanfaatkan informasi yang terkandung dalam data makanan itu sendiri** untuk memberikan rekomendasi yang relevan, tanpa memerlukan data interaksi pengguna seperti rating atau histori pemesanan?


### **Goals**

Proyek ini memiliki dua tujuan utama:

1. **Mengembangkan sistem rekomendasi makanan berbasis content-based filtering** yang menggunakan atribut makanan seperti kategori, jenis, dan deskripsi bahan untuk menyarankan makanan yang sesuai dengan selera pengguna.

2. **Menyediakan solusi rekomendasi yang tetap efektif tanpa data pengguna**, dengan fokus pada analisis kesamaan antar item makanan, sebagai pendekatan yang sesuai dalam kondisi cold-start (minim interaksi pengguna).

### **Solution Approach**

* **Content-Based Filtering:**
  Sistem rekomendasi ini akan bekerja dengan menghitung kemiripan antar makanan berdasarkan atribut-atribut yang tersedia, terutama deskripsi bahan dan kategori makanan. Dengan pendekatan ini, jika seorang pengguna menyukai makanan berbahan dasar lemon atau termasuk kategori *Healthy Food*, maka sistem akan merekomendasikan makanan lain yang memiliki kemiripan atribut tersebut.

* **Tanpa Collaborative Filtering:**
  Pendekatan *collaborative filtering* tidak diterapkan karena tidak tersedianya data interaksi pengguna (seperti rating, histori pesanan, atau preferensi eksplisit). Selain itu, content-based filtering lebih cocok untuk tahap awal pengembangan sistem ketika fokus utama adalah pada informasi produk itu sendiri.

* **Evaluasi:**
  Sistem akan dievaluasi dengan metrik seperti *precision*, *recall*, *F1-Score* dan *top-N recommendation accuracy* untuk menilai relevansi hasil rekomendasi terhadap preferensi yang diketahui.

---

## Data Understanding
### Sumber Data
Data yang diperoleh berasal dari platform open source yaitu kaggle dan dapat diakses melalui link berikut.
[Food Recommendation System](https://www.kaggle.com/datasets/schemersays/food-recommendation-system)

### [Exploratory Data Analysis] - Deskripsi Variabel
Proyek ini menggunakan satu dataset utama yang berisi data makanan dengan atribut yang mendukung pembangunan sistem rekomendasi berbasis content-based filtering. Berikut adalah penjelasan detail dataset tersebut:

| **No** | **Nama Dataset** | **Jumlah Baris** | **Jumlah Kolom** | **Deskripsi**                                                                                                                        |
| ------ | ---------------- | ---------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1      | `food_data.csv`  | 400              | 5                | Dataset berisi informasi makanan yang mencakup ID, nama makanan, kategori makanan, jenis (veg/non-veg), dan deskripsi bahan makanan. |

#### **Detail Kolom pada `food_data.csv`**

| Kolom    | Tipe   | Deskripsi                                                                               |
| -------- | ------ | --------------------------------------------------------------------------------------- |
| Food\_ID | int64  | ID unik untuk setiap makanan                                                            |
| Name     | object | Nama makanan                                                                            |
| C\_Type  | object | Kategori makanan (misalnya Healthy Food, Snack, Dessert, Korean, Mexican, dll.)         |
| Veg\_Non | object | Jenis makanan berdasarkan kandungan hewani (veg = vegetarian, non-veg = non-vegetarian) |
| Describe | object | Deskripsi detail bahan-bahan yang digunakan dalam makanan tersebut                      |

---

### [Exploratory Data Analysis] - Identifikasi Missing Value dan Duplikasi Data

Dalam tahap awal pembersihan data, dilakukan pengecekan terhadap **duplikasi data** dan **missing value**. Hasilnya menunjukkan bahwa **tidak terdapat duplikasi data** maupun **missing value** di seluruh kolom fitur maupun target. Hal ini mengindikasikan bahwa dataset sudah lengkap.

| Kolom                         | Jumlah Missing Value |
| ----------------------------- | -------------------- |
| Food_ID                   | 0                    |
| Name                      | 0                    |
| C_Type                         | 0                    |
| Veg_Non                          | 0                    |
| Describe                           | 0                    |

---

### [Exploratory Data Analysis] - Univariate Analysis

#### Grafik 1: Distribusi Jenis Makanan (Veg / Non-Veg)
![image](https://github.com/user-attachments/assets/8ab3af07-42cf-4e59-b784-db2efb054acc)

Grafik pertama menampilkan distribusi jenis makanan berdasarkan kategori vegetarian dan non-vegetarian:

- Veg (Vegetarian): Memiliki proporsi terbesar yaitu 59.5% dari total sampel makanan.
- Non-Veg (Non-Vegetarian): Mencakup 40.5% dari total sampel makanan.

Dari distribusi ini, dapat disimpulkan bahwa makanan vegetarian mendominasi dataset dengan hampir 6 dari 10 makanan yang tersedia merupakan makanan vegetarian. Hal ini menunjukkan bahwa terdapat variasi yang cukup seimbang antara pilihan makanan vegetarian dan non-vegetarian, dengan sedikit kecenderungan ke arah makanan vegetarian.

#### Grafik 2: Frekuensi Kategori Makanan (C_Type)
![image](https://github.com/user-attachments/assets/4adeda45-b9cf-47ca-8dfc-d27af194a9c0)

Grafik kedua menampilkan frekuensi berbagai kategori makanan yang tersedia dalam dataset:

- Indian: Menjadi kategori paling dominan dengan jumlah sekitar 90 item, menunjukkan popularitas masakan India dalam dataset.
- Healthy Food: Menempati urutan kedua dengan sekitar 58 item.
- Dessert: Mencakup sekitar 53 item makanan penutup.
- Chinese, Italian, Snack, Thai: Masing-masing berkisar antara 22-27 item.
- French, Mexican, Japanese: Memiliki frekuensi sedang sekitar 18-22 item.
- Beverage, Nepalese, Korean, Vietnamese, Korean, Spanish: Menjadi kategori dengan frekuensi terendah, masing-masing kurang dari 12 item.

Dari distribusi ini terlihat bahwa dataset memiliki dominasi masakan Asia, khususnya India dan Chinese, diikuti dengan fokus yang cukup besar pada makanan sehat dan makanan penutup. Kategori makanan internasional lainnya seperti Eropa dan Amerika memiliki representasi yang lebih terbatas dalam dataset ini.

Tentu! Ini versi markdown lengkap termasuk **penjelasan**, **rumus**, **alasan**, dan **syntax code** yang kamu pakai:

---

## Data Preparation

Sebelum membangun sistem **content-based filtering**, data perlu dipersiapkan agar relevan untuk perhitungan kesamaan antar item. Pada tahap ini dilakukan beberapa langkah:

* **Menggabungkan kolom informasi penting:** Kolom `C_Type`, `Veg_Non`, dan `Describe` digabung menjadi satu kolom `content` agar bisa merepresentasikan deskripsi lengkap dari setiap item makanan.
* **Pembersihan teks:** Seluruh teks diubah menjadi huruf kecil (*lowercase*) dan dihapus tanda baca (*punctuation*) menggunakan fungsi `clean_text()`. Selanjutnya, spasi berlebihan dibersihkan menggunakan `clean_spaces()`.
* **Ekstraksi fitur dengan TF-IDF:** Setelah teks bersih, digunakan metode **TF-IDF (Term Frequency - Inverse Document Frequency)** untuk mengubah teks menjadi representasi numerik yang bisa digunakan model.

#### Rumus TF-IDF

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

di mana:

* $\text{TF}(t, d)$ = Frekuensi kemunculan term $t$ dalam dokumen $d$ dibagi total jumlah term dalam dokumen $d$.
* $\text{IDF}(t)$ = $\log \left( \frac{N}{n_t} \right)$, dengan:

  * $N$ = Total jumlah dokumen.
  * $n_t$ = Jumlah dokumen yang mengandung term $t$.

Dengan kata lain, **TF-IDF** memberikan bobot yang lebih tinggi pada kata-kata yang sering muncul di dokumen tertentu tetapi jarang muncul di seluruh koleksi dokumen, sehingga bisa menangkap kata-kata yang lebih spesifik dan penting. Penggunaan **TF-IDF** dalam content-based filtering memiliki beberapa alasan utama:

* Mengurangi pengaruh kata-kata umum yang muncul di hampir semua item.
* Memperkuat bobot kata-kata unik yang membedakan satu item dari yang lain.
* Menghasilkan representasi vektor yang memungkinkan kita menghitung kemiripan antar item dengan metode seperti **cosine similarity**.

Dengan pendekatan ini, kita bisa merekomendasikan makanan yang mirip berdasarkan deskripsi dan kategorinya, tanpa memerlukan data interaksi pengguna (seperti rating atau riwayat pembelian).

#### Syntax Code

```python
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Fungsi membersihkan teks
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def clean_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

# Gabungkan kolom jadi satu
df['content'] = df['C_Type'].astype(str) + ' ' + df['Veg_Non'].astype(str) + ' ' + df['Describe']

# Bersihkan teks
df['content_clean'] = df['content'].apply(clean_text)
df['content_clean'] = df['content_clean'].apply(clean_spaces)

# Hitung TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content_clean'])

# Output
print(tfidf.get_feature_names_out())
print(tfidf_matrix.shape)
```

Kode ini:
- Menghilangkan tanda baca dan mengubah teks menjadi huruf kecil
- Menghapus spasi berlebih
- Menggunakan `TfidfVectorizer` dari Scikit-learn untuk mengubah teks menjadi matriks angka
- Memfilter kata-kata umum dengan parameter `stop_words='english'` agar hanya kata penting yang diperhitungkan

---

## Modeling

### Cosine Similarity

Setelah representasi teks diubah menjadi matriks numerik menggunakan **TF-IDF**, tahap selanjutnya adalah menghitung tingkat kemiripan antar item untuk keperluan rekomendasi. Salah satu metode yang paling sering digunakan adalah **cosine similarity**. Metode ini digunakan berdasarka beberapa alasan berikut.

* **Independen dari panjang vektor:** Cosine similarity tidak bergantung pada panjang absolut dari vektor (misalnya panjang deskripsi), hanya pada sudut antar vektor. Ini cocok karena dua item yang mirip bisa saja memiliki panjang deskripsi berbeda.
* **Efektif untuk data sparce:** TF-IDF menghasilkan vektor yang sebagian besar kosong (sparse), dan cosine similarity mampu bekerja dengan baik dalam ruang berdimensi tinggi seperti ini.
* **Mengukur orientasi, bukan magnitude:** Dua item dengan istilah penting yang sama (meskipun skalanya berbeda) tetap dihitung mirip.

#### Rumus Cosine Similarity

$$
\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
$$

di mana:

* $A \cdot B$ = hasil perkalian dot product antara dua vektor $A$ dan $B$.
* $\|A\|$, $\|B\|$ = panjang (norma) masing-masing vektor.

Nilai cosine similarity berkisar antara:

* **1** → kedua vektor identik (sudut 0 derajat).
* **0** → kedua vektor tidak memiliki hubungan (sudut 90 derajat).
* **-1** → arah vektor sepenuhnya berlawanan (jarang terjadi pada TF-IDF karena nilainya non-negatif).

#### Output Matriks Cosine Similarity

Berikut hasil matriks cosine similarity antar item, hanya ditampilkan sebagian (dengan titik-titik di tengah):

```
[[1.         0.19757503 0.1292097  ... 0.12137388 0.03984156 0.16425487]
 [0.19757503 1.         0.06259989 ... 0.15124545 0.0677997  0.21651763]
 [0.1292097  0.06259989 1.         ... 0.03014219 0.10662772 0.02970139]
  ...
 [0.12137388 0.15124545 0.03014219 ... 1.         0.01768454 0.09281148]
 [0.03984156 0.0677997  0.10662772 ... 0.01768454 1.         0.        ]
 [0.16425487 0.21651763 0.02970139 ... 0.09281148 0.         1.        ]]
```

Pada matriks ini:

* Setiap baris/kolom merepresentasikan satu item makanan.
* Diagonal bernilai **1** karena setiap item tentu 100% mirip dengan dirinya sendiri.
* Nilai-nilai lainnya menunjukkan tingkat kemiripan antar item, misalnya:

  * **0.19757503** → mirip sekitar 19,8%
  * **0.03014219** → mirip sekitar 3%
  * **0** → tidak ada kemiripan

Dengan matriks ini, sistem rekomendasi bisa mencari item dengan nilai cosine similarity tertinggi untuk diberikan sebagai rekomendasi mirip.

---

### Top N Recommendation

Setelah mendapatkan matriks cosine similarity antar item, tahap berikutnya adalah membuat **rekomendasi Top-N**, yaitu mencari item-item dengan tingkat kemiripan tertinggi terhadap suatu item target.

#### Proses Pembuatan Rekomendasi

1. **Pilih item target:** Dalam kasus ini, item makanan dengan `Food_ID = 1`, yaitu **summer squash salad**.
2. **Ambil skor similarity:** Cari baris (atau kolom) terkait di matriks cosine similarity.
3. **Urutkan skor:** Urutkan semua item berdasarkan nilai similarity dari yang tertinggi ke terendah (selain dirinya sendiri, karena similarity dirinya pasti 1).
4. **Ambil Top-N:** Ambil *N* teratas — dalam contoh ini diambil **Top 8 rekomendasi**.

#### Pentingnya Top N Recommendation

* **Efisien:** Sistem tidak memerlukan data rating atau histori pengguna, cukup mengandalkan deskripsi konten.
* **Spesifik:** Hasil rekomendasi langsung merefleksikan kemiripan konten, cocok untuk sistem berbasis konten.
* **Fleksibel:** Bisa digunakan pada item baru (cold-start item) yang belum punya interaksi pengguna.

#### Top 8 Rekomendasi untuk Salah Satu Food_ID

* **Food\_ID:** 1
* **Nama makanan:** summer squash salad
* **Deskripsi bahan:** white balsamic vinegar, lemon juice, lemon rind, red chillies, garlic cloves (crushed), olive oil, summer squash (zucchini), sea salt, black pepper, basil leaves
* **Jenis (Veg/Non-Veg):** veg
* **Kategori (C\_Type):** Healthy Food

| Rank | Food\_ID | Name                                              | Veg\_Non | C\_Type      | Similarity\_Score |
| ---- | -------- | ------------------------------------------------- | -------- | ------------ | ----------------- |
| 1    | 164      | green cucumber shots                              | veg      | Healthy Food | 0.301098          |
| 2    | 17       | baked namakpara with roasted almond dip           | veg      | Snack        | 0.288072          |
| 3    | 70       | shepherds salad (tamatar-kheera salaad)           | veg      | Healthy Food | 0.286719          |
| 4    | 221      | amaranthus granola with lemon yogurt, berries ... | veg      | Healthy Food | 0.283219          |
| 5    | 144      | shrimp & cilantro ceviche                         | veg      | French       | 0.281902          |
| 6    | 379      | Grilled Chicken with Almond and Garlic Sauce      | non-veg  | Healthy Food | 0.273470          |
| 7    | 161      | spanish fish fry                                  | non-veg  | Mexican      | 0.270702          |
| 8    | 178      | oats shallots pulao                               | veg      | Healthy Food | 0.247507          |

Tabel ini menunjukkan makanan-makanan yang secara deskripsi, kategori, atau bahan memiliki kemiripan paling tinggi dengan **summer squash salad**, sehingga membantu pengguna menemukan pilihan serupa sesuai preferensi mereka.

---

## Evaluation

Pada tahap evaluasi, digunakan tiga metrik utama untuk mengukur performa sistem rekomendasi: **Precision\@8**, **Recall\@8**, dan **F1-Score**. Ketiga metrik ini dipilih karena mampu memberikan gambaran seberapa relevan dan seberapa lengkap hasil rekomendasi yang dihasilkan oleh sistem.

#### Alasan Penggunaan Metrik

* **Precision\@N** mengukur proporsi item relevan yang berhasil direkomendasikan dari total item yang direkomendasikan (top-N). Metrik ini fokus pada *ketepatan* sistem, penting untuk memastikan rekomendasi yang diberikan tidak asal-asalan.

* **Recall\@N** mengukur proporsi item relevan yang berhasil direkomendasikan dari seluruh item relevan yang tersedia di dataset. Metrik ini fokus pada *kelengkapan*, yaitu seberapa banyak item relevan yang berhasil ditemukan.

* **F1-Score** merupakan rata-rata harmonik antara precision dan recall, memberikan ukuran keseimbangan antara keduanya. Digunakan untuk mendapatkan penilaian menyeluruh yang mempertimbangkan baik aspek ketepatan maupun kelengkapan.

#### Rumus Metrik

$$
\text{Precision@N} = \frac{\text{Jumlah rekomendasi relevan di Top-N}}{N}
$$

$$
\text{Recall@N} = \frac{\text{Jumlah rekomendasi relevan di Top-N}}{\text{Total item relevan di dataset}}
$$

$$
\text{F1-Score} = 2 \times \frac{\text{Precision@N} \times \text{Recall@N}}{\text{Precision@N} + \text{Recall@N}}
$$

#### Hasil Evaluasi

| Metrik       | Nilai  |
| ------------ | ------ |
| Precision\@8 | 0.6250 |
| Recall\@8    | 0.0877 |
| F1-Score     | 0.1538 |

#### Kesimpulan Evaluasi

Berdasarkan hasil evaluasi, sistem rekomendasi menunjukkan nilai **Precision\@8** sebesar **0.6250**, yang berarti dari 8 item teratas yang direkomendasikan, sekitar 62,5% merupakan item yang relevan. Namun, **Recall\@8** bernilai relatif rendah, yaitu **0.0877**, yang menunjukkan bahwa sistem hanya berhasil menangkap sekitar 8,8% dari seluruh item relevan yang ada di dataset. Akibatnya, **F1-Score** juga berada di angka **0.1538**, mencerminkan ketidakseimbangan antara precision yang cukup baik dengan recall yang rendah.

---

## Kesimpulan Akhir

**Menjawab Problem Statement:**
Model rekomendasi berbasis konten ini berhasil menjawab problem statement dengan baik. Sistem mampu memberikan rekomendasi makanan yang relevan berdasarkan deskripsi bahan, kategori makanan (C\_Type), dan status vegetarian/non-vegetarian (Veg\_Non). Penggunaan TF-IDF dan cosine similarity memungkinkan sistem untuk mengukur kemiripan konten secara efektif tanpa memerlukan data pengguna lain, sesuai dengan kebutuhan sistem rekomendasi berbasis konten.

**Pencapaian Goals:**
Sistem ini mencapai goals yang ditetapkan, yaitu menghasilkan rekomendasi makanan yang relevan dan serupa dengan makanan yang dipilih pengguna. Dengan Precision\@8 sebesar 0.6250, sistem mampu merekomendasikan sekitar 62,5% makanan yang sesuai dalam 8 rekomendasi teratas. Namun, nilai Recall\@8 yang relatif rendah (0.0877) menunjukkan bahwa sistem hanya menemukan sebagian kecil dari seluruh makanan relevan yang tersedia, sehingga cakupan rekomendasi masih terbatas.

**Dampak Solusi Terhadap Bisnis:**
Solusi ini dapat meningkatkan pengalaman pengguna dengan menyediakan rekomendasi makanan yang sesuai preferensi, sehingga mendorong pengguna untuk lebih sering mencoba makanan baru yang relevan dengan selera mereka. Selain itu, sistem yang tidak bergantung pada data pengguna lain menjaga privasi sekaligus menyederhanakan implementasi rekomendasi.

**Saran:**
Meskipun hasil Precision cukup baik, nilai Recall dan F1-Score (0.1538) masih perlu ditingkatkan agar rekomendasi mencakup lebih banyak makanan relevan. Perbaikan bisa dilakukan dengan menambah fitur tambahan, seperti metadata nutrisi atau review pengguna, serta eksplorasi metode filtering hybrid untuk meningkatkan keberagaman dan relevansi rekomendasi.
