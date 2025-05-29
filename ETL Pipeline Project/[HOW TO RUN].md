### 1. **Membuat Virtual Environment**

Langkah pertama adalah membuat virtual environment di proyek ini agar dependensi tidak mengganggu instalasi lain di sistem.

Buka terminal atau command prompt, lalu jalankan perintah berikut untuk membuat virtual environment:

```bash
python -m venv venv
```
Setelah itu, aktifkan virtual environment sesuai dengan sistem operasi yang digunakan:

* **Windows**:

  ```bash
  .\venv\Scripts\activate
  ```

* **MacOS/Linux**:

  ```bash
  source venv/bin/activate
  ```

### 2. **Install Dependensi dari `requirements.txt`**

Setelah virtual environment aktif, install semua dependensi yang diperlukan untuk proyek ini dengan perintah:

```bash
pip install -r requirements.txt
```
Ini akan menginstal semua paket yang tercantum dalam file `requirements.txt`.

### 3. **Jalankan `main.py`**

Setelah dependensi terpasang, jalankan script utama proyek untuk menjalankan ETL pipeline dengan perintah:

```bash
python main.py
```
Ini akan menjalankan proses ETL dan menyimpan data ke PostgreSQL serta Google Sheets (jika sudah dikonfigurasi dengan benar).

### 4. **Jalankan Unit Tests dengan `pytest`**

Untuk memastikan semua fungsi bekerja dengan baik, jalankan unit tests yang ada di folder `tests` dengan menggunakan `pytest`. Jalankan perintah berikut:

```bash
set PYTHONPATH=.
pytest tests
```