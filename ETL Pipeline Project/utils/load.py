import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from google.oauth2 import service_account
from googleapiclient.discovery import build

load_dotenv()

def export_to_csv(df, filename="products.csv"):
    """Menyimpan data ke file CSV lokal dengan pengecekan dan log lebih lanjut."""
    try:
        if df.empty:
            print("[WARNING] DataFrame kosong, tidak ada data yang disimpan.")
            return
        df.to_csv(filename, index=False)
        print(f"[INFO] Data berhasil disimpan ke {filename}")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan data ke CSV: {e}")

def export_to_postgres(df):
    """Menyimpan data ke PostgreSQL dengan pengecekan koneksi database."""
    try:
        db_url = os.getenv("POSTGRES_URL")
        if not db_url:
            print("[ERROR] URL database PostgreSQL tidak ditemukan di environment variable.")
            return
        engine = create_engine(db_url)
        df.to_sql("products", engine, if_exists="replace", index=False)
        print("[INFO] Data berhasil disimpan ke PostgreSQL.")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan data ke PostgreSQL: {e}")

def export_to_gsheet(df, spreadsheet_id, sheet_name="Sheet1"):
    """Mengunggah data ke Google Sheets menggunakan API."""
    try:
        credentials_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
        if not credentials_path or not os.path.exists(credentials_path):
            print("[ERROR] Path file kredensial Google Sheets tidak ditemukan.")
            return
        
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        service = build('sheets', 'v4', credentials=credentials)

        values = [df.columns.to_list()] + df.values.tolist()

        body = {'values': values}

        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=sheet_name,
            valueInputOption="RAW",
            body=body
        ).execute()

        print("[INFO] Data berhasil diunggah ke Google Sheets.")
    except Exception as e:
        print(f"[ERROR] Gagal mengunggah data ke Google Sheets: {e}")
