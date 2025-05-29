from utils.extract import extract_all_products
from utils.transform import process_products
from utils.load import export_to_csv, export_to_postgres, export_to_gsheet

def etl_pipeline():
    print("[START] ETL Pipeline is starting...")

    raw_df = extract_all_products(pages=50)
    if raw_df is None or raw_df.empty:
        print("[FAIL] Extraction failed or returned empty dataset.")
        return
    
    print(f"[SUCCESS] Retrieved {len(raw_df)} rows from website.")

    processed_df = process_products(raw_df)
    print(f"[INFO] Data transformed. {len(processed_df)} valid entries remaining.")

    export_to_csv(processed_df, filename="products.csv")
    export_to_postgres(processed_df)
    export_to_gsheet(processed_df, spreadsheet_id="1_NtdUl3hpwooryQOkQzvew8iBIL0enXCQh1np_HseNw")

    print("[COMPLETE] ETL job finished successfully.")

if __name__ == "__main__":
    etl_pipeline()
