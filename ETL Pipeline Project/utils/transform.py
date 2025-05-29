import pandas as pd

def process_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformasi dan pembersihan data produk untuk ETL pipeline.
    """
    try:
        valid_df = df[df['Title'].str.lower() != 'unknown product'].copy()

        valid_df['Price'] = (
            valid_df['Price']
            .str.replace("$", "", regex=False)
            .replace("Price Unavailable", "0")
            .astype(float)
            .mul(16000)
        )

        valid_df['Rating'] = (
            valid_df['Rating']
            .str.extract(r"(\d+(\.\d+)?)")[0]
            .astype(float)
        )

        valid_df['Colors'] = (
            valid_df['Colors']
            .str.extract(r"(\d+)")[0]
            .astype(float)
            .fillna(0)
            .astype(int)
        )

        valid_df['Size'] = (
            valid_df['Size']
            .str.replace("Size:", "", regex=False)
            .str.strip()
        )

        valid_df['Gender'] = (
            valid_df['Gender']
            .str.replace("Gender:", "", regex=False)
            .str.strip()
        )

        final_df = valid_df.drop_duplicates().dropna()

        return final_df

    except Exception as e:
        print(f"[ERROR] Data transformation failed: {e}")
        return pd.DataFrame()