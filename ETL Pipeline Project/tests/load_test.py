import pandas as pd
from utils.load import export_to_csv

def test_export_to_csv(tmp_path):
    df = pd.DataFrame({
        "Title": ["Mock"],
        "Price": [16000],
        "Rating": [4.0],
        "Colors": [2],
        "Size": ["M"],
        "Gender": ["Unisex"],
        "Timestamp": ["2025-05-12 08:00:00"]
    })
    path = tmp_path / "output.csv"
    export_to_csv(df, filename=str(path))
    assert path.exists()