import pytest
import pandas as pd
from utils.transform import process_products

def test_process_products():
    df = pd.DataFrame({
        "Title": ["Mock Product"],
        "Price": ["$20.0"],
        "Rating": ["Rating: 4.7 / 5"],
        "Colors": ["4 Colors"],
        "Size": ["Size: S"],
        "Gender": ["Gender: Female"],
        "Timestamp": ["2025-05-12 08:00:00"]
    })

    result = process_products(df)
    assert result['Price'].iloc[0] == 320000.0
    assert result['Rating'].iloc[0] == 4.7
    assert result['Colors'].iloc[0] == 4
    assert result['Size'].iloc[0] == "S"
    assert result['Gender'].iloc[0] == "Female"