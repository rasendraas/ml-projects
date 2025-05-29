import pytest
import requests
import pandas as pd
from utils.extract import extract_all_products

class DummyResponse:
    def __init__(self, text):
        self.text = text
    def raise_for_status(self):
        pass

@pytest.fixture
def sample_html():
    return '''
    <div class="collection-card">
        <h3 class="product-title">Mock Product</h3>
        <div class="price-container"><span class="price">$15.00</span></div>
        <p>Rating: 4.8 / 5</p>
        <p>5 Colors</p>
        <p>Size: L</p>
        <p>Gender: Women</p>
    </div>
    '''

def test_extract_all_products(monkeypatch, sample_html):
    def dummy_get(url, timeout=None):
        return DummyResponse(sample_html)

    monkeypatch.setattr(requests, 'get', dummy_get)

    df = extract_all_products(pages=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]['Title'] == 'Mock Product'