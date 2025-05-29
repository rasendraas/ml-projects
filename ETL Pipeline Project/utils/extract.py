import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def get_products_from_page(page_num):
    base_url = "https://fashion-studio.dicoding.dev/"
    url = base_url if page_num == 1 else f"{base_url}page{page_num}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Gagal mengakses halaman {page_num}: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")
    items = soup.select("div.collection-card")

    results = []
    for item in items:
        try:
            title_tag = item.find("h3", class_="product-title")
            price_tag = item.find("span", class_="price")
            title = title_tag.text.strip() if title_tag else "Unknown Product"
            price = price_tag.text.strip() if price_tag else "Price Unavailable"

            details = {}
            for p in item.find_all("p"):
                if ":" in p.text:
                    parts = p.text.split(":")
                    key = parts[0].strip()
                    value = parts[1].strip()
                    details[key] = value

            results.append({
                "Title": title,
                "Price": price,
                "Rating": details.get("Rating"),
                "Colors": details.get("Colors"),
                "Size": details.get("Size"),
                "Gender": details.get("Gender"),
                "FetchedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            print(f"[WARN] Produk gagal diproses: {e}")

    return pd.DataFrame(results)

def extract_all_products(pages=50):
    all_data = []
    for i in range(1, pages + 1):
        print(f"[INFO] Mengambil halaman {i}")
        df = get_products_from_page(i)
        if df.empty:
            print(f"[STOP] Halaman {i} kosong, berhenti scraping.")
            break
        all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True).drop_duplicates()
    print(f"[RESULT] Total produk unik terkumpul: {len(final_df)}")
    return final_df
