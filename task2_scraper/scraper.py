import requests, re, time
import pandas as pd
from bs4 import BeautifulSoup

BASE = "https://books.toscrape.com/catalogue/page-{}.html"
RATINGS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def clean(text):
    return re.sub(r"\s+", " ", text).strip()

def scrape_page(page):
    for attempt in range(3):
        try:
            r = requests.get(BASE.format(page), timeout=30)
            break
        except requests.exceptions.RequestException:
            print(f"Page {page}: retry {attempt + 1}")
            time.sleep(3)
    else:
        return None
    if r.status_code != 200:
        return None
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    for b in soup.select("article.product_pod"):
        rows.append({
            "title": clean(b.h3.a["title"]),
            "price": float(re.sub(r"[^\d.]", "", b.select_one(".price_color").text)),
            "rating": RATINGS[b.select_one("p.star-rating")["class"][1]],
            "in_stock": "In stock" in b.select_one(".availability").text,
        })
    return rows

all_rows = []
for p in range(1, 51):
    rows = scrape_page(p)
    if not rows:
        break
    all_rows += rows
    print(f"Page {p}: {len(rows)} records")
    time.sleep(0.5)

df = pd.DataFrame(all_rows).drop_duplicates()
df.to_csv("books.csv", index=False)

import sqlite3
conn = sqlite3.connect("books.db")
df.to_sql("books", conn, if_exists="replace", index=False)
conn.close()

print(df.shape)
print(df.head())