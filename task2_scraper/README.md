# Task 2: Automated Web Scraping Pipeline

**Target:** books.toscrape.com (public practice site)
**Tools:** Python, Requests, BeautifulSoup, Pandas, SQLite

## How it works
1. Loops through 50 catalogue pages with retries and a 30s timeout
2. Extracts title, price, rating and stock status
3. Cleans whitespace, converts price to float and rating to int
4. Removes duplicates and saves to `books.csv` and `books.db`

## Run
pip install requests beautifulsoup4 pandas
python scraper.py

## Output
1000 records, columns: title, price, rating, in_stock