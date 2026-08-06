# 202618001_kaushal_DS605 

## Data Scraping and Preprocessing using Python and Scrapy
# Books to Scrape — Web Scraping & Analysis

A small end-to-end pipeline that scrapes book data from [books.toscrape.com](https://books.toscrape.com), cleans it, engineers a few features, and visualizes the results.

## What it does

1. **Setup** — imports pandas, numpy, matplotlib, seaborn, and WordCloud.
2. **Scraping** — a Scrapy spider (`book_spider.py`) crawls the catalogue starting at page 1, follows pagination for up to 6 pages, and visits each book's detail page to collect:
   - title, category, price, rating, availability, description, UPC, number of reviews, and URL.
3. **Storing data** — runs the spider as a subprocess and saves the raw output to `scraped_data.csv`.
4. **Preprocessing** — loads the raw CSV and:
   - strips whitespace from string fields and drops duplicate UPCs
   - fills missing descriptions with a placeholder
   - converts price to a float and extracts a numeric stock count from the availability text
   - maps star-rating text (e.g. `"star-rating Three"`) to an integer 1–5
   - engineers three new columns:
     - `description_word_count` — word count of the description
     - `price_band` — price bucketed into Budget / Moderate / Expensive / Luxury
     - `value_score` — rating divided by price
   - saves the result to `cleaned_data.csv`
5. **Visualization** — a 2×2 grid of plots plus a word cloud:
   - price distribution (histogram)
   - rating distribution (count plot)
   - average price for the top 10 categories (bar plot)
   - price vs. rating (box plot)
   - a word cloud built from all book descriptions
6. **Summary** — a short written takeaway on price spread.

## Requirements

```
pandas
numpy
matplotlib
seaborn
wordcloud
scrapy
```

## Running it

Open the notebook and run the cells top to bottom. The scraping cell writes `book_spider.py` to disk and then invokes it via `scrapy runspider`, so it needs internet access and Scrapy installed in the active Python environment. Re-running the scrape cell removes any previous `scraped_data.csv` before creating a new one.

## Outputs

- `scraped_data.csv` — raw scraped records
- `cleaned_data.csv` — cleaned data with engineered features
- inline plots (price distribution, rating distribution, category price comparison, price/rating relationship, description word cloud)
