import aiohttp
import asyncio
import lxml.html
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from transformers import T5ForConditionalGeneration, T5Tokenizer
from .sentiment import sentiment
import requests_cache
from website.models import CrawlingData
from website import db
from datetime import datetime

# Initialize T5 model and tokenizer for summarization
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)

# Setup caching to avoid repeated requests
requests_cache.install_cache('detik_cache', expire_after=1800)  # Cache expires after 30 minutes

# Asynchronous function to fetch page content
async def fetch_page(session, url):
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

# Asynchronous function to scrape article details
async def scrape_article(session, link, article_date):
    try:
        detail_html = await fetch_page(session, link)
        if detail_html:
            detail_soup = lxml.html.fromstring(detail_html)
            # Extract date
            # date = detail_soup.xpath('//div[@class="detail__date"]/text()')
            # date = date[0].strip() if date else 'Date not found'
            # Extract author
            author = detail_soup.xpath('//div[@class="detail__author"]/text()')
            author = author[0].strip() if author else 'Author not found'
            # Extract content
            paragraphs = detail_soup.xpath('//div[@class="detail__body-text itp_bodycontent"]/p')
            content = ' '.join([p.text_content().strip() for p in paragraphs[:5] if 'para_caption' not in p.attrib.get('class', [])])
            return article_date, author, content
        else:
            return 'Date not found', 'Author not found', 'Content not found'
    except Exception as e:
        print(f"Error scraping {link}: {e}")
        return 'Date not found', 'Author not found', 'Content not found'

# Function to summarize text
def summarize_text(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Asynchronous function to scrape a single page of articles
async def scrape_page(session, keyword, page, data, start_date, end_date):
    url = f'https://www.detik.com/search/searchall?query={keyword}&page={page}'
    page_html = await fetch_page(session, url)
    if not page_html:
        return
    
    tree = lxml.html.fromstring(page_html)
    articles = tree.xpath('//div[@class="media media--right media--image-radius block-link"]')

    tasks = []
    for article in articles:
        title_tag = article.xpath('.//h3[@class="media__title"]/a')
        if title_tag:
            title = title_tag[0].text.strip()
            link = title_tag[0].attrib['href']
            # if in the link there's no oto.detik.com, skip it
            if 'https://oto.detik.com/' not in link:
                continue

            #filter date
            # <span d-time="1706270411" title="Jumat, 26 Jan 2024 19:00 WIB">Jumat, 26 Jan 2024 19:00 WIB</span>
            # get d-time attribute value
            date = article.xpath('.//span[@d-time]/@d-time')
            date = date[0].strip() if date else 'Date not found'

            # Convert date to datetime object
            try:
                date = datetime.fromtimestamp(int(date))
            except ValueError:
                continue

            # Check if the article date is within the specified range
            if date:
                if start_date and date < start_date:
                    continue
                if end_date and date > end_date:
                    continue

            tasks.append(scrape_article(session, link, date))

    article_details = await asyncio.gather(*tasks)

    # for title, (date, author, content) in zip([article.xpath('.//h3[@class="media__title"]/a/text()')[0] for article in articles], article_details):
    for title, link, (date, author, content) in zip(
        [article.xpath('.//h3[@class="media__title"]/a/text()')[0] for article in articles], 
        [article.xpath('.//h3[@class="media__title"]/a/@href')[0] for article in articles], 
        article_details):
        sentiment_score = sentiment(content)
        summary = summarize_text(content)
        data.append({
            'Judul': title,
            'Tanggal': date,
            'Author': author,
            'Link': link,
            'Detail': content,
            'Sentiment': sentiment_score,
            'Ringkasan': summary
        })

# Main function to crawl data across multiple pages
def crawl_detik_oto(keyword, start_page=1, end_page=2, start_date_str = "", end_date_str = ""):
    # Initialize list to hold scraped data
    data = []

    # Convert start and end dates to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None

    # Create an asyncio session
    async def main():
        async with aiohttp.ClientSession() as session:
            tasks = []
            for page in range(start_page, end_page + 1):
                tasks.append(scrape_page(session, keyword, page, data, start_date, end_date))
            await asyncio.gather(*tasks)

    # Run the asyncio event loop
    asyncio.run(main())

    # Convert the data into a DataFrame and save it to JSON
    df = pd.DataFrame(data)

    # Bulk insert data into the database
    batch_data = []
    for index, row in df.iterrows():
        new_data = CrawlingData(
            title=row['Judul'], 
            link=row['Link'], 
            author=row['Author'], 
            news_value=52500000, 
            detail=row['Detail'], 
            summary=row['Ringkasan'], 
            media='detik oto', 
            description='Description', 
            news_date=row['Tanggal'],
            sentiment=row['Sentiment']
        )
        batch_data.append(new_data)
    
    # Add to DB in bulk and commit once
    db.session.bulk_save_objects(batch_data)
    db.session.commit()

    return df

