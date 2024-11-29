import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from website.models import CrawlingData
from website import db

def crawl_detik_oto(keyword, start_page=1, end_page=2, output_file='data_antara.json'):
    # Initialize a list to store data
    data = []

    # Loop to scrape data from multiple pages
    for page in tqdm(range(start_page, end_page + 1), desc="Scraping Pages", unit="page"):
        url = f'https://www.detik.com/search/searchall?query={keyword}&page={page}'
        print(url)
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('div', class_='media media--right media--image-radius block-link')

        for article in tqdm(articles, desc="Scraping Articles", unit="article", leave=False):
            media_text = article.find('div', class_='media__text')
            title_tag = media_text.find('h3', class_='media__title')
            if title_tag:
                link_tag = title_tag.find('a')
                title = link_tag.get_text(strip=True) if link_tag else 'Title not found'
                link = link_tag['href'] if link_tag else 'Link not found'

                try:
                    detail_response = requests.get(link)
                    detail_response.raise_for_status()
                except requests.RequestException as e:
                    print(f"Error fetching {link}: {e}")
                    content = 'Content not found'
                else:
                    detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                    article_tag = detail_soup.find('article', class_='detail')
                    # get date
                    date_div = article_tag.find('div', class_='detail__date')
                    date = date_div.get_text(strip=True) if date_div else 'Date not found'

                    # get author
                    author_div = article_tag.find('div', class_='detail__author')
                    author = author_div.get_text(strip=True) if author_div else 'Author not found'

                    # content article
                    detail_div = article_tag.find('div', class_='detail__body itp_bodycontent_wrapper')
                    content_div = detail_div.find('div', class_='detail__body-text itp_bodycontent')
                    content = content_div.get_text(strip=True) if content_div else 'Content not found'

                data.append({'Judul': title, 'Tanggal': date, 'Author': author, 'Link': link, 'Detail': content})

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Load T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def summarize_text(text):
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Apply summarization to the 'Detail' column
    df['Ringkasan'] = df['Detail'].apply(summarize_text)

    # Select all columns
    output_df = df[['Judul', 'Tanggal', 'Author', 'Link', 'Detail', 'Ringkasan']]

    # Save data to database
    for index, row in output_df.iterrows():
        new_data = CrawlingData(title=row['Judul'], link=row['Link'], author=row['Author'], news_value=52500000, detail=row['Detail'], summary=row['Ringkasan'], media='detikoto', description='Description', news_date=row['Tanggal'])
        db.session.add(new_data)
        db.session.commit()

    return output_df
