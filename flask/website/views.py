from flask import Blueprint, render_template, flash, redirect, url_for, request
from .crawling.antara_news import crawl_antara_news
from .crawling.detik_oto import crawl_detik_oto
from .models import CrawlingData
from . import db
from .plot import create_figure_pemberitaan, create_bar_chart_media, create_pie_chart_author
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

views = Blueprint('views', __name__)

# @views.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         data = request.form
#         tanggal = data.get('tanggal')
#         # filter data by date
#         crawl_data = CrawlingData.query.filter_by(created_at=tanggal).all()
#         home_data = create_data_home(crawl_data)
#         return render_template("home.html", home_data=home_data, crawl_data=crawl_data)
        
#     crawl_data = CrawlingData.query.all()
#     home_data = create_data_home(crawl_data)
#     return render_template("home.html", home_data=home_data, crawl_data=crawl_data)

@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data = request.form
        tanggal = data.get('tanggal')
        # add time to date
        tanggal = tanggal + " 00:00:00"
        # filter data by date from tanggal to todays date
        crawl_data = CrawlingData.query.filter(CrawlingData.created_at >= tanggal).all()
        print(f"tanggal: {tanggal}")
        home_data = create_data_home(crawl_data)
    else:
        # Default view (without filtering)
        crawl_data = CrawlingData.query.all()
        home_data = create_data_home(crawl_data)

    # Create the plots using home_data
    plot_pemberitaan = create_figure_pemberitaan(home_data)  # Line chart for berita
    plot_media = create_bar_chart_media(home_data)  # Bar chart for media
    plot_author = create_pie_chart_author(home_data)  # Pie chart for author

    # Convert figures to images
    output_pemberitaan = io.BytesIO()
    FigureCanvas(plot_pemberitaan).print_png(output_pemberitaan)
    pemberitaan = base64.b64encode(output_pemberitaan.getvalue()).decode('utf8')
    
    output_media = io.BytesIO()
    FigureCanvas(plot_media).print_png(output_media)
    output_media = base64.b64encode(output_media.getvalue()).decode('utf8')
    
    output_author = io.BytesIO()
    FigureCanvas(plot_author).print_png(output_author)
    output_author = base64.b64encode(output_author.getvalue()).decode('utf8')

    # Pass images as base64 encoded strings to the template (optional for inline display)
    return render_template("home.html", home_data=home_data,
                            crawl_data=crawl_data,
                            plot_pemberitaan=pemberitaan,
                            plot_media=output_media,
                            plot_author=output_author)


@views.route('/crawl')
def crawl():
    # crawl_data = CrawlingData.query.all()
    # get crawl data from database order by news_date
    crawl_data = CrawlingData.query.order_by(CrawlingData.news_date.desc()).all()
    return render_template("crawl.html", crawl_data=crawl_data)

@views.route('/sentiment')
def sentiment():
    crawl_data = CrawlingData.query.all()
    return render_template("sentiment.html", crawl_data=crawl_data)

@views.route('/crawl_add', methods=['GET', 'POST'])
def crawl_add():
    data = request.form
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    media = data.getlist('media')
    keyword = data.getlist('keyword')
    antara = []
    detik = []
    if 'antara' in media:
        for key in keyword:
            data = crawl_antara_news(key, start_page=1, end_page=5, start_date_str=start_date, end_date_str=end_date)
            antara.append(data)
    if 'detikoto' in media:
        for key in keyword:
            data = crawl_detik_oto(key, start_page=1, end_page=5, start_date_str=start_date, end_date_str=end_date)
            detik.append(data)

    if antara or detik:
        flash("Crawling data successfully!", category="success")
        return redirect(url_for('views.crawl'))
    else:
        flash("Failed to crawl data.", category="error")

    return render_template("crawladd.html")


def create_data_home(crawl_data):
    # sum of news value
    sum_news_value = 0
    for data in crawl_data:
        sum_news_value += data.news_value
    
    # count of news
    count_news = len(crawl_data)

    # sum news group by created_at
    created_at = {}
    for data in crawl_data:
        if data.created_at in created_at:
            created_at[data.created_at] += 1
        else:
            created_at[data.created_at] = 1

    # sum news group by media
    media = {}
    for data in crawl_data:
        if data.media in media:
            media[data.media] += 1
        else:
            media[data.media] = 1
    
    # sum news group by author select only top 5
    author = {}
    for data in crawl_data:
        if data.author in author:
            author[data.author] += 1
        else:
            author[data.author] = 1
    author = dict(sorted(author.items(), key=lambda item: item[1], reverse=True)[:5])

    # create dictionary
    data_home = {
        'created_at': created_at,
        'media': media,
        'author': author,
        'sum_news_value': sum_news_value,
        'count_news': count_news
    }

    return data_home
    