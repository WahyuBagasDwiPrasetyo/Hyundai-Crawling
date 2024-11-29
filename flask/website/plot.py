from flask import Blueprint, render_template, flash, redirect, url_for, request, Response
import io
from .models import CrawlingData
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from . import db

plot = Blueprint('plot', __name__)

@plot.route('/plot_pemberitaan.png')
def plot_pemberitaan_png():
    fig = create_figure_pemberitaan()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@plot.route('/plot_bar_media.png')
def plot_bar_media_png():
    fig = create_bar_chart_media()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@plot.route('/plot_pie_penulis.png')
def plot_pie_penulis_png():
    fig = create_pie_chart_author()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

# def create_bar_chart_media():
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
    
#     # Sample data points (Replace these with your actual data)
#     categories = ['Kompas', 'Detik.com', 'Oto Driver', 'CNN', 'IDNTimes', 'Liputan6', 'Kumparan', 'Pikiran Rakyat', 'tempo.co', 'Antara News']
#     values = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    
#     # Plotting the horizontal bar chart
#     axis.barh(categories, values, color='skyblue')
    
#     # Setting titles for clarity
#     axis.set_title('Jumlah Berita')
#     axis.set_xlabel('Count')
#     axis.set_ylabel('News Source')

#     return fig

# def create_figure_pemberitaan():
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
    
#     # Sample data points (Replace these with your actual data)
#     days = [1, 7, 14, 21, 28]
#     values = [1000, 6000, 2000, 7000, 11000]
    
#     # Plotting the line chart
#     axis.plot(days, values)
    
#     # Filling the area under the line chart
#     axis.fill_between(days, values, alpha=0.3)

#     # Setting titles for clarity (as seen in the image)

#     return fig

# def create_pie_chart_author():
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
    
#     # Sample data points (Replace these with your actual data)
#     labels = ['Dina Rayanti', 'Septian farhan Nurhuda', 'Rangga Rahadiansyah', 'Arief Aszhari']
#     sizes = [15, 22, 10, 18]
#     colors = ['yellow', 'orange', 'purple', 'blue']
    
#     # Plotting the pie chart
#     axis.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    
#     # Equal aspect ratio ensures that pie is drawn as a circle.
#     axis.axis('equal')

#     return fig

def create_figure_pemberitaan(home_data):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    
    created_at = home_data['created_at']  # This is the data from create_data_home
    # print(f"created_at: {created_at}")
    # value of created_at created_at: {datetime.datetime(2024, 11, 10, 16, 44, 1): 20}
    # turn the datetime object into date object and get only day
    created_at = {date.strftime('%Y-%m-%d'): count for date, count in created_at.items()}
    # print(f"created_at: {created_at}")

    days = list(created_at.keys())
    values = list(created_at.values())

    # Plotting the line chart
    axis.plot(days, values)
    
    # Filling the area under the line chart
    axis.fill_between(days, values, alpha=0.3)

    # Setting titles for clarity
    axis.set_title('Jumlah Berita per Tanggal')
    axis.set_xlabel('Tanggal')
    axis.set_ylabel('Jumlah Berita')

    return fig

def create_bar_chart_media(home_data):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    
    media = home_data['media']  # This is the data from create_data_home
    print(f"media: {media}")
    # value of media media: {'detikoto': 20}

    categories = list(media.keys())
    values = list(media.values())

    # Plotting the horizontal bar chart
    axis.barh(categories, values, color='skyblue')
    
    # Setting titles for clarity
    axis.set_title('Jumlah Berita per Media')
    axis.set_xlabel('Jumlah Berita')
    axis.set_ylabel('Media')

    return fig

def create_pie_chart_author(home_data):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    
    author = home_data['author']  # This is the data from create_data_home
    print(f"author: {author}")
    # value of author author: {'Ridwan Arifin -detikOto': 8, 'Septian Farhan Nurhuda -detikOto': 7, 'Luthfi Anshori -detikOto': 2, 'Rangga Rahadiansyah -detikOto': 1, 'M Luthfi Andika -detikOto': 1}

    # Prepare data for pie chart
    labels = list(author.keys())
    sizes = list(author.values())
    colors = ['yellow', 'orange', 'purple', 'blue', 'red']
    
    # Plotting the pie chart
    axis.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    axis.axis('equal')

    return fig
