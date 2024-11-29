from textblob import TextBlob

def sentiment(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity

    if sentiment_score > 0:
        return 'Positif'
    elif sentiment_score < 0:
        return 'Negatif'
    else:
        return 'Netral'