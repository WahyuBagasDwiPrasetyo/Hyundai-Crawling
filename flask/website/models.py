from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

class CrawlingData(db.Model):
    __tablename__ = 'crawling_data'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150))
    link = db.Column(db.String(150))
    created_at = db.Column(db.DateTime, server_default=func.now())
    author = db.Column(db.String(150))
    news_value = db.Column(db.Integer)
    detail = db.Column(db.String(150))
    summary = db.Column(db.String(150))
    media = db.Column(db.String(150))
    description = db.Column(db.String(150))
    news_date = db.Column(db.DateTime)
    sentiment = db.Column(db.String(150))
    bookmark = db.Column(db.Boolean, default=False)

class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    role = db.Column(db.String(150))