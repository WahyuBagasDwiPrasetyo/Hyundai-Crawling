from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
import os

db = SQLAlchemy()
DB_NAME = "database.db"

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secrethjujsyakd'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_FILE_DIR'] = './.flask_session/'
    app.config['SESSION_PERMANENT'] = False

    Session(app)
    db.init_app(app)

    from .views import views
    from .auth import auth
    from .user import user
    from .plot import plot

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    app.register_blueprint(user, url_prefix='/user')
    app.register_blueprint(plot, url_prefix='/plot')

    from .models import User, CrawlingData
    create_db(app)

    return app

def create_db(app):
    db_path = os.path.join('instance', DB_NAME)
    print(f"Checking if database exists at {db_path}")
    if not os.path.exists(db_path):
        print("Database does not exist. Creating...")
        with app.app_context():
            db.create_all()
            print(f'Created Database in {os.path.abspath(db_path)}')
    else:
        print("Database already exists.")