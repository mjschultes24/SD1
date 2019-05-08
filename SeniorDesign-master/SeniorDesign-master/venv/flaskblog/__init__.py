from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
#from flask_login import LoginManager
import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'cf56a7dbe6befc17221b02d868b96724'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
# db = SQLAlchemy(app)

from flaskblog import routes

