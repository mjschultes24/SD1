from flask import render_template, url_for, flash, redirect
from flaskblog import app

if __name__ == '__main__':
    app.run(debug=True) ##__name__ is main when running in python, this is only true when we run the script directly
