from flask import Blueprint, render_template, flash, redirect, url_for, request, session
from .models import User
from . import db
from werkzeug.security import generate_password_hash, check_password_hash

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    data = request.form
    email = data.get('email')
    password = data.get('password')
    user = User.query.filter_by(email=email).first()
    if user:
        if check_password_hash(user.password, password):
            flash("Logged in successfully!", category="success")
            session['user_id'] = user.id
            session['user_email'] = user.email
            session['user_first_name'] = user.first_name
            session['user_role'] = user.role
            return redirect(url_for('views.home'))
        else:
            flash("Incorrect password, try again.", category="error")
    return render_template("login.html")

@auth.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_role', None)
    session.pop('user_email', None)
    session.pop('user_first_name', None)
    flash("Logged out successfully!", category="success")
    return redirect(url_for('auth.login'))

@auth.route('/sign-up-admin')
def sign_up_admin():
    new_user = User(email="admin@gmail.com", password=generate_password_hash("admin", method='pbkdf2:sha256'), first_name="Admin", role="admin")
    db.session.add(new_user)
    db.session.commit()
    flash("User created successfully!", category="success")
    return redirect(url_for('auth.login'))