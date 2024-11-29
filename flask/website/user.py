from flask import Blueprint, render_template, flash, redirect, url_for, request, session
from .models import User
from . import db
from werkzeug.security import generate_password_hash, check_password_hash

user = Blueprint('user', __name__)

@user.route('/')
def index():
    user_data = User.query.all()
    return render_template("user/index.html", user_data=user_data)

@user.route('/add')
def add():
    return render_template("user/add.html")

@user.route('/edit/<int:id>')
def edit(id):
    user = User.query.filter_by(id=id).first()
    return render_template("user/edit.html", user=user)

@user.route('/create', methods=['POST'])
def create():
    data = request.form
    email = data.get('email')
    password = data.get('password')
    first_name = data.get('first_name')
    role = data.get('role')

    user = User(email=email, password=generate_password_hash(password, method='pbkdf2:sha256'), first_name=first_name, role=role)
    db.session.add(user)
    db.session.commit()

    flash("User added successfully!", category="success")
    return redirect(url_for('user.index'))

@user.route('/update/<int:id>', methods=['POST'])
def update(id):
    data = request.form
    email = data.get('email')
    password = data.get('password')
    first_name = data.get('first_name')
    role = data.get('role')

    user = User.query.filter_by(id=id).first()
    user.email = email
    user.password = generate_password_hash(password, method='pbkdf2:sha256')
    user.first_name = first_name
    user.role = role
    db.session.commit()

    flash("User updated successfully!", category="success")
    return redirect(url_for('user.index'))

@user.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    user = User.query.filter_by(id=id).first()
    db.session.delete(user)
    db.session.commit()
    flash("User deleted successfully!", category="success")
    return redirect(url_for('user.index'))