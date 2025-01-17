from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_bcrypt import check_password_hash
from flask_login import login_user, login_required, logout_user
from . import db
from .models import User

auth = Blueprint('auth', __name__)


@auth.route('/login')
def login():
    return render_template('login.html')


@auth.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        flash('Invalid username or password entered, please try again.')
        return redirect(url_for('auth.login'))

    login_user(user, remember=remember)
    return redirect(url_for('main.profile'))


@auth.route("/signup")
def signup():
    return render_template('signup.html')


@auth.route("/signup", method=['POST'])
def signup_post():
    return redirect(url_for("auth.login"))


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))
