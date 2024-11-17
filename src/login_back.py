from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_mysqldb import MySQL
from flask_login import LoginManager, login_user, logout_user, login_required, current_user 
from flask_wtf import CSRFProtect

from models.modelUser import ModelUser
# entities
from models.entities.user import User

app = Flask(__name__)
app.secret_key = '123456789'

app.config['MYSQL_HOST'] = 'localhost'    
app.config['MYSQL_USER'] = 'root'       
app.config['MYSQL_PASSWORD'] = '' 
app.config['MYSQL_DB'] = 'levelapp' 
mysql = MySQL(app)

app.config["SESSION_COOKIE_SECURE"] = True  # Solo en HTTPS
app.config["SESSION_COOKIE_HTTPONLY"] = True  # Evita accesos desde JavaScript
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"  # Protege contra ataques CSRF

csrf = CSRFProtect()

loginManagerapp = LoginManager(app)
loginManagerapp.login_view = "login"

@loginManagerapp.user_loader
def load_user(id):
    return ModelUser.getId(mysql, id)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        print('validación de datos')
        print(request.form['email'])
        print(request.form['password'])
        user = User(0,request.form['email'], request.form['password'])
        loggedUser = ModelUser.login(mysql, user)
        if loggedUser:
            if loggedUser.password:
                login_user(loggedUser)
                return render_template("index.html")
            else:
                flash("Contraseña incorrecta")
        else:
            flash("Usuario no encontrado")
    return render_template('login.html')
    
@app.route("/appy")
@login_required
def index_cam():
    return render_template("index.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


def create_app():
    csrf.init_app(app)
    return app
