from werkzeug.security import check_password_hash #importamos el metodo para hashear las contraseñas
from flask_login import UserMixin 

class User(UserMixin):
    
    #metodo constructor 
    #clase de reflejo tabla usuario
    #maneja entidades tipo usuario
    def __init__(self, id,  username, password) -> None:
        self.id = id
        self.username = username 
        self.password = password
    
    #este metodo recibe el password guardado y el password que recibe (en texto plano)
    @classmethod #este decorador se utiliza para no instanciar la clase
    def check_password(self, hashed_password, password):
        return check_password_hash(hashed_password, password)
    
#generate_password_hash('')