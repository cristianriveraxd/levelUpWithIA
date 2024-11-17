from .entities.user import User

class ModelUser():
    
    @classmethod
    def login(self, db, user):
        try:
            cursor=db.connection.cursor()
            sql = """SELECT id_usuario, email, password FROM usuarios 
            WHERE email = '{}'""".format(user.username)
            cursor.execute(sql)
            row=cursor.fetchone()
            if row != None:
                user=User(row[0],row[1],User.check_password(row[2],user.password))
                return user
            else: 
                return None
        except Exception as e:
            raise Exception(e)
    
    @classmethod
    def getId (self, db, id):
        try:
            cursor=db.connection.cursor()
            sql = "SELECT id_usuario, email, password FROM usuarios WHERE id_usuario = {}".format(id)
            cursor.execute(sql)
            row=cursor.fetchone()
            cursor.close()
            if row != None:
                return User(row[0], row[1], row[2])
            else: 
                return None 
        except Exception as e:
            raise Exception(e)
    
   