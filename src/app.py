from flask import Flask, render_template, request, redirect, url_for, Response, flash
import numpy as np
import cv2
import mediapipe as mp
from Detector import Detector
import login_back

gesture_detector = Detector(num_frames_to_track=5, movement_threshold=10)

#objetos
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5)

pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

#funcion para dectectar posición de la palma
def get_hand_position(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    # Comparar la posición X del pulgar y del meñique
    if thumb_tip.x < pinky_tip.x:
        #return palma de frente
        return 1
    else:
        #return palma volteada
        return 0   
#funcion para obtener orientacion de la palma
def get_hand_orientation(landmarks):
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    # Promedio de las posiciones Y de los MCPs
    mcp_y_average = (index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 4

    if wrist.y > mcp_y_average:
        #return arriba
        return 2
    elif wrist.y < mcp_y_average:
        #return tumbado
        return 1
    else:
        #return de lado
        return 0
#funcion de saludo
def is_hello(landmarks):
    fingers_up = [landmarks[mp_hands.HandLandmark.THUMB_TIP].y,
                  landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                  landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                  landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y,
                  landmarks[mp_hands.HandLandmark.PINKY_TIP].y]
    #capturar posicion
    position=get_hand_position(landmarks)
    if position == 1 and all(y > landmarks[mp_hands.HandLandmark.WRIST].y for y in fingers_up):
        return True
    else:
      return False    
    
    
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    # Promedio de las posiciones Y de los MCPs
    mcp_y_average = (index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 4

    if wrist.y > mcp_y_average:
        #return arriba
        return 2
    elif wrist.y < mcp_y_average:
        #return tumbado
        return 1
    else:
        #return de lado
        return 0
#funcion de pulgar arriba
def is_thumb_up(landmarks):
    
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    index_finger_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_finger_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_finger_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    
    #lista de coordenadas en x para sacar promedio
    x_coords = [index_finger_mcp.x, middle_finger_mcp.x, ring_finger_mcp.x, pinky_mcp.x]
    
    #calcular el promedio en x
    promedio_x = np.mean(x_coords)
    #variación posible segun toma de coordenadas
    variacion_max =1
    
    if(thumb_tip.y < thumb_ip.y and all(abs(x - promedio_x) <= variacion_max for x in x_coords) and middle_finger_tip.x > middle_finger_dip.x):
      return True
    return False
# Función para detectar "Okey"
def is_okay(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    
     #lista de coordenadas en x para sacar promedio
    x_coords = [thumb_tip.x,index_tip.x]
    
    #calcular el promedio en x
    promedio_x = np.mean(x_coords)
    #variación posible segun toma de coordenadas
    variacion_max =3
    
    position=get_hand_position(landmarks)
    orientation=get_hand_orientation(landmarks)
    
    if thumb_tip.y > index_tip.y and position==1 and orientation==2 and middle_finger_tip.y > ring_finger_tip.y and all(abs(x - promedio_x) <= variacion_max for x in x_coords):
        return True
    return False



app = Flask(__name__)
#Logica de inicio
@app.route('/')
def login():
   login_back. load_user()


#Logica de login
@app.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')

    return redirect(url_for('index'))

#Ruta para index
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/show_points', methods=['POST'])
def toggle_show_points():
    global show_points
    show_points = not show_points  # Cambiar el estado
    return {'show_points': show_points}

# Mostramos el video en RT
def gen_frame():
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return  # Asegúrate de que la cámara se abra correctamente

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el marco.")
            break
        
        frame = cv2.flip(frame, 1)    
        # Convertir la imagen a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        message = "No gesture detected"
        
        results = hands.process(image_rgb)
        pose_results = pose.process(image_rgb)

        # Solo dibujar los puntos si show_points es True
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar los puntos de las manos
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                if is_hello(landmarks):
                    print("Hola")
                    message = "Hola"
                    
                if is_thumb_up(landmarks):
                    print("Bien") 
                    message = "Bien"
                    
                if is_okay(landmarks):
                    print("Okey")
                    message = "okey"
                    
                cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 255, 51), 2, cv2.LINE_AA)        
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)  
            
                             
        # Codificamos nuestro video en Bytes
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

       

@app.route('/video', methods=['GET'])
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)