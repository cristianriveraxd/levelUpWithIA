import cv2
import mediapipe as mp
import numpy as np
from Detector import Detector

#Crear instancia del objeto
gesture_detector = Detector(num_frames_to_track=5, movement_threshold=10)

#objetos
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

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
    
# #funcion para adios
def is_goodbye(landmarks):
    #Llamar al metodo para detectar movimiento en dedosññlñ
    hand_center_x = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x + landmarks[mp_hands.HandLandmark.PINKY_TIP].x) / 2

    if landmarks[mp_hands.HandLandmark.WRIST].x < hand_center_x:
        return True
    else:
        return False

#funcion para como estas
def is_how_are_you(landmarks):
    hand_center_y = (landmarks[mp_hands.HandLandmark.THUMB_TIP].y + landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y + landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y + landmarks[mp_hands.HandLandmark.PINKY_TIP].y) / 4
    if landmarks[mp_hands.HandLandmark.WRIST].y < hand_center_y:
        return True
    return False

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

#funcion de pulgar abajo
def is_thumb_down(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    position=get_hand_position(landmarks)
    if position == 0:
      return (thumb_tip.y > thumb_mcp.y and 
              thumb_tip.y > thumb_ip.y and 
              thumb_tip.y > thumb_cmc.y and
              thumb_tip.y > pinky_mcp.y )
      
#funcion en v
def is_v_sign(landmarks):
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    index_finger_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    distance_tip = ((index_finger_tip.x - middle_finger_tip.x) ** 2 + 
                    (index_finger_tip.y - middle_finger_tip.y) ** 2) ** 0.5
    distance_mcp = ((index_finger_mcp.x - middle_finger_mcp.x) ** 2 + 
                    (index_finger_mcp.y - middle_finger_mcp.y) ** 2) ** 0.5

    return distance_tip > distance_mcp

# Función para detectar "Gracias"
# def is_thank_you(landmarks,pose_landmarks):
    
     # Distancia Euclidiana
    def euclidean_distance(landmark1, landmark2):
        return ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5

    # 1. Verificar si la mano está cerca de la boca
    mouth_landmark = pose_landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]
    wrist_landmark = landmarks[mp_hands.HandLandmark.WRIST]
    if euclidean_distance(wrist_landmark, mouth_landmark) > 0.1:  # Ajustar el umbral según sea necesario
        return False

    # 2. Verificar que los dedos estén juntos y la mano esté extendida
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    if euclidean_distance(index_tip, middle_tip) > 0.03 or euclidean_distance(middle_tip, ring_tip) > 0.03 or euclidean_distance(ring_tip, pinky_tip) > 0.03:
        return False

    # 3. Verificar el movimiento hacia adelante
    current_x = (index_tip.x + pinky_tip.x) / 2
    current_y = (index_tip.y + pinky_tip.y) / 2

    # Guardar posiciones anteriores en una variable global o de instancia
    if not hasattr(is_thank_you, 'previous_position'):
        is_thank_you.previous_position = (current_x, current_y)

    previous_x, previous_y = is_thank_you.previous_position
    movement_x = current_x - previous_x
    movement_y = current_y - previous_y

    # Actualizar la posición anterior
    is_thank_you.previous_position = (current_x, current_y)

    # Verificar si la mano se ha movido hacia adelante
    if movement_x > 0.05 and movement_y > -0.05:  # Ajustar el umbral según sea necesario
        return True

    return False

# Función para detectar "Por favor"
def is_please(landmarks):
    if landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.WRIST].y and landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.WRIST].y:
        return True
    return False

# Función para detectar "Amor"
def is_love(landmarks):
    thumb_tip_y = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip_y = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    pinky_tip_y = landmarks[mp_hands.HandLandmark.PINKY_TIP].y
    if thumb_tip_y > index_tip_y and thumb_tip_y > pinky_tip_y:
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

# Función para detectar "¿Qué hora es?"
def is_what_time(landmarks):
    hand_center_x = (landmarks[mp_hands.HandLandmark.THUMB_TIP].x + landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x + landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x + landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x + landmarks[mp_hands.HandLandmark.PINKY_TIP].x) / 5
    if landmarks[mp_hands.HandLandmark.WRIST].x < hand_center_x:
        return True
    return False
  
# Función para detectar "Sí"
def is_yes(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    if thumb_tip.y < thumb_mcp.y and thumb_tip.y < thumb_ip.y and thumb_tip.y < thumb_cmc.y:
        return True
    return False

# Función para detectar "No"
def is_no(landmarks):
    index_tip_x = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    pinky_tip_x = landmarks[mp_hands.HandLandmark.PINKY_TIP].x
    if index_tip_x < pinky_tip_x:
        return True
    return False
  
cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2) as hands, \
    mp_pose.Pose(
         model_complexity=1,
         min_detection_confidence=0.7,
         min_tracking_confidence=0.7) as pose:
  
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        #voltear la camara
        image = cv2.flip(image, 1)
    
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #proceso de hands
        results = hands.process(image)

        #proceso de pose
        results_pose= pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark

                #Llamado de funciones aqui
                if is_hello(landmarks):
                    print("Hola")
                    cv2.putText(image, 'Hola', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # if is_goodbye(landmarks):
                #     print("Adios")
                #     cv2.putText(image, 'Adios', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
                # if is_how_are_you(landmarks):
                #     print("Como estas?")
                 #     cv2.putText(image, '¿Como estas?', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
                if is_thumb_up(landmarks):
                    print("Bien")
                    cv2.putText(image, 'Bien', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if is_thumb_down(landmarks):
                    print("Mal")
                    cv2.putText(image, 'Mal', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
               
                # # if is_v_sign(landmarks):
                # #     print("Seña V") 
                # #     cv2.putText(image, 'Seña V', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            
                # if is_please(landmarks):
                #     print("Por favor")
                #     cv2.putText(image, 'Por favor', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # if is_love(landmarks):
                #     print("Amor")
                #     cv2.putText(image, 'Amor', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
              
                if is_okay(landmarks):
                    print("Okey")
                    cv2.putText(image, 'Okey', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
                 # if is_what_time(landmarks):
                #     print("¿Que hora es?")
                #     cv2.putText(image, '¿Que hora es?', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) 
            
                # if is_yes(landmarks):
                #     print("Si")
                #     cv2.putText(image, 'Si', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) 
            
                # if is_no(landmarks):
                #     print("No")
                #     cv2.putText(image, 'No?', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) 
                
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
                pose_landmarks = results_pose.pose_landmarks.landmark  
                  
                # if is_thank_you(landmarks,pose_landmarks):
                #     print("Gracias")
                #     cv2.putText(image, 'Gracias', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  
                  
                  
            cv2.imshow('Translate', image)
        
            if cv2.waitKey(5) & 0xFF == 27:
                break
cap.release()
cv2.destroyAllWindows()