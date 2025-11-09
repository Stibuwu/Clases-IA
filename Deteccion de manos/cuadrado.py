import cv2
import mediapipe as mp
import numpy as np
import math

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Variable global para almacenar el ángulo de rotación anterior (para suavizar la rotación)
angulo_rotacion_anterior = 0

def dibujar_cuadrado_rotado(frame, centro, tamaño, angulo_rotacion, color, grosor=-1):
    """
    Dibuja un cuadrado rotado en el frame.
    
    Args:
        frame: Imagen donde se dibujará
        centro: Tupla (x, y) con el centro del cuadrado
        tamaño: Lado del cuadrado en píxeles
        angulo_rotacion: Ángulo de rotación en grados
        color: Color del cuadrado (B, G, R)
        grosor: Grosor de la línea (-1 para relleno)
    """
    # Calcular las esquinas del cuadrado antes de rotar
    mitad = tamaño / 2
    esquinas = np.array([
        [-mitad, -mitad],
        [mitad, -mitad],
        [mitad, mitad],
        [-mitad, mitad]
    ], dtype=np.float32)
    
    # Crear matriz de rotación
    angulo_rad = math.radians(angulo_rotacion)
    cos_ang = math.cos(angulo_rad)
    sin_ang = math.sin(angulo_rad)
    matriz_rotacion = np.array([
        [cos_ang, -sin_ang],
        [sin_ang, cos_ang]
    ])
    
    # Rotar las esquinas
    esquinas_rotadas = esquinas @ matriz_rotacion.T
    
    # Trasladar al centro
    esquinas_finales = esquinas_rotadas + np.array(centro)
    esquinas_finales = esquinas_finales.astype(np.int32)
    
    # Dibujar el cuadrado
    cv2.fillPoly(frame, [esquinas_finales], color) if grosor == -1 else cv2.polylines(frame, [esquinas_finales], True, color, grosor)

def calcular_angulo_rotacion(pulgar, indice, muñeca):
    """
    Calcula el ángulo de rotación basado en la posición de los dedos.
    Usa el ángulo del vector pulgar-índice respecto a la horizontal.
    
    Args:
        pulgar: Coordenadas (x, y) del pulgar
        indice: Coordenadas (x, y) del índice
        muñeca: Coordenadas (x, y) de la muñeca (para referencia)
    
    Returns:
        Ángulo en grados (0-360)
    """
    # Calcular vector desde pulgar hacia índice
    vector = np.array(indice) - np.array(pulgar)
    
    # Calcular ángulo en radianes
    angulo_rad = math.atan2(vector[1], vector[0])
    
    # Convertir a grados (0-360)
    angulo_grados = math.degrees(angulo_rad)
    if angulo_grados < 0:
        angulo_grados += 360
    
    return angulo_grados

def procesar_mano(hand_landmarks, frame):
    """
    Procesa la detección de la mano y dibuja el cuadrado con rotación.
    
    Args:
        hand_landmarks: Landmarks de la mano detectada
        frame: Frame de video donde se dibujará
    """
    global angulo_rotacion_anterior
    
    h, w, _ = frame.shape  # Tamaño de la imagen
    
    # Obtener coordenadas de los puntos clave en píxeles
    dedos = [(int(hand_landmarks.landmark[i].x * w), int(hand_landmarks.landmark[i].y * h)) for i in range(21)]
    
    # Obtener posiciones clave (puntas de los dedos y muñeca)
    muñeca = dedos[0]
    pulgar = dedos[4]
    indice = dedos[8]
    medio = dedos[12]
    anular = dedos[16]
    meñique = dedos[20]
    
    # Calcular distancia entre pulgar e índice (para el tamaño del cuadrado)
    distancia_pulgar_indice = np.linalg.norm(np.array(pulgar) - np.array(indice))
    
    # Calcular ángulo de rotación basado en la posición de los dedos
    angulo_rotacion = calcular_angulo_rotacion(pulgar, indice, muñeca)
    
    # Suavizar la rotación para evitar movimientos bruscos
    # Interpolación lineal simple entre el ángulo anterior y el actual
    factor_suavizado = 0.3
    angulo_rotacion = angulo_rotacion_anterior * (1 - factor_suavizado) + angulo_rotacion * factor_suavizado
    angulo_rotacion_anterior = angulo_rotacion
    
    # Calcular el centro del cuadrado (punto medio entre pulgar e índice, o cerca del pulgar)
    centro_x = int((pulgar[0] + indice[0]) / 2)
    centro_y = int((pulgar[1] + indice[1]) / 2)
    centro = (centro_x, centro_y)
    
    # Tamaño del cuadrado basado en la distancia pulgar-índice
    # Ajustar el tamaño mínimo y máximo según sea necesario
    tamaño_minimo = 30
    tamaño_maximo = 200
    tamaño_cuadrado = np.clip(distancia_pulgar_indice * 1.5, tamaño_minimo, tamaño_maximo)
    
    # Dibujar el cuadrado rotado
    color_cuadrado = (34, 234, 65)  # Color verde en BGR
    dibujar_cuadrado_rotado(frame, centro, tamaño_cuadrado, angulo_rotacion, color_cuadrado, -1)
    
    # Dibujar línea entre pulgar e índice para visualización
    cv2.line(frame, pulgar, indice, (244, 34, 12), 3)
    
    # Mostrar información en pantalla
    cv2.putText(frame, f'Distancia: {int(distancia_pulgar_indice)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Angulo: {int(angulo_rotacion)}°', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Tamaño: {int(tamaño_cuadrado)}', (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Mostrar los números de los landmarks en la imagen (opcional)
    for i, (x, y) in enumerate(dedos):
        cv2.circle(frame, (x, y), 5, (233, 23, 0), -1)
        cv2.putText(frame, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe
    results = hands.process(frame_rgb)

    # Dibujar puntos de la mano y procesar gestos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar el esqueleto de la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Procesar la mano y dibujar el cuadrado con rotación
            procesar_mano(hand_landmarks, frame)

    # Mostrar el video
    cv2.imshow("Cuadrado Interactivo con Rotación", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()