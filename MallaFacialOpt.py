import cv2
import mediapipe as mp
import math
import threading

# Inicialización de captura de video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Definición del ancho de la ventana
cap.set(4, 720)   # Definición del alto de la ventana

# Configuración de MediaPipe para la malla facial
mpMallaFacial = mp.solutions.face_mesh
MalllaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)

# Variables globales para multihilo
frame = None
resultados = None

# Función de procesamiento de frames
def process_frame():
    global frame, resultados
    while True:
        if frame is not None:
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = MalllaFacial.process(frameRGB)

# Inicio del hilo para procesamiento de frames
thread = threading.Thread(target=process_frame)
thread.start()

# Variables para control de frecuencia de procesamiento
frame_counter = 0
process_interval = 5  # Procesar cada 5 frames

while True:
    ret, frame = cap.read()
    frame_counter += 1

    # Corregir el error de espejo
    frame = cv2.flip(frame, 1)

    if frame_counter % process_interval == 0 and resultados is not None:
        if resultados.multi_face_landmarks:
            for rostros in resultados.multi_face_landmarks:
                lista = []
                for id, puntos in enumerate(rostros.landmark):
                    al, an, c = frame.shape
                    x, y = int(puntos.x * an), int(puntos.y * al)
                    lista.append([id, x, y])

                if len(lista) == 468:
                    # Pre-calculo de coordenadas
                    x1, y1 = lista[65][1], lista[65][2]
                    x2, y2 = lista[158][1], lista[158][2]
                    x3, y3 = lista[295][1], lista[295][2]
                    x4, y4 = lista[385][1], lista[385][2]
                    x5, y5 = lista[78][1], lista[78][2]
                    x6, y6 = lista[308][1], lista[308][2]
                    x7, y7 = lista[13][1], lista[13][2]
                    x8, y8 = lista[14][1], lista[14][2]

                    # Cálculo de longitudes
                    longitud1 = math.hypot(x2 - x1, y2 - y1)
                    longitud2 = math.hypot(x4 - x3, y4 - y3)
                    longitud3 = math.hypot(x6 - x5, y6 - y5)
                    longitud4 = math.hypot(x8 - x7, y8 - y7)

                    # Clasificación de emociones
                    emotion = ""
                    if longitud1 < 19 and longitud2 < 19 and longitud3 > 80 and longitud4 < 5:
                        emotion = 'Enojado'
                    elif 20 < longitud1 < 30 and 20 < longitud2 < 30 and longitud3 > 109 and 10 < longitud4 < 20:
                        emotion = 'Feliz'
                    elif longitud1 > 35 and longitud2 > 35 and 80 < longitud3 < 90 and longitud4 > 20:
                        emotion = 'Asombrado'
                    elif 20 < longitud1 < 35 and longitud2 < 20 and 80 < longitud3 < 95 and longitud4 < 5:
                        emotion = 'Triste'

                    cv2.putText(frame, emotion, (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow('Reconocimiento de emociones', frame)

    if cv2.getWindowProperty('Reconocimiento de emociones', cv2.WND_PROP_VISIBLE) < 1:
        break

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
