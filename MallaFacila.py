import cv2
import mediapipe as mp
import math

#--------- relizamos la videoCaputura ----------------
cap = cv2.VideoCapture(0)
cap.set(3,1280) #Definicion del ancho de la ventana
cap.set(4,720) #Definicion del alto de la ventana

#--------------Creamos mnuestra funcison de dibujo ------------------
mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)#Ajustamos la configuracion de dibujo

#---------------Creamos un objeto donde almacenaremos la malla facial --------------------
mpMallaFacial = mp.solutions.face_mesh #Primero llamamos la funcion
MalllaFacial = mpMallaFacial.FaceMesh(max_num_faces=1) #Creamos el objeto (Ctrl + Clcik) -> ingresamo a la fuincion se puede agregar el maximo de caras a mostrar  tambien los puntos que representan las partes de la cara ojos labios etc las lineas de marcacion

while True:
    ret, frame = cap.read()

    #Corregir el error de espejo
    frame = cv2.flip(frame, 1)

    #Corregir el color
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Observamos lo resultados
    resultados = MalllaFacial.process(frameRGB)

    #Creamos uinas listas donde almacenaremos los resultados
    px = []
    py = []
    lista = []
    r = 5
    t = 5

    if resultados.multi_face_landmarks: #Si detectamos algun rostro
        for rostros in resultados.multi_face_landmarks: #mostramos el rostro detectado
            mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_TESSELATION, ConfDibu, ConfDibu)

            #Ahora vamos a extraer los puntos del rostro detectado
            for id,puntos in enumerate(rostros.landmark):
                #print(puntos) #nos entrega una proporcion

                al, an ,c = frame.shape
                x,y = int(puntos.x*an), int(puntos.y*al) # Nos entrega un pixel

                px.append(x)
                py.append(y)
                lista.append([id,x,y])
                if len(lista) == 468:
                    #Ceja Derecha
                    x1, y1 = lista[65][1:] #punto 65 con sus cordenadas
                    x2, y2 = lista[158][1:]
                    cx, xy = (x1 + x2) //2, (y1 +y) //2

                    #Como se comporta la medicion
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), t)
                    cv2.circle(frame, (x1, y1), r, (0, 0, 0), 2)
                    cv2.circle(frame, (x2, y2), r, (0, 0, 0), 2)
                    cv2.circle(frame, (cx, xy), r, (0, 0, 0), 2)

                    longitud1 = math.hypot(x2 - x1 , y2 - y1)
                    print(longitud1)

                    #Ceja Izquierda
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3 +x4) //2, (y3 + y4) // 2
                    longitud2 = math.hypot(x4 - x3 , y4 - y3)
                    print(longitud2)

                    #Boca Extremos
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    cx3, y3 = (x5 + x6) //2, (y5 - y6) // 2
                    longitud3 = math.hypot(x6 - x5 , y6 - y5)
                    print(longitud3)

                    #Boca Apertura
                    x7, y7 = lista[13][1:]
                    x8, y8 = lista[14][1:]
                    cx4, cy4 = (x7 +x8) //2, (y7 - y8) // 2
                    longitud4 = math.hypot(x8 - x7 , y8 - y7)
                    print(longitud4)

                    #Clasificacion
                    #Bravo
                    if longitud1 < 19 and longitud2 < 19 and longitud3 > 80 and longitud4 < 95 and longitud4 < 5:
                        cv2.putText(frame, 'Enojado', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    #feliz
                    elif longitud1 >20 and longitud1 <30 and longitud2 > 20 and longitud2 <30 and longitud3 >109 and longitud4 > 10 and longitud4 < 20:
                        cv2.putText(frame, 'Feliz ', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    #Asombrado
                    elif longitud1 > 35 and longitud2 >35 and longitud3 > 80 and longitud3 <90 and longitud4 >20:
                        cv2.putText(frame, 'Asombrado', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    #triste
                    elif longitud1 >20 and longitud1 <35 and longitud2 < 20 and longitud2 <35 and longitud3 >80 and longitud3 < 95 and longitud4 <5:
                        cv2.putText(frame, 'Trsite', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('Reconocimiento de emociones', frame)
    t = cv2.waitKey(1)

    if t == 27:
        break
cap.release()
cv2.destroyAllWindows()