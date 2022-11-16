import cv2
import os
import imutils

personName = 'Cristiano Ronaldo' #Se crea carpeta carpeta con el nombre de la persona que deseamos reconocer
dataPath = r'C:\Users\Joan Pinedo\Desktop\Devs\Reconocimiento facial\Data'
personPath = dataPath + '/' + personName

if not os.path.exists(personPath): #Se crea un directorio con el nombre de la persona dentro de la carpeta data
    print('Carpeta creada: ',personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture('Cr7.mp4') #Leemos el video


faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #Clasificador pre entrenado opencv
count = 0

while True:
    
    ret, frame = cap.read()
    if ret == False: break
    frame =  imutils.resize(frame, width=640) #Redimensionamos el tamaño de los fotogramas del video de entrada
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC) #Redimension de las imagenes correspondientes al rostro para que todas tengan el mismo tamaño
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count),rostro)
        count = count + 1
    cv2.imshow('frame',frame)

    k =  cv2.waitKey(1)
    if k == 27 or count >= 300: #Cantidad de imagenes a tomar del video
        break

cap.release()
cv2.destroyAllWindows()