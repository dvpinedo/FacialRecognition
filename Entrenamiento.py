from email.mime import image
import cv2
import os
import numpy as np

dataPath = r'C:\Users\Joan Pinedo\Desktop\Devs\Reconocimiento facial\Data' #ruta donde guardamos las imagenes de los videos
personList = os.listdir(dataPath) #listamos todos los nombres almacenados
print('Lista de personas: ', personList) #imprimimos la lista obtenida

labels = [] #Se almacena las etiquetas correspondientes a cada persona
facesData = [] #Se almacena cada una de las imagenes de los rostros
label = 0 #Se establece contador en 0 para la lectura de las imagenes

for nameDir in personList:
    personPath = dataPath + '/' + nameDir #Ruta de la carpeta de cada persona
    print('Analizando las imagenes...')

    for fileName in os.listdir(personPath): #lectura de imagenes correspondientes a cada rostro
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label) #añadiendo la etiqueta de cada imagen
        facesData.append(cv2.imread(personPath+ '/' +fileName,0)) #se añade cada imagen en el array

    label= label + 1 #se encarga de reconocer las carpetas en un contador empezando por 0 


face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#Entrenamiento

print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

#Guardado modelo de entrenamiento obtenido
#face_recognizer.write('modeloEigenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
face_recognizer.write('modeloEigenFace.xml')

print('Modelo guardado')

