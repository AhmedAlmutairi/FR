import numpy as np
import cv2
import os




class FR:

        def __init__(self, classifier=None):
                self.db = ['Geoffery Hinton', 'MBS']
                self.classifier = classifier

        def start_training(self):
                #faces, labels = self.train_dataset('Training')
                #print(faces)
                #print(labels)
                self.classifier = cv2.face.createLBPHFaceRecognizer()
                self.classifier.load('classifier.xml')
                #self.classifier.save('classifier.xml')
                

        def train_dataset(self, path_to_folder):
                faces = []
                labels = []
                folders = os.listdir(path_to_folder)
                
                for f in folders:
                        if f == ".DS_Store":
                                print(f)
                                pass
                        else:
                                label = int(f)
                                images = path_to_folder + '/' + f + '/'
                                print(images)
                                for img in os.listdir(images):
                                        path = images + img
                                        print(path)
                                        if path.endswith('.jpg') or path.endswith('.jpeg') or path.endswith('.png'):
                                                image = cv2.imread(path)
                                                face, rectangle = self.select_faces(image)
                                                faces.append(face)
                                                labels.append(label)
                                                cv2.imshow(' ', face)
                                                cv2.waitKey(500)
                                                cv2.destroyAllWindows()

                return faces, labels

        def select_faces(self, image):
                cascPath = "haarcascade_frontalface_default.xml"
                faceCascade = cv2.CascadeClassifier(cascPath)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                print(faces[0])
                (x,y,w,h) = faces[0]
                return gray[y:y+w, x:x+h], faces[0]


        def predict_image(self):
                img = cv2.imread("Testing/1/12.jpeg")
                face, rectangle = self.select_faces(img)
                label = self.classifier.predict(face)
                label_text = self.db[label-1]
                print (label)
                print (label_text)
                (x,y,w,h) = rectangle
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(img, label_text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.imshow('Face Recognition', img)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()

if __name__ == '__main__':
        fr = FR()
        fr.start_training()
        fr.predict_image()


