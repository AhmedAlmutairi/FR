import flask
from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2


PEOPLE_FOLDER = os.path.join('static', 'people_photo')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

app = flask.Flask(__name__)
app.config["DEBUG"] = True
db = ['Geoffery Hinton', 'MBS']
classifier = cv2.face.createLBPHFaceRecognizer()
classifier.load('/Users/ahmed/Documents/project/api/templates/classifier.xml')

@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
	return render_template("index.html")
	#full_filename = os.path.join(PEOPLE_FOLDER, 'uploaded.jpeg')
	#return render_template("index.html", user_image = full_filename)

def select_faces(image):
	cascPath = "/Users/ahmed/Documents/project/api/templates/haarcascade_frontalface_default.xml"
	faceCascade = cv2.CascadeClassifier(cascPath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
	(x,y,w,h) = faces[0]
	return gray[y:y+w, x:x+h], faces[0]

@app.route('/predict_image', methods = ['POST'])
def predict_image():
	f = request.files['file']
	file_name = f.filename
	f.save(os.path.join(PEOPLE_FOLDER, file_name))
	full_filename = os.path.join(PEOPLE_FOLDER, file_name)
	img = cv2.imread(full_filename)
	face, rectangle = select_faces(img)
	label = classifier.predict(face)
	label_text = db[label-1]
	(x,y,w,h) = rectangle
	cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
	cv2.putText(img, label_text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
	#cv2.imshow('Face Recognition', img)
	#cv2.waitKey(5000)
	#cv2.destroyAllWindows()
	resized = cv2.resize(img, (300, 300), interpolation = cv2.INTER_AREA)
	save_path = os.path.join('static', 'save_photo')
	cv2.imwrite(os.path.join(save_path, file_name), resized)
	#saved_image = cv2.imread(file_name)
	#saved_image.save(os.path.join(PEOPLE_FOLDER, file_name))
	full_filename = os.path.join(save_path, file_name)
	#print(full_filenames)
	#os.path.join()
	return render_template("index.html", user_image = full_filename)
	#return redirect(url_for('index'))

app.run()




