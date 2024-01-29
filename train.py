import face_recognition
import cv2
import numpy as np
import string
import os
import pyrebase


#DATABASE KEY & INITIALIZATION
config = {
        "apiKey": "AIzaSyAWcOr7SsFnbzgbRBrpyWzXl9mEgKtIgrM",
        "authDomain": "uitfarmerclass.firebaseapp.com",
        "projectId": "uitfarmerclass",
        "storageBucket": "uitfarmerclass.appspot.com",
        "messagingSenderId": "603995142746",
        "appId": "1:603995142746:web:417413652766ab9906f663",
        "measurementId": "G-CJEW8BHZ4C",
        "databaseURL": "gs://uitfarmerclass.appspot.com"
}

firestore = pyrebase.initialize_app(config)
storage = firestore.storage()


known_face_encodings =[]
known_face_names =[]
label_folder_name = 'Models'
images_dir = os.path.join(".", label_folder_name)
for files in os.listdir(images_dir):    
    print(files)
    count = 0
    for file in os.listdir(images_dir+"/"+files):
        count=count+1
        print(file)
        image = face_recognition.load_image_file(images_dir+"/"+files+"/"+file)
        try:
            known_face_encodings.append(face_recognition.face_encodings(image)[0])
            known_face_names.append(str(files))
            print('Success')
        except:
            print(file + "  ERROR")
            os.remove(images_dir+"/"+files+"/"+file)
np.savetxt("encoding.txt",known_face_encodings)
np.savetxt("encodingname.txt",known_face_names,delimiter=" ", fmt="%s")


storage.child("traindata/encoding.txt").put("encoding.txt")
storage.child("traindata/encodingname.txt").put("encodingname.txt")
