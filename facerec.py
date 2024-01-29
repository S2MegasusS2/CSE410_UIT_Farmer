import face_recognition
import cv2
import numpy as np
import string
import os
import pyrebase
import time

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

SelectedSeat = 31

seated = False
seatid = ''
import subprocess
def checkInternet():
    try:
        subprocess.check_output(["ping","-c","1","8.8.8.8"])
        return True
    except: return False
def GetSeat(id):
    if not(checkInternet()): return
    global seated
    global seatid
    global seatname
    docs = (db.collection('users-info').stream())
    for doc in docs:
        if (doc.id == id): 
            if (not seated):
                print(doc._data)
                seated = True
                if (doc._data['seat']=='none'):
                    doc_ref = db.collection('users-info').document(id)
                    seat_docs = (db.collection('seat-data').stream())
                    for seat in seat_docs: 
                        if seat._data['seat-id'] == SelectedSeat:
                            if (seat._data['owner'] == 'none'):
                                seatid = seat.id
                                seat_ref = db.collection('seat-data').document(seat.id)
                                seat_ref.update({
                                    'owner': doc._data['id'],
                                    '`owner-name`': doc._data['name'],
                                    'email':doc._data['email'],
                                    'type':'Offline'
                                }) 
                                doc_ref.update({
                                    'seat': seat.id
                                })
                                break
                #Replace current online seat into Offline
                else:
                    seatid = doc._data['seat']
                    RemoveSeat(id)
                    GetSeat(id)
                    #doc_ref = db.collection('users-info').document(id)
                    #seat_ref = db.collection('seat-data').document(doc._data['seat'])
                    #seat_ref.update({
                    #    'owner': doc._data['id'],
                    #    '`owner-name`': doc._data['name'],
                    #    'email':doc._data['email'],
                    #    'type':'Offline'
                    #    }) 
                
def RemoveSeat(id):
    if not(checkInternet()): return
    global seated
    global seatid
    print(seatid)
    doc_ref = db.collection('users-info').document(id)
    seat_ref = db.collection('seat-data').document(seatid)
    seat_ref.update({
        'owner':'none',
        '`owner-name`':'none',
        'email':'none',
        'type':'Offline'
        }) 
    doc_ref.update({
        'seat':'none'
    })
    seated = False
     
cred = credentials.Certificate(r"/home/tanphat/Desktop/dlib_face/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
if True:   
    try:
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
    except: print("No internet")


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
known_face_encodings =[]
known_face_names =[]
#try:
#    storage.child("traindata/encoding.txt").download('./','encoding.txt')
#    storage.child("traindata/encodingname.txt").download("./","encodingname.txt")
#except: print("No DB")

known_face_encodings = np.loadtxt("/home/tanphat/Desktop/dlib_face/encoding.txt")
process_this_frame = True
known_face_names = np.genfromtxt("/home/tanphat/Desktop/dlib_face/encodingname.txt", dtype='str')

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
first_face='none'
start = time.time()
end = time.time()
while True:
    while not checkInternet(): print("Retrying internet")
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        end = time.time()
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                if face_distances[best_match_index] <=0.45:
                    name = known_face_names[best_match_index]
                    if (first_face=='none'): 
                        first_face = name
                    if (first_face == name): start = time.time()
            face_names.append(name)

            if (first_face!='none'):
                GetSeat(first_face)
        if (first_face!='none' and end - start >=10):
            RemoveSeat(first_face) 
            first_face = 'none'
            print('Seat Expired')
            seated = False
    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
