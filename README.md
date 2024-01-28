# Smart table using HOG-SVM to indentify student's face
## Introduction
Recognize and manipulate faces from Python or from the command line with the world's simplest face recognition library.
Built using [dlib](http://dlib.net/)'s state-of-the-art face recognition
built with deep learning. The model has an accuracy of 90.38% on the
[Large-scale CelebFaces Attributes (CelebA) ](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) benchmark.

## Requirements
- Python 3.8+ or Python 2.7
- macOS or Linux (Windows not officially supported, but might work)

## Installation options 
### Installing on Mac or Linux
First, make sure you have dlib already installed with Python bindings:

  * [How to install dlib from source on macOS or Ubuntu](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)
  
Then, make sure you have cmake installed:  
 
```bash
brew install cmake
```

Finally, install this module from pypi using `pip3` (or `pip2` for Python 2):

```bash
pip3 install face_recognition
```

### Installing on an Nvidia Jetson Nano board

 * [Jetson Nano installation instructions](https://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd)
   * Please follow the instructions in the article carefully. There is current a bug in the CUDA libraries on the Jetson Nano that will cause this library to fail silently if you don't follow the instructions in the article to comment out a line in dlib and recompile it.

### Installing on Raspberry Pi 2+

  * [Raspberry Pi 2+ installation instructions](https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65)

### Installing on Windows

While Windows isn't officially supported, helpful users have posted instructions on how to install this library:

  * [@masoudr's Windows 10 installation guide (dlib + face_recognition)](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508)

## Python Module

You can import the `face_recognition` module and then easily manipulate
faces with just a couple of lines of code. It's super easy!

API Docs: [https://face-recognition.readthedocs.io](https://face-recognition.readthedocs.io/en/latest/face_recognition.html).

### Automatically find all the faces in an image

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image)

# face_locations is now an array listing the co-ordinates of each face!
```

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
 to try it out.

You can also opt-in to a somewhat more accurate deep-learning-based face detection model.

Note: GPU acceleration (via NVidia's CUDA library) is required for good
performance with this model. You'll also want to enable CUDA support
when compliling `dlib`.

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image, model="cnn")

# face_locations is now an array listing the co-ordinates of each face!
```

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)
 to try it out.

If you have a lot of images and a GPU, you can also
[find faces in batches](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py).

### Automatically locate the facial features of a person in an image

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)

# face_landmarks_list is now an array with the locations of each facial feature in each face.
# face_landmarks_list[0]['left_eye'] would be the location and outline of the first person's left eye.
```

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)
 to try it out.

### Recognize faces in images and identify who they are

```python
import face_recognition

picture_of_me = face_recognition.load_image_file("me.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

unknown_picture = face_recognition.load_image_file("unknown.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# Now we can see the two face encodings are of the same person with `compare_faces`!

results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")
```

See [this example](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)
 to try it out.

 ## Python Code Examples

All the examples are available [here](https://github.com/ageitgey/face_recognition/tree/master/examples).



## Face Detection

* [Find faces in a photograph](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
* [Find faces in a photograph (using deep learning)](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)
* [Find faces in batches of images w/ GPU (using deep learning)](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py)
* [Blur all the faces in a live video using your webcam (Requires OpenCV to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/blur_faces_on_webcam.py)

## Facial Features

* [Identify specific facial features in a photograph](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)
* [Apply (horribly ugly) digital make-up](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py)

## Facial Recognition

* [Find and recognize unknown faces in a photograph based on photographs of known people](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)
* [Identify and draw boxes around each person in a photo](https://github.com/ageitgey/face_recognition/blob/master/examples/identify_and_draw_boxes_on_faces.py)
* [Compare faces by numeric face distance instead of only True/False matches](https://github.com/ageitgey/face_recognition/blob/master/examples/face_distance.py)
* [Recognize faces in live video using your webcam - Simple / Slower Version (Requires OpenCV to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py)
* [Recognize faces in live video using your webcam - Faster Version (Requires OpenCV to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py)
* [Recognize faces in a video file and write out new video file (Requires OpenCV to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_video_file.py)
* [Recognize faces on a Raspberry Pi w/ camera](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_on_raspberry_pi.py)
* [Run a web service to recognize faces via HTTP (Requires Flask to be installed)](https://github.com/ageitgey/face_recognition/blob/master/examples/web_service_example.py)
* [Recognize faces with a K-nearest neighbors classifier](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py)
* [Train multiple images per person then recognize faces using a SVM](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_svm.py)

## Creating a Standalone Executable
If you want to create a standalone executable that can run without the need to install `python` or `face_recognition`, you can use [PyInstaller](https://github.com/pyinstaller/pyinstaller). However, it requires some custom configuration to work with this library. See [this issue](https://github.com/ageitgey/face_recognition/issues/357) for how to do it.

## Articles and Guides that cover `face_recognition`

- Ageitgey's article on how Face Recognition works: [Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
  - Covers the algorithms and how they generally work
- [Face recognition with OpenCV, Python, and deep learning](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) by Adrian Rosebrock
  - Covers how to use face recognition in practice
- [Raspberry Pi Face Recognition](https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/) by Adrian Rosebrock
  - Covers how to use this on a Raspberry Pi
- [Face clustering with Python](https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/) by Adrian Rosebrock
  - Covers how to automatically cluster photos based on who appears in each photo using unsupervised learning

## How Face Recognition Works

If you want to learn how face location and recognition work instead of
depending on a black box library, [read Ageitgey's article](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78).

## HOG - Histograms of Oriented Diagram for Human Detection
If you want to learn more how HOG work, read this [document](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)