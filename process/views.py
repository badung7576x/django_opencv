from unittest import result

from django.http.response import StreamingHttpResponse
from django.shortcuts import render
from django.conf import settings
from .forms import objectCountingForm
from .object_counting import counting
import os
import numpy as np
import cv2
from os import path
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def objectCounting(request):
    if request.method == 'POST':
        form = objectCountingForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            imageUrl = readImage(image)
            type = int(request.POST['type'])
            result = counting(imageUrl, type)

            if path.exists(settings.MEDIA_ROOT / 'images/input.jpg'):
                os.remove(settings.MEDIA_ROOT / 'images/input.jpg')
            inputUrl = upload(image, 'input.jpg')
            outputUrl = os.path.join(settings.MEDIA_ROOT / 'images', 'output.jpg')
            cv2.imwrite(outputUrl, result['outputImg'])

            return render(request, 'object_counting.html',
                          {'input': imageUrl, 'output': outputUrl, 'count': result['count']})

    return render(request, 'object_counting.html', )


def faceDetection(request):
    return render(request, 'face_detection.html', )


def stream():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            break

        frame = cv2.flip(frame, 1)
        detect_faces(frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('frame.jpg', 'rb').read() + b'\r\n')


def detect_faces(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_coordinates = face_cascade.detectMultiScale(gray_img)
    for coordinate in face_coordinates:
        (x, y, w, h) = coordinate
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    cv2.imwrite('frame.jpg', image)


def camera(request):
    return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=frame')


def upload(f, fileName):
    file = open(os.path.join(settings.MEDIA_ROOT / 'images', fileName), 'wb+')
    for chunk in f.chunks():
        file.write(chunk)
    return settings.MEDIA_ROOT / 'images' / fileName


def readImage(img):
    img = img.read()
    img = np.asarray(bytearray(img), dtype="uint8")
    return cv2.imdecode(img, cv2.IMREAD_COLOR)
