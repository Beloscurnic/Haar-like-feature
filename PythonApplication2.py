
import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict

path = r"test_images\F28B71.jpg"

def blur(image_path, kernel_size=(5, 5)):
    img = cv2.imread(image_path)

    fig, ax = plt.subplots(ncols=2, figsize=(5, 5))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original_image')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # получить координаты и размеры найденного лица
    x, y, w, h = faces[0]
    roi = img[y:y+h, x:x+w]
    blurred_roi = cv2.GaussianBlur(roi, (15, 15), 3)
    img[y:y+h, x:x+w] = blurred_roi

    # сохранение результата
    cv2.imwrite('blurred.jpg', img)
    ax[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Blurred')

    plt.show()


def sharpen(image_path):
    img = cv2.imread(image_path)

    kernel = np.array([[0,-1,0],
                       [-1, 5,-1],
                       [0,-1,0]])

    sharpened = cv2.filter2D(img, -1, kernel)

    fig, ax = plt.subplots(ncols=2, figsize=(5, 5))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original_image')
    ax[1].imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Sharpened_image')

    plt.show()

def detect_face(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None
    else:
        (x, y, w, h) = faces[0]
        return (x, y, x+w, y+h)


def recognize_face(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Face Recognition', img)
        cv2.waitKey(0)



def accepted(image_path):
    img = cv2.imread(image_path)
    # проверка, является ли изображение цветным
    if len(img.shape) ==2:
        # преобразует изображение в другой цветовой формат,
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    height, width, _ = img.shape
    # соотношение сторон изображения на соответствие лицу.
    if height < width and width!=height:
        return False


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minSize=(30, 30))

    if len(faces) != 1:
        return False
    # получить координаты и размеры найденного лица
    x, y, w, h = faces[0]

    # вычислить площадь найденного лица и общую площадь изображения
    face_area = w * h
    image_area = img.shape[0] * img.shape[1]

    # вычислить процент площади изображения, занимаемый лицом
    percent_face_area = face_area / image_area * 100
    if percent_face_area<20 and percent_face_area>50:
        print('faces do not occupy 70-80% of the area!')
        return False

    # определяем количество глаз в области леца
    roi_gray = gray[y:y+h, x:x+w]
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eyes_cascade.detectMultiScale(roi_gray)
    if len(eyes) != 2:
        return False
    # Проверяет, что разница между вертикальными координатами двух глаз меньше или равна 5 пикселям. 
    (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes
    if abs(ey1 - ey2) > 5:
        return False

    return True

blur(path)
sharpen(path)
detect_face(path)

recognize_face(path)
# (is_photo_accepted(path))

with open('test.csv', 'r') as file:
    data = defaultdict(list)
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        key = row[0]
        value = row[1] == 'True'
        data[key].append(value)

count = 0
for key, values in data.items():
    for value in values:
        if accepted(key) == value:
            count += 1

accuracy = count / len(data.items())
print("Accuracy: ", accuracy)