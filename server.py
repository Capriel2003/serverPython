from flask import Flask, Response
from flask_cors import CORS
import cv2 as cv
import numpy as np

app = Flask(__name__)
CORS(app)

def extrairMaiorCtn(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgTh = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 12)
    kernel = np.ones((2, 2), np.uint8)
    imgDil = cv.dilate(imgTh, kernel)
    contours, _ = cv.findContours(imgDil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    maiorCtn = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(maiorCtn)
    bbox = [x, y, w, h]
    recorte = img[y:y+h, x:x+w]
    recorte = cv.resize(recorte, (400, 750))
    return recorte, bbox

def gen_frames():
    video = cv.VideoCapture(0)
    while True:
        success, imagem = video.read()
        if not success:
            break
            
        imagem = cv.resize(imagem, (600, 700))
        imgContours = imagem.copy()
        gabarito, bbox = extrairMaiorCtn(imagem)
        imgGray2 = cv.cvtColor(gabarito, cv.COLOR_BGR2GRAY)
        imgBlur = cv.GaussianBlur(imgGray2, (5, 5), 1)
        imgCanny = cv.Canny(imgBlur, 10, 50)
        contours, _ = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(imgContours, contours, -1, (0, 255, 0), 2)
        cv.rectangle(imagem, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 3)

        # Codifique o frame para JPEG
        ret, buffer = cv.imencode('.jpg', imgContours)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
