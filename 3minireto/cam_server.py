from flask import Flask, Response
import cv2

GST = ("nvarguscamerasrc sensor-mode=4 ! "
         "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
         "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
         "video/x-raw,format=BGR ! appsink drop=1")

cap = cv2.VideoCapture(GST, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("No pude abrir la cámara CSI")

app = Flask(__name__)

def frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b'--f\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')

@app.route('/')
def index():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=f')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
