import cv2

pipeline = (
    "nvarguscamerasrc sensor-mode=4 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, width=320, height=240, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=1"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
print("Abierta:", cap.isOpened())

for i in range(5):
    ok, frame = cap.read()
    print(f"Frame {i}: ok={ok}, shape={frame.shape if ok else None}")

cap.release()