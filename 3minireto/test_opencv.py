import cv2
cap = cv2.VideoCapture(0)
print("Abierta:", cap.isOpened())
ok, frame = cap.read()
print("Frame leído:", ok, "shape:", frame.shape if ok else None)
cap.release()
