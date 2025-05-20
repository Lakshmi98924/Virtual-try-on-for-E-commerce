import cv2
import numpy as np

# Overlay function
def overlay_image(bg, fg, x, y, w, h):
    fg = cv2.resize(fg, (w, h))

    if fg.shape[2] == 4:  # Transparent image
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (
                fg[:, :, c] * alpha + bg[y:y+h, x:x+w, c] * (1.0 - alpha)
            )
    else:  # No transparency
        fg_rgb = fg[:, :, :3]
        bg[y:y+h, x:x+w] = fg_rgb

    return bg

# Load accessory image
accessory = cv2.imread(r'C:\Users\laksh\Downloads\sun.jpeg', cv2.IMREAD_UNCHANGED)
if accessory is None:
    print("Accessory image not found!")
    exit()

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        y_offset = y + int(h / 4.5)  # Adjust y position for better fitting
        frame = overlay_image(frame, accessory, x, y_offset, w, int(h / 3.5))

    cv2.imshow("Virtual Try-On", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()




