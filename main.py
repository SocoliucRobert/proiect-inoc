import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import mediapipe as mp
import time

pdf_path = "carte.pdf"
doc = fitz.open(pdf_path)
current_page = 0
zoom_factor = 1.0

def render_pdf_page(page_num, zoom=1.0):
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(img)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)

page_image = render_pdf_page(current_page, zoom_factor)
last_gesture_time = 0
gesture_cooldown = 1.5
last_action = ""

TIP_IDS = [8, 12, 16, 20]
PIP_IDS = [6, 10, 14, 18]

while True:
    # Afișare PDF fără redimensionare fixă (zoom real)
    disp = page_image.copy()
    cv2.putText(disp, f"Pagina {current_page+1}/{len(doc)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("PDF Viewer", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))

    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    now = time.time()

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        fingers = [lm[tip].y < lm[pip].y for tip, pip in zip(TIP_IDS, PIP_IDS)]
        count = sum(fingers)
        action = ""

        if now - last_gesture_time > gesture_cooldown:
            if count == 4:
                current_page = min(current_page + 1, len(doc) - 1)
                action = "Pagina urmatoare"
                last_gesture_time = now
            elif count == 1 and fingers[0]:
                current_page = max(current_page - 1, 0)
                action = "Pagina anterioara"
                last_gesture_time = now
            elif count >= 2 and fingers[0] and fingers[1]:
                zoom_factor = min(zoom_factor + 0.1, 3.0)
                action = f"Zoom in ({zoom_factor:.1f}x)"
                last_gesture_time = now
            elif count >= 2 and fingers[2] and fingers[3]:
                zoom_factor = max(zoom_factor - 0.1, 0.5)
                action = f"Zoom out ({zoom_factor:.1f}x)"
                last_gesture_time = now

            if action:
                page_image = render_pdf_page(current_page, zoom_factor)
                last_action = action

    if last_action:
        cv2.putText(frame, f"Gest: {last_action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
doc.close()
