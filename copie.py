import cv2 
import fitz
import numpy as np
from PIL import Image
import mediapipe as mp
import time
import tkinter as tk
from tkinter import filedialog

HELP_TEXT = [
    "Gesturi disponibile:",
    "Cap dreapta - pagina urmatoare",
    "Cap stanga - pagina anterioara",
    "Umar drept - zoom in",
    "Umar stang - zoom out",
    "Nas - prima pagina si reset zoom",
    "Gat - help",
    "Piept - quit (iesire)",
    "Like - confirma gestul",
    "Dislike - anuleaza gestul"
]

COLOR_TITLE = (255, 0, 0)     # albastru
COLOR_GEST = (255, 0, 255)    # roz
COLOR_CONFIRM = (0, 180, 255) # galben-portocaliu

def convert_pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(np.array(img))
    doc.close()
    return pages

def select_pdf_file_from_menu():
    selected = []
    def handle_file():
        file_path = filedialog.askopenfilename(
            title="Alege un fisier PDF",
            filetypes=[("Fisiere PDF", "*.pdf")]
        )
        if not file_path:
            return
        selected.extend(convert_pdf_to_images(file_path))
        root.destroy()
    root = tk.Tk()
    root.title("Incarca fisier PDF")
    root.geometry("400x200")
    btn = tk.Button(root, text="Incarca fisier PDF", command=handle_file, font=("Arial", 18))
    btn.pack(expand=True)
    root.mainloop()
    return selected

def apply_zoom(img, factor):
    if abs(factor - 1.0) < 1e-2:
        return img
    h, w = img.shape[:2]
    if factor > 1.0:
        new_w, new_h = int(w / factor), int(h / factor)
        start_x = max((w - new_w) // 2, 0)
        start_y = max((h - new_h) // 2, 0)
        cropped = img[start_y:start_y+new_h, start_x:start_x+new_w]
        return cv2.resize(cropped, (w, h))
    else:
        out = np.zeros_like(img)
        scale = factor
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        out[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        return out

def show_help(frame):
    y0 = 60
    for line in HELP_TEXT:
        cv2.putText(frame, line, (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TITLE, 2)
        y0 += 38
    return frame

def is_thumbs_up(hand_landmarks):
    # Like: deget mare ridicat, restul jos
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    other_fingers = [index_tip, middle_tip, ring_tip, pinky_tip]
    thumb_up = all(thumb_tip.y < f.y for f in other_fingers)
    rest_down = all(f.y > thumb_tip.y + 0.05 for f in other_fingers)
    return thumb_up and rest_down

def is_thumbs_down(hand_landmarks):
    # Dislike: deget mare jos, restul sus
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    other_fingers = [index_tip, middle_tip, ring_tip, pinky_tip]
    thumb_down = all(thumb_tip.y > f.y for f in other_fingers)
    rest_up = all(f.y < thumb_tip.y - 0.05 for f in other_fingers)
    return thumb_down and rest_up

def run_pdf_viewer(pages):
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)
    current_page = 0
    zoom_factor = 1.0
    last_gesture_time = 0
    gesture_cooldown = 1.0
    last_action = ""
    pending_action = None
    pending_gesture = None
    help_start_time = None
    show_help_screen = False

    candidate_gesture = None
    candidate_action = None
    candidate_since = None
    ARM_DELAY = 2.0  # 2 secunde pentru armat gest

    def get_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def is_close(hand, target, threshold=0.06):
        return get_distance(hand, target) < threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_pose = pose.process(rgb)
        result_hands = hands.process(rgb)
        now = time.time()
        action = ""

        hand_landmarks = None
        if result_hands.multi_hand_landmarks:
            hand_landmarks = result_hands.multi_hand_landmarks[0]

        detected_gesture = None
        detected_action = None

        if result_pose.pose_landmarks and hand_landmarks:
            pose_lm = result_pose.pose_landmarks.landmark
            hand_lm = hand_landmarks.landmark

            mp_drawing.draw_landmarks(frame, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_tip = hand_lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            hand_point = np.array([hand_tip.x, hand_tip.y])

            mouth_left = pose_lm[mp_pose.PoseLandmark.MOUTH_LEFT]
            mouth_right = pose_lm[mp_pose.PoseLandmark.MOUTH_RIGHT]
            mouth_center = np.array([(mouth_left.x + mouth_right.x)/2,
                                     (mouth_left.y + mouth_right.y)/2 + 0.07])
            chest = np.array([pose_lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                              (pose_lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y + pose_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)/2 + 0.07])

            keypoints = {
                "cap_dreapta": np.array([pose_lm[mp_pose.PoseLandmark.RIGHT_EAR].x,
                                         pose_lm[mp_pose.PoseLandmark.RIGHT_EAR].y]),
                "cap_stanga": np.array([pose_lm[mp_pose.PoseLandmark.LEFT_EAR].x,
                                        pose_lm[mp_pose.PoseLandmark.LEFT_EAR].y]),
                "umar_drept": np.array([pose_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                        pose_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]),
                "umar_stang": np.array([pose_lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                        pose_lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]),
                "nas": np.array([pose_lm[mp_pose.PoseLandmark.NOSE].x,
                                 pose_lm[mp_pose.PoseLandmark.NOSE].y]),
                "gat": mouth_center,
                "piept": chest
            }

            if now - last_gesture_time > gesture_cooldown:
                if is_close(hand_point, keypoints["cap_dreapta"]):
                    detected_gesture = "pagina_urm"
                    detected_action = "Pagina urmatoare"
                elif is_close(hand_point, keypoints["cap_stanga"]):
                    detected_gesture = "pagina_ant"
                    detected_action = "Pagina anterioara"
                elif is_close(hand_point, keypoints["umar_drept"]):
                    detected_gesture = "zoom_in"
                    detected_action = "Zoom in"
                elif is_close(hand_point, keypoints["umar_stang"]):
                    detected_gesture = "zoom_out"
                    detected_action = "Zoom out"
                elif is_close(hand_point, keypoints["nas"]):
                    detected_gesture = "home"
                    detected_action = "Home"
                elif is_close(hand_point, keypoints["gat"], 0.10):
                    detected_gesture = "help"
                    detected_action = "Help "
                elif is_close(hand_point, keypoints["piept"], 0.10):
                    detected_gesture = "quit"
                    detected_action = "Quit"

        # Gestionare candidati pentru pending
        if pending_action is not None:
            if hand_landmarks and is_thumbs_up(hand_landmarks):
                action = pending_action
                if pending_gesture == "pagina_urm":
                    current_page = min(current_page + 1, len(pages) - 1)
                elif pending_gesture == "pagina_ant":
                    current_page = max(current_page - 1, 0)
                elif pending_gesture == "zoom_in":
                    zoom_factor = min(zoom_factor + 0.2, 3.0)
                elif pending_gesture == "zoom_out":
                    zoom_factor = max(zoom_factor - 0.2, 0.5)
                elif pending_gesture == "home":
                    current_page = 0
                    zoom_factor = 1.0
                elif pending_gesture == "help":
                    show_help_screen = True
                    help_start_time = now
                elif pending_gesture == "quit":
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(0)
                pending_action = None
                pending_gesture = None
                last_gesture_time = now
                last_action = "Confirmat cu like!"
            elif hand_landmarks and is_thumbs_down(hand_landmarks):
                # Anulare gest!
                last_action = "Gest anulat cu dislike!"
                pending_action = None
                pending_gesture = None
                candidate_gesture = None
                candidate_action = None
                candidate_since = None
        else:
            if detected_gesture:
                if candidate_gesture != detected_gesture:
                    candidate_gesture = detected_gesture
                    candidate_action = detected_action
                    candidate_since = now
                elif candidate_since and now - candidate_since > ARM_DELAY:
                    pending_gesture = candidate_gesture
                    pending_action = candidate_action
                    candidate_gesture = None
                    candidate_action = None
                    candidate_since = None
                    last_action = "Like = confirmare/ dislike = anulare!"
            else:
                candidate_gesture = None
                candidate_action = None
                candidate_since = None

        # Afisare help pe webcam pentru 5 secunde
        if show_help_screen:
            if help_start_time and now - help_start_time < 5:
                frame = show_help(frame)
            else:
                show_help_screen = False

        img = apply_zoom(pages[current_page].copy(), zoom_factor)
        disp = img.copy()
        cv2.putText(disp, f"Pagina {current_page+1}/{len(pages)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TITLE, 2)
        cv2.imshow("PDF Viewer", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))

        if last_action:
            cv2.putText(frame, f"Gest: {last_action}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_GEST, 2)
        if pending_action:
            cv2.putText(frame, f"{pending_action} confirma sau anuleaza!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_CONFIRM, 2)
        elif candidate_action:
            cv2.putText(frame, f"{candidate_action} - mentine gestul...", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TITLE, 2)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

while True:
    pages = select_pdf_file_from_menu()
    if not pages:
        print("Niciun fisier PDF selectat. Inchidere.")
        break
    run_pdf_viewer(pages)
