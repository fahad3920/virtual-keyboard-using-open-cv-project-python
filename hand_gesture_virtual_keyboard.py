import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
time.sleep(2)

# Keyboard layout
keyboard_rows = [
    ["ESC", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "DEL"],
    ["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "BACKSPACE"],
    ["TAB", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]", "\\"],
    ["CAPS", "A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'", "ENTER"],
    ["SHIFT", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "SHIFT"],
    ["CTRL", "WIN", "ALT", "SPACE", "ALT", "WIN", "MENU", "CTRL"]
]

key_size = (60, 60)
key_spacing = 10
start_x = 50
start_y = 300

# Typing state
last_pressed_key = None
typed_text = ""
last_key_time = 0
key_cooldown = 0.3

def draw_virtual_keyboard(frame, highlight_key=None):
    y = start_y
    for row in keyboard_rows:
        x = start_x
        for key in row:
            w = key_size[0]
            if key == "SPACE":
                w = key_size[0] * 5 + key_spacing * 4
            elif key in ["BACKSPACE", "ENTER", "SHIFT", "CAPS", "TAB"]:
                w = key_size[0] * 2 + key_spacing
            elif key in ["CTRL", "WIN", "ALT", "MENU", "ESC", "DEL"]:
                w = key_size[0] + 10

            color = (0, 255, 0) if key == highlight_key else (255, 0, 0)
            thickness = -1 if key == highlight_key else 2
            cv2.rectangle(frame, (x, y), (x + w, y + key_size[1]), color, thickness)

            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (key_size[1] + text_size[1]) // 2
            cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            x += w + key_spacing
        y += key_size[1] + key_spacing

def get_key_at_pos(x, y):
    y_pos = start_y
    for row in keyboard_rows:
        x_pos = start_x
        for key in row:
            w = key_size[0]
            if key == "SPACE":
                w = key_size[0] * 5 + key_spacing * 4
            elif key in ["BACKSPACE", "ENTER", "SHIFT", "CAPS", "TAB"]:
                w = key_size[0] * 2 + key_spacing
            elif key in ["CTRL", "WIN", "ALT", "MENU", "ESC", "DEL"]:
                w = key_size[0] + 10

            if x_pos <= x <= x_pos + w and y_pos <= y <= y_pos + key_size[1]:
                return key
            x_pos += w + key_spacing
        y_pos += key_size[1] + key_spacing
    return None

def distance(lm1, lm2):
    return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

def process_hand(landmarks, width, height):
    global last_pressed_key, typed_text, last_key_time
    current_time = time.time()
    ix, iy = int(landmarks[8].x * width), int(landmarks[8].y * height)
    click_dist = distance(landmarks[8], landmarks[4])
    key = get_key_at_pos(ix, iy)

    if key and click_dist < 0.05 and (current_time - last_key_time > key_cooldown):
        if key == "SPACE":
            pyautogui.press("space")
            typed_text += " "
        elif key == "BACKSPACE":
            pyautogui.press("backspace")
            typed_text = typed_text[:-1]
        elif key == "ENTER":
            pyautogui.press("enter")
            typed_text += "\n"
        elif key in ["SHIFT", "CAPS", "TAB", "CTRL", "WIN", "ALT", "MENU", "ESC", "DEL"]:
            pyautogui.press(key.lower())
        else:
            pyautogui.press(key.lower())
            typed_text += key
        last_pressed_key = key
        last_key_time = current_time
    else:
        last_pressed_key = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            process_hand(hand_landmarks.landmark, frame.shape[1], frame.shape[0])
    else:
        last_pressed_key = None

    draw_virtual_keyboard(frame, highlight_key=last_pressed_key)
    cv2.putText(frame, typed_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    cv2.namedWindow("Hand Gesture Keyboard", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Hand Gesture Keyboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Hand Gesture Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
