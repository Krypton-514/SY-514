import cv2
import mediapipe as mp
import pyautogui
import time
import math
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "hand_landmarker.task")


# =========================================================
# SETTINGS
# =========================================================

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

CURSOR_SENSITIVITY = 2.0
DEAD_ZONE = 0.003

PINCH_DISTANCE = 0.045
ACTION_DELAY = 0.6

SWIPE_THRESHOLD = 0.18
SWIPE_VERTICAL_THRESHOLD = 0.16

SCROLL_THRESHOLD = 0.012
ZOOM_STEP = 0.035

DOUBLE_CLICK_TIME = 0.45

# Remove PyAutoGUI's default 0.1s sleep after every call (causes lag)
pyautogui.PAUSE = 0


# =========================================================
# CHECK MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    print()
    print("ERROR: hand_landmarker.task not found!")
    print("Make sure this file exists:")
    print(MODEL_PATH)
    print()
    input("Press Enter to exit...")
    exit()


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    print("ERROR: Camera could not be opened!")
    detector.close()
    input("Press Enter to exit...")
    exit()


# =========================================================
# SCREEN
# =========================================================

screen_width, screen_height = pyautogui.size()


# =========================================================
# VARIABLES
# =========================================================

last_timestamp = -1

# Cursor
previous_x = None
previous_y = None

# Scroll (separate from cursor -> this was the crash)
scroll_prev_y = None

last_action_time = 0
last_pinch_time = 0

index_pinch_active = False
middle_pinch_active = False

gesture_start_x = None
gesture_start_y = None

previous_zoom_distance = None


# =========================================================
# HELPERS
# =========================================================

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def can_action(delay=ACTION_DELAY):
    global last_action_time
    current_time = time.time()
    if current_time - last_action_time >= delay:
        last_action_time = current_time
        return True
    return False


def show_text(frame, text, y=100, scale=0.8):
    cv2.putText(frame, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2)


def move_cursor_clamped(dx, dy):
    # Keep cursor off the screen corners so PyAutoGUI's
    # fail-safe never triggers and kills the program
    mx, my = pyautogui.position()
    nx = min(max(mx + dx, 1), screen_width - 2)
    ny = min(max(my + dy, 1), screen_height - 2)
    pyautogui.moveTo(nx, ny, duration=0)


# =========================================================
# FINGER DETECTION
# =========================================================

def fingers_up(hand):
    thumb = hand[4].x < hand[3].x
    index = hand[8].y < hand[6].y
    middle = hand[12].y < hand[10].y
    ring = hand[16].y < hand[14].y
    pinky = hand[20].y < hand[18].y
    return [int(thumb), int(index), int(middle), int(ring), int(pinky)]


# =========================================================
# BASIC GESTURE
# =========================================================

def get_gesture(fingers):
    if fingers == [0, 1, 0, 0, 0]:
        return "CURSOR"
    if fingers == [0, 1, 1, 0, 0]:
        return "SCROLL"
    if fingers == [1, 1, 1, 0, 0]:
        return "ZOOM"
    if fingers == [0, 1, 0, 0, 1]:
        return "SWITCH WINDOW"
    if fingers == [1, 1, 1, 1, 1]:
        return "OPEN PALM"
    if fingers == [0, 0, 0, 0, 0]:
        return "FIST"
    return "UNKNOWN"


# =========================================================
# START MESSAGE
# =========================================================

print()
print("====================================================")
print("           TOUCHLESS COMPUTER INTERFACE")
print("====================================================")
print()
print("GESTURES")
print("----------------------------------------------------")
print("Index finger            -> Move Cursor")
print("Thumb + Index pinch     -> Left Click")
print("Thumb + Middle pinch    -> Right Click")
print("Two quick pinches       -> Double Click / Open App")
print("Index + Middle          -> Scroll")
print("Index + Pinky           -> Switch Window")
print("Open Palm + Right       -> Next")
print("Open Palm + Left        -> Previous")
print("Open Palm + Up          -> Start Slideshow")
print("Open Palm + Down        -> End Slideshow")
print("Thumb+Index+Middle      -> Zoom (spread/close thumb-index)")
print()
print("Press Q to exit")
print("====================================================")
print()


# =========================================================
# MAIN LOOP
# =========================================================

try:
    while True:

        success, frame = cap.read()
        if not success:
            print("Camera frame error")
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # MediaPipe needs strictly increasing timestamps in ms
        timestamp = int(time.monotonic() * 1000)
        if timestamp <= last_timestamp:
            timestamp = last_timestamp + 1
        last_timestamp = timestamp

        result = detector.detect_for_video(mp_image, timestamp)

        # =================================================
        # HAND DETECTED
        # =================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            wrist = hand[0]
            thumb_tip = hand[4]
            index_tip = hand[8]
            middle_tip = hand[12]

            fingers = fingers_up(hand)
            gesture = get_gesture(fingers)

            # Draw index point
            x = int(index_tip.x * frame_width)
            y = int(index_tip.y * frame_height)
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # =============================================
            # CURSOR
            # =============================================

            if gesture == "CURSOR":

                current_x = index_tip.x
                current_y = index_tip.y

                if previous_x is None or previous_y is None:
                    previous_x = current_x
                    previous_y = current_y

                dx = current_x - previous_x
                dy = current_y - previous_y

                previous_x = current_x
                previous_y = current_y

                if abs(dx) < DEAD_ZONE:
                    dx = 0
                if abs(dy) < DEAD_ZONE:
                    dy = 0

                mouse_dx = int(dx * screen_width * CURSOR_SENSITIVITY)
                mouse_dy = int(dy * screen_height * CURSOR_SENSITIVITY)

                if mouse_dx != 0 or mouse_dy != 0:
                    move_cursor_clamped(mouse_dx, mouse_dy)

            else:
                previous_x = None
                previous_y = None

            # =============================================
            # LEFT CLICK / DOUBLE CLICK
            # =============================================

            index_pinch = distance(thumb_tip, index_tip) < PINCH_DISTANCE

            if index_pinch and not index_pinch_active:

                current_time = time.time()

                if current_time - last_pinch_time <= DOUBLE_CLICK_TIME:
                    pyautogui.doubleClick()
                    show_text(frame, "DOUBLE CLICK")
                    last_pinch_time = 0
                else:
                    pyautogui.click()
                    show_text(frame, "LEFT CLICK")
                    last_pinch_time = current_time

                index_pinch_active = True

            if not index_pinch:
                index_pinch_active = False

            # =============================================
            # RIGHT CLICK
            # =============================================

            middle_pinch = distance(thumb_tip, middle_tip) < PINCH_DISTANCE

            if middle_pinch and not index_pinch and not middle_pinch_active:
                pyautogui.click(button="right")
                show_text(frame, "RIGHT CLICK")
                middle_pinch_active = True

            if not middle_pinch:
                middle_pinch_active = False

            # =============================================
            # SCROLL
            # =============================================

            if gesture == "SCROLL":

                if scroll_prev_y is not None:
                    movement = index_tip.y - scroll_prev_y

                    if movement < -SCROLL_THRESHOLD:
                        pyautogui.scroll(3)
                    elif movement > SCROLL_THRESHOLD:
                        pyautogui.scroll(-3)

                scroll_prev_y = index_tip.y

            else:
                scroll_prev_y = None

            # =============================================
            # SWITCH WINDOW
            # =============================================

            if gesture == "SWITCH WINDOW":
                if can_action():
                    pyautogui.hotkey("alt", "tab")
                    show_text(frame, "SWITCH WINDOW")

            # =============================================
            # OPEN PALM SWIPES
            # =============================================

            if gesture == "OPEN PALM":

                current_x = wrist.x
                current_y = wrist.y

                if gesture_start_x is None:
                    gesture_start_x = current_x
                    gesture_start_y = current_y

                swipe_x = current_x - gesture_start_x
                swipe_y = current_y - gesture_start_y

                action = None

                if swipe_x > SWIPE_THRESHOLD:
                    action = ("right", "NEXT")
                elif swipe_x < -SWIPE_THRESHOLD:
                    action = ("left", "PREVIOUS")
                elif swipe_y < -SWIPE_VERTICAL_THRESHOLD:
                    action = ("f5", "START SLIDESHOW")
                elif swipe_y > SWIPE_VERTICAL_THRESHOLD:
                    action = ("esc", "END SLIDESHOW")

                if action is not None:
                    if can_action():
                        pyautogui.press(action[0])
                        show_text(frame, action[1])
                    gesture_start_x = None
                    gesture_start_y = None

            else:
                gesture_start_x = None
                gesture_start_y = None

            # =============================================
            # ZOOM
            # =============================================

            if gesture == "ZOOM":

                zoom_distance = distance(thumb_tip, index_tip)

                if previous_zoom_distance is not None:
                    change = zoom_distance - previous_zoom_distance

                    if change > ZOOM_STEP:
                        if can_action():
                            pyautogui.hotkey("ctrl", "+")
                            show_text(frame, "ZOOM IN", y=140)

                    elif change < -ZOOM_STEP:
                        if can_action():
                            pyautogui.hotkey("ctrl", "-")
                            show_text(frame, "ZOOM OUT", y=140)

                previous_zoom_distance = zoom_distance

            else:
                previous_zoom_distance = None

            # Show gesture name
            show_text(frame, gesture, y=50, scale=1)

        # =================================================
        # NO HAND
        # =================================================

        else:
            previous_x = None
            previous_y = None
            scroll_prev_y = None

            gesture_start_x = None
            gesture_start_y = None

            previous_zoom_distance = None

            index_pinch_active = False
            middle_pinch_active = False

            cv2.putText(frame, "NO HAND DETECTED", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # =================================================
        # DISPLAY
        # =================================================

        cv2.imshow("Touchless Computer Interface", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print()
    print("Touchless Interface stopped.")