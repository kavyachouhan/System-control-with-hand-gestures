import cv2  
import mediapipe as mp  
import math  
import pyautogui  
import numpy as np  
import screen_brightness_control as sbc   
import subprocess
import sys

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Create a window
cv2.namedWindow('System Control with hand gestures', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

# Move the window to the bottom-right corner of the screen
cv2.moveWindow('System Control with hand gestures', 1450, 635)

# Set the window to be non-resizable
cv2.setWindowProperty('System Control with hand gestures', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty('System Control with hand gestures', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

# Set the window to stay on top
cv2.setWindowProperty('System Control with hand gestures', cv2.WND_PROP_TOPMOST, 1)

def is_palm():
    # Check if the palm is facing the front
    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x:
        # Check if all fingers are open
        if all([
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
        ]):
            # Check if fingers are not touching each other
            fingers = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
            for i in range(len(fingers) - 1):
                for j in range(i + 1, len(fingers)):
                    bright_dist = ((hand_landmarks.landmark[fingers[i]].x - hand_landmarks.landmark[fingers[j]].x)**2 + (hand_landmarks.landmark[fingers[i]].y - hand_landmarks.landmark[fingers[j]].y)**2)**0.5
                    if bright_dist < 0.01: 
                        return False  # Fingers are touching

            text = "Neutral Gesture!"
            position = (50, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 0, 255)  
            cv2.putText(frame, text, position, font, 1, color, 2, cv2.LINE_AA)  # Draw the text on the frame
            cv2.circle(frame, (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])), radius=10, color=(0, 0, 255), thickness=2) # Draw a circle at the position of the thumb tip
            cv2.circle(frame, (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])), radius=10, color=(0, 0, 255), thickness=2) # Draw a circle at the position of the index finger tip
            cv2.circle(frame, (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0])), radius=10, color=(0, 0, 255), thickness=2)  # Draw a circle at the position of the middle finger tip
            cv2.circle(frame, (int(ring_tip.x * frame.shape[1]), int(ring_tip.y * frame.shape[0])), radius=10, color=(0, 0, 255), thickness=2) # Draw a circle at the position of the ring finger tip
            cv2.circle(frame, (int(pinky_tip.x * frame.shape[1]), int(pinky_tip.y * frame.shape[0])), radius=10, color=(0, 0, 255), thickness=2)  # Draw a circle at the position of the pinky finger tip

            return True  # Palm is detected
    return False  # Palm is not detected


def virtual_mouse_calc():
    # Calculate the vectors representing the thumb to index finger and thumb to middle finger
    vector_thumb_index = [index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y]
    vector_thumb_middle = [middle_tip.x - thumb_tip.x, middle_tip.y - thumb_tip.y]

    # Calculate the dot product of the vectors
    dot_product = vector_thumb_index[0]*vector_thumb_middle[0] + vector_thumb_index[1]*vector_thumb_middle[1]

    # Calculate the magnitudes of the vectors
    magnitude_thumb_index = math.sqrt(vector_thumb_index[0]**2 + vector_thumb_index[1]**2)
    magnitude_thumb_middle = math.sqrt(vector_thumb_middle[0]**2 + vector_thumb_middle[1]**2)

    # Calculate the angle between the vectors using the dot product and magnitudes
    angle = math.degrees(math.acos(dot_product / (magnitude_thumb_index * magnitude_thumb_middle)))

    # Calculate the distances from the base of the palm to the tips of the ring and pinky fingers
    distance_ring = math.sqrt((ring_tip.x - palm_base.x)**2 + (ring_tip.y - palm_base.y)**2)
    distance_pinky = math.sqrt((pinky_tip.x - palm_base.x)**2 + (pinky_tip.y - palm_base.y)**2)

    # Check if the index finger is close to the thumb
    distance_left = math.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)

    # Check if the middle finger is close to the thumb
    distance_right = math.sqrt((middle_tip.x - thumb_tip.x)**2 + (middle_tip.y - thumb_tip.y)**2)

    return angle, distance_ring, distance_pinky, distance_left, distance_right

def virtual_mouse_func(): 
    screen_width, screen_height = pyautogui.size()  # Get the size of the screen
    text = "Moving Mouse!"
    position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255) 
    pyautogui.moveTo(thumb_tip.x * screen_width, thumb_tip.y * screen_height)  # Move the mouse to the position of the thumb tip
    cv2.putText(frame, text, position, font, 1, color, 2, cv2.LINE_AA)   
    cv2.circle(frame, (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])), radius=10, color=(0, 0, 255), thickness=2)  
    cv2.circle(frame, (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])), radius=10, color=(0, 0, 255), thickness=2)  
    cv2.circle(frame, (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0])), radius=10, color=(0, 0, 255), thickness=2)  

def left_click():
    pyautogui.leftClick()  # Perform a left click
    text = "Left Click!"
    position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255) 
    cv2.putText(frame, text, position, font, 1, color, 2, cv2.LINE_AA)   
    cv2.circle(frame, (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])), radius=10, color=(0, 0, 255), thickness=2) 

def right_click():
    pyautogui.rightClick()  # Perform a right click
    text = "Right Click!"
    position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)  
    cv2.putText(frame, text, position, font, 1, color, 2, cv2.LINE_AA)   
    cv2.circle(frame, (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0])), radius=10, color=(0, 0, 255), thickness=2)  

# Previous position of the pinch for brightness
bright_prev_pinch_pos = None

def brightness_controll_calc():
    # Calculate the distances from the base of the palm to the tips of the middle, ring, and pinky fingers
    distance_middle = math.sqrt((middle_tip.x - palm_base.x)**2 + (middle_tip.y - palm_base.y)**2)
    distance_ring = math.sqrt((ring_tip.x - palm_base.x)**2 + (ring_tip.y - palm_base.y)**2)
    distance_pinky = math.sqrt((pinky_tip.x - palm_base.x)**2 + (pinky_tip.y - palm_base.y)**2)

    return distance_ring, distance_pinky, distance_middle 

def bright_calculate_position(hand_landmarks):
    thumb_bright_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
    index_bright_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
    bright_dist = np.linalg.norm(thumb_bright_tip - index_bright_tip)
    return thumb_bright_tip, index_bright_tip, bright_dist 

def bright_execute(current_pinch_pos, bright_prev_pinch_pos, frame):
    textup = "Brightness Increased" 
    textdown = "Brightness Decreased" 
    position = (50, 50) 
    font = cv2.FONT_HERSHEY_SIMPLEX  
    color = (0, 0, 255)  

    if bright_prev_pinch_pos is not None:
        if current_pinch_pos[0] > bright_prev_pinch_pos[0]:
            current_brightness = sbc.get_brightness()[0]
            sbc.set_brightness(current_brightness + 5)
            cv2.putText(frame, textup, position, font, 1, color, 2, cv2.LINE_AA)
        elif current_pinch_pos[0] < bright_prev_pinch_pos[0]:
            current_brightness = sbc.get_brightness()[0]
            sbc.set_brightness(current_brightness - 5)
            cv2.putText(frame, textdown, position, font, 1, color, 2, cv2.LINE_AA)
    return frame


def delete_calc(hand_landmarks):
        # Check if index finger and thumb are open and other fingers are closed
        index_finger_is_open = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        thumb_is_open = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
        other_fingers_are_closed = sum([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
                                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y]) == 3

        return index_finger_is_open and thumb_is_open and other_fingers_are_closed

def delete_func():
        decreaseText = "Deleted" 
        position = (50, 50) 
        font = cv2.FONT_HERSHEY_SIMPLEX  
        color = (0, 0, 255)

        pyautogui.hotkey('del')
        pyautogui.sleep(0.5)
        cv2.putText(frame, decreaseText, position, font, 1, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])), radius=10, color=(0, 0, 255), thickness=2) # Draw a circle at the position of the thumb tip
        cv2.circle(frame, (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])), radius=10, color=(0, 0, 255), thickness=2) # Draw a circle at the position of the index finger tip
        cv2.circle(frame, (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0])), radius=10, color=(0, 0, 255), thickness=2)  # Draw a circle at the position of the middle finger tip

def thumbs_up_calc(hand_landmarks):
    # Check if the palm is facing the front
    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x:
        # Check if all fingers are closed and thumb is open
        if all([
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
        ]):
            return True
    return False

def thumbs_up_func():
    cap.release()
    cv2.destroyAllWindows()
    subprocess.Popen(["python", "air_canvas.py"])
    sys.exit()

# Initialize the current gesture as neutral
current_gesture = "neutral"

# Initialize a flag for whether a gesture is being performed
gesture_in_progress = False

while cap.isOpened():  # While the video capture is open
    ret, frame = cap.read()  # Read a frame from the video capture
    if not ret:  # If the frame was not read correctly, skip this iteration
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame_height, frame_width, _ = frame.shape  # Get the height and width of the frame

    # Convert the BGR image to RGB and process it with MediaPipe Hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Draw hand landmarks on the image
    if results.multi_hand_landmarks:  # If hand landmarks were detected
        for hand_landmarks in results.multi_hand_landmarks:  # For each set of hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw the hand landmarks on the frame

            landmarks = hand_landmarks.landmark  # Get the landmarks
            # Get the landmarks for the thumb, index finger, middle finger, ring finger, pinky, and base of the palm
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            angle, distance_ring, distance_pinky, distance_left, distance_right = virtual_mouse_calc()
            distance_ring, distance_pinky, distance_middle = brightness_controll_calc()
            thumb_bright_tip, index_bright_tip, bright_dist = bright_calculate_position(hand_landmarks)

            # Check for neutral gesture
            if is_palm():
                current_gesture = "neutral"
                gesture_in_progress = False  # Reset the gesture flag

            # Only recognize other gestures if no gesture is currently being performed
            if not gesture_in_progress:
                # Check for virtual mouse control
                if 30 < angle < 60 and distance_ring < 0.2 and distance_pinky < 0.2 and distance_left > 0.1 and distance_right > 0.1:   
                    virtual_mouse_func()  # Call the virtual mouse function
                    current_gesture = "virtual_mouse"

                # Check for left click
                elif distance_left < 0.1 and distance_right > 0.1 and distance_ring < 0.2 and distance_pinky < 0.2: 
                    left_click()  # Call the left click function
                    current_gesture = "left_click"

                # Check for right click
                elif distance_right < 0.1 and distance_left > 0.1 and distance_ring < 0.2 and distance_pinky < 0.2:          
                    right_click()  # Call the right click function
                    current_gesture = "right_click"

                # Check for brightness control
                elif bright_dist < 0.1: 
                    current_pinch_pos = (thumb_bright_tip + index_bright_tip) / 2
                    frame = bright_execute(current_pinch_pos, bright_prev_pinch_pos, frame)
                    bright_prev_pinch_pos = current_pinch_pos
                    current_gesture = "brightness_control"

                elif delete_calc(hand_landmarks):
                    delete_func()
                    current_gesture = "delete"
                # Check for thumbs up
                elif thumbs_up_calc(hand_landmarks):
                    thumbs_up_func()
                    current_gesture = "thumbs_up"

                    # If a gesture is recognized, set the gesture flag
                    if current_gesture != "neutral":
                        gesture_in_progress = True

    else:
        # If no hand landmarks are detected, do nothing
        pass        

    # Resize the captured frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

    # Display the resulting frame in the small window
    cv2.imshow('System Control with hand gestures', small_frame)

    # Display the image
    cv2.imshow('System Control with hand gestures', frame)

    # Exit if 'esc' is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()  # Release the video capture
cv2.destroyAllWindows()  # Close all OpenCV windows
