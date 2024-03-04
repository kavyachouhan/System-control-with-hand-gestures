import cv2
import mediapipe as mp
import numpy as np
import subprocess
import sys

# Initialize MediaPipe Hand module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

# Create a blank whiteboard to draw on
canvas = None

# Initialize x1, y1 points
x1, y1 = None, None

# Set the initial drawing color to red
color = [0, 0, 255]

# Define color palette (BGR)
colors = {
    'red': [0, 0, 255],
    'green': [0, 255, 0],
    'blue': [255, 0, 0],
    'white': [255, 255, 255]
}

# Define buttons
buttons = {
    'red': (10, 10, 50, 50),
    'green': (70, 10, 110, 50),
    'blue': (130, 10, 170, 50),
    'white': (190, 10, 230, 50),
    'save': (310, 10, 350, 50),  
    'clear': (250, 10, 290, 50)
}

def draw_buttons(frame, buttons, selected_color):
    for color in colors.keys():
        cv2.rectangle(frame, buttons[color][:2], buttons[color][2:], colors[color], -1)
        if color == selected_color:
            cv2.rectangle(frame, buttons[color][:2], buttons[color][2:], (0, 0, 0), 3)
    cv2.rectangle(frame, buttons['clear'][:2], buttons['clear'][2:], (0, 0, 0), -1)
    cv2.putText(frame, 'Clear', (buttons['clear'][0], buttons['clear'][3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, buttons['save'][:2], buttons['save'][2:], (128, 128, 128), -1)
    cv2.putText(frame, 'Save', (buttons['save'][0], buttons['save'][3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(frame)

        # Convert the image color back so it can be displayed
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Initialize the canvas as a black image of the same size as the frame
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Draw the hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the position of the index finger tip
                x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                # Check if the index finger tip is over any button
                for button in buttons.keys():
                    bx1, by1, bx2, by2 = buttons[button]
                    if bx1 < x2 < bx2 and by1 < y2 < by2:
                        if button == 'clear':
                            canvas = np.zeros_like(frame)
                        elif button == 'save':
                            cv2.imwrite('canvas.png', canvas)  # Save the canvas when the 'save' button is pressed
                        else:
                            color = colors[button]
                        break
                
                # Get the position of the thumb tip
                x_thumb = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1])
                y_thumb = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])

                # Calculate the distance between the index finger tip and thumb tip
                distance = np.sqrt((x2 - x_thumb)**2 + (y2 - y_thumb)**2)

                # If the distance between the index finger tip and thumb tip is less than 30, consider it as a pinch gesture
                if distance > 30: 
                    if x1 and y1:
                        # Draw on the canvas
                        canvas = cv2.line(canvas, (x1, y1), (x2, y2), color, 4)

                    # After the line is drawn, the new points become the previous points
                    x1, y1 = x2, y2
                else:
                    x1, y1 = None, None
        else:
            # If no hand is found, reset the points
            x1, y1 = None, None

        # Merge the canvas and the frame
        frame = cv2.add(frame, canvas)

        # Draw the buttons on the frame
        draw_buttons(frame, buttons, color)

        # Display the frame
        cv2.imshow('Air Canvas', frame)
        cv2.imshow('Drawing', canvas)  # Display the canvas in a separate window

        # Break the loop on 'esc'
        if cv2.waitKey(5) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            subprocess.Popen(["python", "system_control_with_hand_gestures.py"])
            sys.exit()

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
