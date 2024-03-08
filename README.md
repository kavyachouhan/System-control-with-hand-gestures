# System Control with Hand Gestures

Welcome to the 'System Control with Hand Gestures' project! This project allows you to control your system using simple hand gestures. It's an innovative way to interact with your computer, providing a more intuitive and natural method of control. While this is a first year project, it's accuracy is not 100%.

# Features
 click on dropdown to know more <br>

### Gesture Recognition:
<details>
<summary>Neutral Gesture</summary>
 <figure>
  <img src="https://github.com/xenon-19/Gesture_Controller/blob/9be82cfc75aa4c04fff0e12dd4de853f9d83a101/demo_media/palm.gif" alt="Palm" width="711" height="400"><br>
  <figcaption>Neutral Gesture. Used to halt/stop execution of current gesture.</figcaption>
</figure>
</details>
 

<details>
<summary>Virtual Mouse</summary>
  <img src="https://github.com/xenon-19/Gesture_Controller/blob/e20edfb1f368ffa600d96bd91031942ec97cb2ab/demo_media/move%20mouse.gif" alt="Move Cursor" width="711" height="400"><br>
  <figcaption>Control your mouse cursor with hand movements. Cursor is assigned to the midpoint of index and middle fingertips.</figcaption>
</details>

<details>
<summary>Left Click</summary>
<img src="https://github.com/xenon-19/Gesture_Controller/blob/9be82cfc75aa4c04fff0e12dd4de853f9d83a101/demo_media/left%20click.gif" alt="Left Click" width="711" height="400"><br>
 <figcaption>Perform left mouse clicks when index fingertip on thumb.</figcaption>
</details>

<details>
<summary>Right Click</summary>
<img src="https://github.com/xenon-19/Gesture_Controller/blob/9be82cfc75aa4c04fff0e12dd4de853f9d83a101/demo_media/right%20click.gif" alt="Right Click" width="711" height="400"><br>
 <figcaption>Perform right mouse clicks when middle fingertip on thumb.</figcaption>
</details>

<details>
<summary>Double Click</summary>
<img src="https://github.com/xenon-19/Gesture_Controller/blob/9be82cfc75aa4c04fff0e12dd4de853f9d83a101/demo_media/double%20click.gif" alt="Double Click" width="711" height="400"><br>
 <figcaption>Execute a double click.</figcaption>
</details>

<details>
<summary>Delete</summary>
<img src="https://github.com/xenon-19/Gesture_Controller/blob/9be82cfc75aa4c04fff0e12dd4de853f9d83a101/demo_media/double%20click.gif" alt="Double Click" width="711" height="400"><br>
 <figcaption>Delete files or text by performing a gun hand gesture.</figcaption>
</details>

<details>
<summary>Brightness Control</summary>
<img src="https://github.com/xenon-19/Gesture_Controller/blob/9be82cfc75aa4c04fff0e12dd4de853f9d83a101/demo_media/Brigntness%20Control.gif" alt="Brightness Control" width="711" height="400"><br>
 <figcaption>Adjust your screen's brightness level with a simple pinch hand movement. The rate of increase/decrease of brightness is proportional to the distance moved by pinch gesture from start point. </figcaption>
</details>

# Use Cases

This system can be used in a variety of scenarios:

- *Accessibility*: It can be a powerful tool for individuals with mobility impairments.
- *Presentations*: During a presentation, instead of using a clicker or a mouse to navigate through slides, just use your hand!
- *Education*: In an educational setting, teachers can use this system to interact with their digital materials in a more engaging way.
- *Gaming*: It can be used to create a more immersive gaming experience.

# Hand Detection, logic used
![handlandmark](https://github.com/Parthiba-Mukhopadhyay/hand_gesture_media_player/assets/89331202/80c7e10e-48ac-44c5-90ea-be40643f6cab)
<br>
The following reference points are considered for hand landmark detection.
<br>

# Requirements

To run the project, you will need the following:

- *Hardware*:
  - A computer with a camera (built-in or external) for capturing hand gestures.
  - Adequate processing power to run the computer vision algorithms smoothly.

- *Software*:
  - Operating system: Windows, macOS, or Linux.
  - Python (version 3.6 or higher) and numpy.
  - OpenCV library for computer vision tasks.
  - Mediapipe, pyautogui, sys and sbc to perform related tasks.

# Installation

1. Clone the project repository from GitHub:

   shell
   git clone https://github.com/Parthiba-Mukhopadhyay/hand_gesture_media_player
   

2. Install the required dependencies using pip:

   shell
   pip install -r requirements.txt
   

3. Connect a camera to your computer or ensure that the built-in camera is functional.
