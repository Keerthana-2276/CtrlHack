import cv2
import numpy as np
import time
import PoseModule as pm  # Importing PoseModule
import tkinter as tk
from tkinter import ttk
import os
import openai  # For Gemini Flash 2.0
import requests
from dotenv import load_dotenv

print("Script started")


# Google Cloud API Setup - REMOVE THIS if not using JSON!
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/google-cloud-key.json"  # DELETE this line if not using a JSON
# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
  # Replace with your actual API key!
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent" # replace with correct endpoint


# Check if OpenCV is working
cap = cv2.VideoCapture(0)  # Try different indices like 1, 2, 3 if needed
if not cap.isOpened():
    print("Error: Camera not detected!")
else:
    print("Camera opened successfully")

ret, frame = cap.read()
if ret:
    print("Frame captured successfully")
else:
    print("Error: Couldn't capture frame")

cap.release()
print("Script finished")

# Initialize video capture and pose detector
cap = cv2.VideoCapture(0)
detector = pm.PoseTracker(detectionCon=0.7, trackCon=0.7)  # Correct class name


def get_motivational_feedback(reps, exercise):
    """Generates a motivational message using Gemini Flash 2.0 API using the API endpoint directly."""
    url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    prompt = f"The user has completed {reps} reps of {exercise}. Generate a short, motivational message."
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        json_response = response.json()

        # Extract the motivational message. This part is crucial and depends on the exact structure of the API response.
        # Adapt it according to the Gemini API's response format for `generateContent`.
        # Example (check your API response with `print(json_response)` to confirm):
        if 'candidates' in json_response and json_response['candidates']:
            if 'content' in json_response['candidates'][0]:
                if 'parts' in json_response['candidates'][0]['content']:
                   motivational_message = json_response['candidates'][0]['content']['parts'][0]['text']
                else:
                    print("Error: 'parts' not found in content.")
                    return "Keep going!" # Default message
            else:
                print("Error: 'content' not found in candidate.")
                return "Keep going!" # Default message
        else:
            print("Error: 'candidates' not found in the response.")
            print("Full response from Gemini API:", json_response) # Print full response for debugging
            return "Keep going!" # Default message
        return motivational_message

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return "Keep going!"
    except ValueError as e:
        print(f"JSON decoding failed: {e}")
        return "Keep going!"  # Return a default message if parsing fails

# Google Text-to-Speech API setup (modified for REST API)
def text_to_speech(text):
    """Converts text to speech using Google Cloud Text-to-Speech API (REST API)."""
    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    headers = {'Content-Type': 'application/json'}
    data = {
        "input": {"text": text},
        "voice": {"languageCode": "en-US", "name": "en-US-Neural2-D"},  # Choose your preferred voice
        "audioConfig": {"audioEncoding": "MP3"}
    }

    try:
        response = requests.post(url, headers=headers, json=data, params={'key': GEMINI_API_KEY})
        response.raise_for_status()  # Raise HTTPError for bad responses
        json_response = response.json()

        # Decode the audio content (base64 encoded)
        audio_content = json_response['audioContent']
        import base64
        audio_bytes = base64.b64decode(audio_content)

        # Play the audio using a simple approach (requires simpleaudio)
        import simpleaudio as sa
        wave_obj = sa.WaveObject(audio_bytes, num_channels=1, bytes_per_sample=2, sample_rate=24000)  # Adjust parameters as needed
        play_obj = wave_obj.play()
        play_obj.wait_done()

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except KeyError as e:
        print(f"KeyError: {e}. Check the API response structure.")
        print("Full response from Text-to-Speech API:", json_response)  # Print for debugging
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

class ExerciseTracker:
    def __init__(self):  # Fixed constructor
        self.prev_angles = {}

    def calculate_angle(self, lmlist, p1, p2, p3):
        """Calculates the angle between three points using the dot product formula."""
        try:
            x1, y1 = lmlist[p1][1], lmlist[p1][2]
            x2, y2 = lmlist[p2][1], lmlist[p2][2]
            x3, y3 = lmlist[p3][1], lmlist[p3][2]

            v1 = np.array([x1 - x2, y1 - y2])
            v2 = np.array([x3 - x2, y3 - y2])

            dot_product = np.dot(v1, v2)
            magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
            cosine_angle = dot_product / magnitude
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

            return angle
        except:
            return None  # Return None if calculation fails

def push_up_logic(img, lmlist, count, dir, tracker):
    if len(lmlist) < 27:
        return count, dir

    left_elbow = tracker.calculate_angle(lmlist, 11, 13, 15)
    right_elbow = tracker.calculate_angle(lmlist, 12, 14, 16)
    left_hip = tracker.calculate_angle(lmlist, 23, 25, 27)
    right_hip = tracker.calculate_angle(lmlist, 24, 26, 28)

    if None in [left_elbow, right_elbow, left_hip, right_hip]:
        return count, dir

    left_per = np.interp(left_elbow, (85, 160), (0, 100))
    right_per = np.interp(right_elbow, (85, 160), (0, 100))
    body_aligned = left_hip > 160 and right_hip > 160
    avg_per = (left_per + right_per) / 2

    if avg_per > 95 and body_aligned and dir == 0:
        count += 0.5
        dir = 1
    if avg_per < 5 and body_aligned and dir == 1:
        count += 0.5
        dir = 0

    return count, dir

def squat_logic(img, lmlist, count, dir, tracker):
    if len(lmlist) < 27:
        return count, dir

    left_knee = tracker.calculate_angle(lmlist, 23, 25, 27)
    right_knee = tracker.calculate_angle(lmlist, 24, 26, 28)

    if None in [left_knee, right_knee]:
        return count, dir

    avg_knee = (left_knee + right_knee) / 2

    if avg_knee < 100 and dir == 0:
        count += 0.5
        dir = 1
    if avg_knee > 160 and dir == 1:
        count += 0.5
        dir = 0

    return count, dir

def shoulder_press_logic(img, lmlist, count, dir, tracker):
    if len(lmlist) < 17:
        return count, dir

    left_elbow = tracker.calculate_angle(lmlist, 11, 13, 15)
    right_elbow = tracker.calculate_angle(lmlist, 12, 14, 16)

    if None in [left_elbow, right_elbow]:
        return count, dir

    avg_elbow = (left_elbow + right_elbow) / 2

    if avg_elbow < 100 and dir == 0:
        count += 0.5
        dir = 1
    if avg_elbow > 160 and dir == 1:
        count += 0.5
        dir = 0

    return count, dir

def draw_skeleton(img, lmlist):
    """Draws skeleton and angle annotations on the frame."""
    if len(lmlist) > 16:
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:  # Major joints
            cv2.circle(img, (lmlist[idx][1], lmlist[idx][2]), 5, (0, 255, 0), cv2.FILLED)

        # Draw connections (e.g., arm, leg, torso lines)
        connections = [(11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)]
        for p1, p2 in connections:
            cv2.line(img, (lmlist[p1][1], lmlist[p1][2]), (lmlist[p2][1], lmlist[p2][2]), (255, 0, 0), 3)



def get_exercise_choice():
    """Creates a stylish dropdown GUI for exercise selection and returns the choice."""
    def on_submit():
        """Callback function to set exercise choice and close window."""
        nonlocal exercise_choice
        exercise_choice = exercise_var.get()
        root.destroy()  # Close the GUI window

    def on_close():
        """Callback function for window close button (X)."""
        nonlocal exercise_choice
        exercise_choice = None  # Set default if closed without selection
        root.destroy()  # Close the window

    # Initialize window
    exercise_choice = None
    root = tk.Tk()
    root.title("Select Exercise")
    root.geometry("350x250")  # Set window size
    root.configure(bg="#2C2F33")  # Dark background color

    # Handle window close button (X)
    root.protocol("WM_DELETE_WINDOW", on_close)

    # Style Configuration
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#2C2F33", foreground="white", font=("Arial", 12))
    style.configure("TButton", background="#7289DA", foreground="white", font=("Arial", 12, "bold"), padding=5)
    style.map("TButton", background=[("active", "#677BC4")])  # Button hover effect

    # Title Label
    ttk.Label(root, text="Choose an Exercise:", anchor="center", font=("Arial", 14, "bold")).pack(pady=15)

    # Dropdown Menu
    exercise_var = tk.StringVar()
    exercise_dropdown = ttk.Combobox(root, textvariable=exercise_var, state="readonly", font=("Arial", 12))
    exercise_dropdown['values'] = ("Push-up", "Squat", "Shoulder Press")
    exercise_dropdown.pack(pady=10)
    exercise_dropdown.current(0)  # Default to first option

    # Start Button
    ttk.Button(root, text="Start", command=on_submit).pack(pady=20)

    root.mainloop()
    return exercise_choice


def exercise_logic():
    count, dir = 0, 0
    start_time = time.time()
    tracker = ExerciseTracker()

    exercise = get_exercise_choice()

    while True:
        success, img = cap.read()
        if not success:
            continue

        img = cv2.resize(img, (1288, 720))
        img = detector.findPose(img, False)
        lmlist = detector.findPosition(img, False)

        if len(lmlist) != 0:
            if exercise == "Push-up":  # Corrected exercise name
                count, dir = push_up_logic(img, lmlist, count, dir, tracker)
            elif exercise == "Squat":  # Corrected exercise name
                count, dir = squat_logic(img, lmlist, count, dir, tracker)
            elif exercise == "Shoulder Press":  # Corrected exercise name
                count, dir = shoulder_press_logic(img, lmlist, count, dir, tracker)

            draw_skeleton(img, lmlist)  # Draw keypoints and connections

            cv2.putText(img, f'Reps: {int(count)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.putText(img, f'Time: {int(time.time() - start_time)}s', (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            if int(count) % 5 == 0 and int(count) > 0:  # Give motivation every 5 reps
                message = get_motivational_feedback(int(count), exercise)
                text_to_speech(message)

        cv2.imshow("Exercise Tracker", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":  # Fixed typo here
    exercise_logic()