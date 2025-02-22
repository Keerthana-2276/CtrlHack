import cv2
import numpy as np
import time
import PoseModule as pm

# Initialize video capture and pose detector
cap = cv2.VideoCapture(0)
detector = pm.poseDetector(detectionCon=0.7, trackCon=0.7)

class ExerciseTracker:
    def _init_(self):  # Fixed constructor
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

def exercise_logic():
    count, dir = 0, 0
    start_time = time.time()
    tracker = ExerciseTracker()
    
    exercise = input("Choose exercise: pushup, squat, shoulder_press: ").strip().lower()
    
    while True:
        success, img = cap.read()
        if not success:
            continue
        
        img = cv2.resize(img, (1288, 720))
        img = detector.findPose(img, False)
        lmlist = detector.findPosition(img, False)
        
        if len(lmlist) != 0:
            if exercise == "pushup":
                count, dir = push_up_logic(img, lmlist, count, dir, tracker)
            elif exercise == "squat":
                count, dir = squat_logic(img, lmlist, count, dir, tracker)
            elif exercise == "shoulder_press":
                count, dir = shoulder_press_logic(img, lmlist, count, dir, tracker)

            draw_skeleton(img, lmlist)  # Draw keypoints and connections
            
            cv2.putText(img, f'Reps: {int(count)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.putText(img, f'Time: {int(time.time() - start_time)}s', (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        
        cv2.imshow("Exercise Tracker", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":  # Fixed typo here
    exercise_logic()