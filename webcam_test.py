
import cv2
import mediapipe as mp
import math
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection

smoothed_lean = None
alpha = 0.15

CALIBRATION_SECONDS = 5
POSTURE_TOLERANCE = 10
TURN_THRESHOLD = 0.08
FACE_LOST_SECONDS = 1.5

baseline_samples = []
baseline = None
calibration_start = time.time()
face_last_seen = None

window_name = "Posture Robot"


with mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose, mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
) as face_detector:

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = pose.process(rgb)
        face_results = face_detector.process(rgb)

        now = time.time()
        face_detected = bool(face_results.detections)

        if face_detected:
            face_last_seen = now
            face_status = "FACE DETECTED"
            face_color = (0, 255, 0)

            for detection in face_results.detections:
                mp_draw.draw_detection(frame, detection)
        else:
            face_status = "NO FACE DETECTED"
            face_color = (0, 0, 255)

        facing_direction = "UNKNOWN"

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            visible = min(
                landmarks[i].visibility
                for i in [11, 12, 23, 24]
            )

            if pose_results.pose_world_landmarks and visible > 0.5:
                world = pose_results.pose_world_landmarks.landmark

                shoulder_depth = world[11].z - world[12].z
                hip_depth = world[23].z - world[24].z
                orientation_score = (
                    shoulder_depth + hip_depth
                ) / 2

                if abs(orientation_score) >= TURN_THRESHOLD:
                    if orientation_score > 0:
                        facing_direction = "LEFT"
                    else:
                        facing_direction = "RIGHT"
                elif face_detected:
                    facing_direction = "FORWARD"
                elif (
                    face_last_seen is not None
                    and now - face_last_seen >= FACE_LOST_SECONDS
                ):
                    facing_direction = "BACKWARD"
                else:
                    facing_direction = "UNKNOWN"

            if visible > 0.5:
                shoulder_x = (
                    landmarks[11].x + landmarks[12].x
                ) / 2
                shoulder_y = (
                    landmarks[11].y + landmarks[12].y
                ) / 2
                hip_x = (
                    landmarks[23].x + landmarks[24].x
                ) / 2
                hip_y = (
                    landmarks[23].y + landmarks[24].y
                ) / 2

                dx = shoulder_x - hip_x
                dy = hip_y - shoulder_y
                current_lean = math.degrees(
                    math.atan2(dx, dy)
                )

                if smoothed_lean is None:
                    smoothed_lean = current_lean
                else:
                    smoothed_lean = (
                        alpha * current_lean
                        + (1 - alpha) * smoothed_lean
                    )

                if baseline is None:
                    baseline_samples.append(smoothed_lean)

            mp_draw.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        if baseline is None:
            elapsed = time.time() - calibration_start
            remaining = CALIBRATION_SECONDS - elapsed

            cv2.putText(
                frame,
                "SIT IN YOUR USUAL COMFORTABLE POSTURE",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            if remaining > 0:
                cv2.putText(
                    frame,
                    f"Calibrating: {remaining:.1f}s",
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
            elif baseline_samples:
                baseline = sum(baseline_samples) / len(
                    baseline_samples
                )
            else:
                calibration_start = time.time()

        else:
            cv2.putText(
                frame,
                f"Baseline: {baseline:+.1f} deg",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if smoothed_lean is not None:
                deviation = smoothed_lean - baseline

                if abs(deviation) < POSTURE_TOLERANCE:
                    posture_status = "GOOD POSTURE"
                    status_color = (0, 255, 0)
                elif deviation > 0:
                    posture_status = "LEANING LEFT"
                    status_color = (0, 0, 255)
                else:
                    posture_status = "LEANING RIGHT"
                    status_color = (0, 0, 255)

                cv2.putText(
                    frame,
                    f"Current: {smoothed_lean:+.1f} deg",
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"Change: {deviation:+.1f} deg",
                    (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    posture_status,
                    (15, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2
                )

        cv2.putText(
            frame,
            face_status,
            (15, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            face_color,
            2
        )
        cv2.putText(
            frame,
            f"Facing: {facing_direction}",
            (15, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if cv2.getWindowProperty(
            window_name, cv2.WND_PROP_VISIBLE
        ) < 1:
            break

cap.release()
cv2.destroyAllWindows()
