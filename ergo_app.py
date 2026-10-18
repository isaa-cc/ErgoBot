import cv2
import math
import time
import threading
import copy
import serial

import mediapipe as mp

from flask import Flask, Response, jsonify, render_template, request


app = Flask(__name__)

# ============================================================
# Feather RP2040 USB serial connection
# ============================================================

SERIAL_PORT = "COM4"
SERIAL_BAUD = 115200

servo_serial = None
servo_serial_lock = threading.Lock()
last_connect_attempt = 0.0


def connect_to_feather():
    """Open the Feather serial connection and allow startup time."""
    global servo_serial
    global last_connect_attempt

    with servo_serial_lock:
        if servo_serial is not None and servo_serial.is_open:
            return True

        # Don't hammer the port if it keeps failing.
        now = time.time()
        if now - last_connect_attempt < 3.0:
            return False
        last_connect_attempt = now

        try:
            servo_serial = serial.Serial(
                SERIAL_PORT,
                SERIAL_BAUD,
                timeout=1,
                write_timeout=1
            )

            print(f"Connected to Feather on {SERIAL_PORT}", flush=True)

            # Opening USB serial can reset the RP2040.
            time.sleep(2)
            servo_serial.reset_input_buffer()

            print("Feather startup wait complete.", flush=True)
            return True

        except (serial.SerialException, OSError) as error:
            servo_serial = None
            print(f"Feather connection failed: {error}", flush=True)
            return False


# ============================================================
# MediaPipe
# ============================================================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection


# ============================================================
# Configuration
# ============================================================

CAMERA_INDEX = 0
USE_WINDOWS_CAMERA_BACKEND = True

ALPHA = 0.15
LANDMARK_ALPHA = 0.20  # Lower values reduce jitter but add tracking lag.
ROLLING_SCORE_ALPHA = 0.05

CALIBRATION_SECONDS = 5
POSTURE_TOLERANCE = 10.0
FACE_LOST_SECONDS = 1.5
TURN_THRESHOLD = 0.08

HEAD_THRESHOLD = 15.0
SPINE_THRESHOLD = 15.0
SHOULDER_THRESHOLD = 6.0
PELVIS_THRESHOLD = 6.0

HEAD_PENALTY_RATE = 1.5
SPINE_PENALTY_RATE = 2.0
SHOULDER_PENALTY_RATE = 3.0
PELVIS_PENALTY_RATE = 3.0

POOR_POSTURE_SCORE_THRESHOLD = 70.0
FATIGUE_START_SECONDS = 60.0
FATIGUE_RATE = 0.1

# Pose model. 1 gives noticeably better depth (needed for forward/back lean)
# than 0. Drop back to 0 if the video gets laggy.
MODEL_COMPLEXITY = 1

# Every posture metric is measured RELATIVE TO THE CALIBRATED BASELINE, so the
# posture you hold during calibration is by definition "good" (deviation 0).
#   head        signed ear-vs-shoulder offset angle        (deg)
#   spine       signed trunk lean in the image plane       (deg)
#   shoulder    shoulder-line tilt                         (deg)
#   pelvis      hip-line tilt                              (deg)
#   head_pitch  nose height below the ears, % of shoulder width (up = head down)
#   lean        forward(+) / back(-) trunk lean from depth (deg)
#   torso       shoulder-to-hip height / shoulder width    (change in %)
#   neck        ear-to-shoulder height / shoulder width    (change in %)
METRIC_KEYS = (
    "head", "spine", "shoulder", "pelvis",
    "head_pitch", "lean", "torso", "neck"
)
RATIO_METRICS = ("torso", "neck")

# If leaning toward the camera shows up as a NEGATIVE lean, flip this to -1.
LEAN_SIGN = 1

# Posture-type thresholds (deviation from baseline). Tune these first.
LEAN_FORWARD_THRESHOLD = 10.0     # deg   -> LEANING FORWARD
LEAN_BACK_THRESHOLD = 12.0        # deg   -> LEANING BACK
HEAD_DOWN_THRESHOLD = 12.0        # % shoulder width -> HEAD DOWN / TILTED BACK
TORSO_COMPRESS_THRESHOLD = 12.0   # %     -> SUNKEN BODY / part of SLOUCHING
TORSO_STRETCH_THRESHOLD = 12.0    # %     -> part of HYPEREXTENSION
NECK_SHRUG_THRESHOLD = 15.0       # %     -> SHOULDERS RAISED
LABEL_HOLD_SECONDS = 1.0          # a new label must persist this long

# Adaptive noise guard: while you hold still during calibration, the
# jitter of every metric is measured. A metric must then move by at least
# NOISE_MULTIPLIER x its own jitter before it counts as a posture change,
# no matter how low the fixed thresholds above are.
NOISE_MULTIPLIER = 3.0

# metric: (threshold, penalty per degree/percent beyond it)
PENALTY_CONFIG = {
    "head": (HEAD_THRESHOLD, HEAD_PENALTY_RATE),
    "spine": (SPINE_THRESHOLD, SPINE_PENALTY_RATE),
    "shoulder": (SHOULDER_THRESHOLD, SHOULDER_PENALTY_RATE),
    "pelvis": (PELVIS_THRESHOLD, PELVIS_PENALTY_RATE),
    "head_pitch": (HEAD_DOWN_THRESHOLD, 1.0),
    "lean": (LEAN_FORWARD_THRESHOLD, 1.5),
    "torso": (TORSO_COMPRESS_THRESHOLD, 1.0),
    "neck": (NECK_SHRUG_THRESHOLD, 0.8),
}


# ============================================================
# Shared tracking state
# ============================================================

smoothed_pose_points = None  # EMA-filtered image-space pose landmarks.

smoothed_metrics = {key: None for key in METRIC_KEYS}

current_metrics = {key: None for key in METRIC_KEYS}

baseline_metrics = None
metric_deviations = {key: None for key in METRIC_KEYS}

# Debounce state so the posture label doesn't flicker frame to frame.
label_state = {"stable": "CALIBRATING", "pending": None, "since": 0.0}

# Per-metric jitter measured during calibration (same units as deviations).
baseline_noise = {}

calibration_samples = []
calibration_start = time.time()
calibration_valid_since = None

face_last_seen = None
last_update = time.time()
last_valid_pose_time = None  # Used so servos hold still if the pose is lost.

posture_status = "CALIBRATING"
face_status = "NO FACE DETECTED"
facing_direction = "UNKNOWN"

instantaneous_score = None
rolling_score = None
poor_posture_seconds = 0.0
last_posture_time = None

tracking_active = False

state_lock = threading.Lock()


# ============================================================
# Automatic servo control (mimic the user's posture)
# ============================================================

SERVO_CONTROL_ENABLED = True
SERVO_UPDATE_INTERVAL = 0.1   # how often the servo thread checks for work
SERVO_DEADBAND = 1.5          # degrees of user motion to ignore
SERVO_MAX_RANGE = 40          # max degrees away from home for any servo
SERVO_SLEW = 6                # max servo change per update (smooth motion)

# The firmware moves 1 degree per 15 ms and handles commands one at a time.
# Python tracks how long the Feather is busy and never sends more than it can
# execute, so commands can't pile up and make the robot lag behind you.
FIRMWARE_SECONDS_PER_DEGREE = 0.015
FIRMWARE_COMMAND_OVERHEAD = 0.04
feather_busy_until = 0.0

# Must match HOME_ANGLES in the Feather firmware: [80, 90, 60]
SERVO_HOME = {
    0: 80,  # Head / neck
    1: 90,  # Chest
    2: 60   # Torso / pelvis
}

# servo index: (metric, gain, sign)
# Signs were flipped because the robot leaned BACK when the user leaned forward.
# If a servo moves the wrong direction, flip its sign (+1 <-> -1).
#
# Alternative if you sit facing the camera (uses depth instead of image-plane tilt):
#   0: ("head_pitch", 1.5, -1),
#   1: ("lean",       1.5, +1),
#   2: ("lean",       1.5, +1),
SERVO_MAP = {
    0: ("head",  1.5, +1),
    1: ("spine", 1.5, -1),
    2: ("spine", 1.5, -1),
}

# After a "go home", mimicking pauses briefly so the slow home move can finish.
SERVO_PAUSE_SECONDS = 3.0
servo_pause_until = 0.0

# While posture is GOOD the home pose is re-sent this often, so the robot
# heals itself even if a command was dropped or the angles drifted.
HOME_RESYNC_SECONDS = 2.0
last_home_sync = 0.0

last_servo_angles = SERVO_HOME.copy()
servo_stop_event = threading.Event()
servo_thread = None


def send_serial_line(line):
    """Send one raw text line to the Feather."""
    global servo_serial

    if not connect_to_feather():
        return False

    try:
        with servo_serial_lock:
            if servo_serial is None or not servo_serial.is_open:
                return False

            servo_serial.write((line + "\n").encode())
            servo_serial.flush()

        return True

    except (serial.SerialException, OSError) as error:
        print(f"Servo command failed: {error}", flush=True)

        # Drop the dead port so the next command reconnects.
        with servo_serial_lock:
            try:
                if servo_serial is not None:
                    servo_serial.close()
            except Exception:
                pass
            servo_serial = None

        return False


def send_servo_command(servo_index, angle):
    """Send one bounded servo command to the Feather."""
    if servo_index not in SERVO_HOME:
        return False

    angle = max(0, min(180, int(angle)))
    return send_serial_line(f"{servo_index} {angle}")


def send_home_commands():
    """Send an explicit absolute angle to every servo, then a firmware home.

    The explicit per-servo commands use the exact same path as normal
    mimicking, so if the robot follows you it will also follow these.
    """
    global last_home_sync

    ok = True
    for servo_index, angle in SERVO_HOME.items():
        ok = send_servo_command(servo_index, angle) and ok

    ok = send_serial_line("home") and ok

    last_home_sync = time.time()
    if ok:
        last_servo_angles.update(SERVO_HOME)

    return ok


def robot_go_home(reason="requested"):
    """Send the robot home and pause mimicking while it moves."""
    global servo_pause_until
    global feather_busy_until

    ok = send_home_commands()

    until = time.time() + SERVO_PAUSE_SECONDS
    servo_pause_until = until
    feather_busy_until = until

    print(
        f"Robot home ({reason}): {'sent' if ok else 'FAILED - check COM port'}",
        flush=True
    )
    return ok


def servo_step():
    """One control cycle: compute targets and send what the Feather can run."""
    global feather_busy_until

    if not SERVO_CONTROL_ENABLED:
        return

    now = time.time()
    if now < servo_pause_until or now < feather_busy_until:
        return

    with state_lock:
        active = tracking_active
        metrics = dict(current_metrics)
        baseline = dict(baseline_metrics) if baseline_metrics else None
        noise = dict(baseline_noise)
        status_now = posture_status
        pose_fresh = (
            last_valid_pose_time is not None
            and time.time() - last_valid_pose_time < 1.0
        )

    # Hold still until calibrated and while the pose is valid.
    if not active or baseline is None or not pose_fresh:
        return

    # Good posture = robot upright. Go home, then keep re-confirming it.
    if status_now == "GOOD POSTURE":
        needs_home = any(
            last_servo_angles[i] != SERVO_HOME[i] for i in SERVO_HOME
        )

        if needs_home:
            robot_go_home("good posture")
        elif now - last_home_sync > HOME_RESYNC_SECONDS:
            send_home_commands()
            feather_busy_until = time.time() + 0.15

        return

    busy_seconds = 0.0

    for servo_index, (key, gain, sign) in SERVO_MAP.items():
        if time.time() < servo_pause_until:
            break

        home = SERVO_HOME[servo_index]
        m = metrics.get(key)
        b = baseline.get(key)

        if m is None or b is None:
            # Metric can't be measured right now (e.g. ears hidden):
            # relax toward home instead of freezing in a bent pose.
            desired = home
        else:
            deviation = m - b

            # Soft deadband: ignore jitter, and shrink what remains so
            # returning to your baseline brings the robot back to home.
            deadband = max(SERVO_DEADBAND, 1.5 * noise.get(key, 0.0))
            if abs(deviation) <= deadband:
                deviation = 0.0
            else:
                deviation -= math.copysign(deadband, deviation)

            desired = home + sign * gain * deviation

        desired = max(
            home - SERVO_MAX_RANGE,
            min(home + SERVO_MAX_RANGE, desired)
        )

        # Slew-limit so the robot glides instead of jumping.
        last = last_servo_angles[servo_index]
        if abs(desired - last) < 1.0:
            target = int(round(desired))
        else:
            step = max(-SERVO_SLEW, min(SERVO_SLEW, desired - last))
            target = int(round(last + step))

        if target != last and send_servo_command(servo_index, target):
            busy_seconds += (
                abs(target - last) * FIRMWARE_SECONDS_PER_DEGREE
                + FIRMWARE_COMMAND_OVERHEAD
            )
            last_servo_angles[servo_index] = target

    if busy_seconds > 0:
        feather_busy_until = time.time() + busy_seconds


def servo_worker():
    """Runs in its own thread so serial I/O never stalls the camera loop."""
    print("SERVO WORKER STARTED", flush=True)

    while not servo_stop_event.is_set():
        time.sleep(SERVO_UPDATE_INTERVAL)

        try:
            servo_step()
        except Exception as error:
            # Never let the servo thread die silently.
            print(f"SERVO WORKER ERROR: {error!r}", flush=True)
            time.sleep(0.5)

    print("SERVO WORKER STOPPED", flush=True)


def start_servo_worker():
    global servo_thread

    if servo_thread is not None and servo_thread.is_alive():
        return

    servo_stop_event.clear()
    servo_thread = threading.Thread(target=servo_worker, daemon=True)
    servo_thread.start()


# ============================================================
# Camera and frame state
# ============================================================

camera = None
camera_thread = None

stop_event = threading.Event()

latest_frame = None
frame_lock = threading.Lock()


# ============================================================
# Utility functions
# ============================================================

def empty_metrics():
    return {key: None for key in METRIC_KEYS}


def open_camera():
    if USE_WINDOWS_CAMERA_BACKEND and hasattr(cv2, "CAP_DSHOW"):
        print(
            f"Trying camera {CAMERA_INDEX} with DirectShow...",
            flush=True
        )

        cam = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        if cam.isOpened():
            return cam

        cam.release()

    print(
        f"Trying camera {CAMERA_INDEX} with default backend...",
        flush=True
    )

    cam = cv2.VideoCapture(CAMERA_INDEX)

    if cam.isOpened():
        return cam

    cam.release()
    return None


def midpoint(first, second):
    return (
        (first.x + second.x) / 2.0,
        (first.y + second.y) / 2.0
    )


def signed_angle_from_vertical(dx, dy):
    return math.degrees(math.atan2(dx, dy))


def put_text(frame, text, y, color=(255, 255, 255), scale=0.55):
    cv2.putText(
        frame,
        text,
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2
    )


# ============================================================
# Landmark smoothing
# ============================================================

def smooth_pose_landmarks(pose_landmarks):
    """Return a copied pose with temporally smoothed x/y/z coordinates.

    Visibility is kept raw so confidence checks still reflect the current frame.
    The same filtered landmarks are used for drawing and angle calculations.
    """
    global smoothed_pose_points

    if pose_landmarks is None:
        smoothed_pose_points = None
        return None

    filtered = copy.deepcopy(pose_landmarks)
    raw_points = pose_landmarks.landmark

    if smoothed_pose_points is None or len(smoothed_pose_points) != len(raw_points):
        smoothed_pose_points = [
            (point.x, point.y, point.z) for point in raw_points
        ]
    else:
        updated_points = []
        for point, previous in zip(raw_points, smoothed_pose_points):
            alpha = LANDMARK_ALPHA if point.visibility >= 0.4 else LANDMARK_ALPHA * 0.5
            updated_points.append((
                alpha * point.x + (1.0 - alpha) * previous[0],
                alpha * point.y + (1.0 - alpha) * previous[1],
                alpha * point.z + (1.0 - alpha) * previous[2],
            ))
        smoothed_pose_points = updated_points

    for point, (x, y, z) in zip(filtered.landmark, smoothed_pose_points):
        point.x = x
        point.y = y
        point.z = z

    return filtered


# ============================================================
# Posture measurements
# ============================================================

def calculate_metrics(landmarks, world_landmarks=None):
    left_ear = landmarks[7]
    right_ear = landmarks[8]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    ear_mid_x, ear_mid_y = midpoint(left_ear, right_ear)
    shoulder_mid_x, shoulder_mid_y = midpoint(
        left_shoulder, right_shoulder
    )
    hip_mid_x, hip_mid_y = midpoint(left_hip, right_hip)

    # Head angle (SIGNED, so the robot knows which way the head moved).
    ear_visibility = min(
        left_ear.visibility,
        right_ear.visibility
    )

    if ear_visibility >= 0.3:
        head_dx = ear_mid_x - shoulder_mid_x
        head_dy = abs(shoulder_mid_y - ear_mid_y)
        head_angle = math.degrees(
            math.atan2(head_dx, max(head_dy, 0.001))
        )
    else:
        head_angle = None

    # Trunk angle relative to vertical.
    spine_dx = shoulder_mid_x - hip_mid_x
    spine_dy = hip_mid_y - shoulder_mid_y
    spine_angle = signed_angle_from_vertical(
        spine_dx, spine_dy
    )

    # Shoulder tilt.
    shoulder_dx = right_shoulder.x - left_shoulder.x
    shoulder_dy = right_shoulder.y - left_shoulder.y
    shoulder_angle = math.degrees(
        math.atan2(
            shoulder_dy,
            max(abs(shoulder_dx), 0.001)
        )
    )

    # Pelvis tilt.
    pelvis_dx = right_hip.x - left_hip.x
    pelvis_dy = right_hip.y - left_hip.y
    pelvis_angle = math.degrees(
        math.atan2(
            pelvis_dy,
            max(abs(pelvis_dx), 0.001)
        )
    )

    # Scale reference: shoulder width keeps ratios independent of how far
    # the user sits from the camera.
    shoulder_width = max(math.hypot(shoulder_dx, shoulder_dy), 0.001)

    # Torso height (shoulder -> hip) relative to shoulder width.
    # Shrinks when sunken / collapsed, grows when sitting tall or arching.
    torso_ratio = (hip_mid_y - shoulder_mid_y) / shoulder_width

    # Neck height (ear -> shoulder) relative to shoulder width.
    # Shrinks when the shoulders are raised.
    if ear_visibility >= 0.3:
        neck_ratio = (shoulder_mid_y - ear_mid_y) / shoulder_width
    else:
        neck_ratio = None

    # Head pitch: how far the nose sits below the ear line.
    # Increases when the head tips down, decreases when it tips back.
    nose = landmarks[0]
    if ear_visibility >= 0.3 and nose.visibility >= 0.4:
        head_pitch = 100.0 * (nose.y - ear_mid_y) / shoulder_width
    else:
        head_pitch = None

    # Forward/back trunk lean from the depth (world) landmarks.
    lean = None
    if world_landmarks is not None:
        ws_l, ws_r = world_landmarks[11], world_landmarks[12]
        wh_l, wh_r = world_landmarks[23], world_landmarks[24]
        shoulder_z = (ws_l.z + ws_r.z) / 2.0
        hip_z = (wh_l.z + wh_r.z) / 2.0
        shoulder_y = (ws_l.y + ws_r.y) / 2.0
        hip_y = (wh_l.y + wh_r.y) / 2.0
        torso_height = max(hip_y - shoulder_y, 0.05)
        # Shoulders closer to the camera than the hips (smaller z) = forward.
        lean = LEAN_SIGN * math.degrees(
            math.atan2(hip_z - shoulder_z, torso_height)
        )

    return {
        "head": head_angle,
        "spine": spine_angle,
        "shoulder": shoulder_angle,
        "pelvis": pelvis_angle,
        "head_pitch": head_pitch,
        "lean": lean,
        "torso": torso_ratio,
        "neck": neck_ratio
    }


def smooth_metrics(metrics):
    global smoothed_metrics

    for name, value in metrics.items():
        if value is None:
            continue

        previous = smoothed_metrics[name]

        if previous is None:
            smoothed_metrics[name] = value
        else:
            smoothed_metrics[name] = (
                ALPHA * value
                + (1.0 - ALPHA) * previous
            )

    return smoothed_metrics.copy()


# ============================================================
# Calibration
# ============================================================

def reset_calibration():
    global smoothed_metrics
    global smoothed_pose_points
    global current_metrics
    global baseline_metrics
    global metric_deviations
    global calibration_samples
    global calibration_start
    global calibration_valid_since
    global posture_status
    global instantaneous_score
    global rolling_score
    global poor_posture_seconds
    global last_posture_time

    with state_lock:
        smoothed_metrics = empty_metrics()
        smoothed_pose_points = None
        current_metrics = empty_metrics()
        baseline_metrics = None
        metric_deviations = empty_metrics()
        calibration_samples = []
        calibration_start = time.time()
        calibration_valid_since = None
        posture_status = "CALIBRATING"
        instantaneous_score = None
        rolling_score = None
        poor_posture_seconds = 0.0
        last_posture_time = None
        label_state.update(stable="CALIBRATING", pending=None, since=0.0)
        baseline_noise.clear()


def establish_baseline():
    global baseline_metrics

    if not calibration_samples:
        return False

    baseline_metrics = {}
    baseline_noise.clear()

    for metric in METRIC_KEYS:
        valid_values = [
            sample[metric]
            for sample in calibration_samples
            if sample[metric] is not None
        ]

        if valid_values:
            mean = sum(valid_values) / len(valid_values)
            baseline_metrics[metric] = mean

            variance = sum(
                (value - mean) ** 2 for value in valid_values
            ) / len(valid_values)
            noise = math.sqrt(variance)

            # Ratio metrics are reported as % change, so convert their noise.
            if metric in RATIO_METRICS and abs(mean) > 1e-6:
                noise = noise / abs(mean) * 100.0

            baseline_noise[metric] = noise
        else:
            # No valid samples: this metric is simply ignored.
            baseline_metrics[metric] = None

    print(
        "Baseline noise:",
        {k: round(v, 2) for k, v in baseline_noise.items()},
        flush=True
    )
    return True


# ============================================================
# Scoring
# ============================================================

def penalty_for_deviation(deviation, threshold, penalty_rate):
    if deviation is None:
        return 0.0

    amount = abs(float(deviation))

    if amount <= threshold:
        return 0.0

    degrees_beyond_threshold = amount - threshold

    return 1.0 + (penalty_rate * degrees_beyond_threshold)


def compute_deviations(current, baseline):
    """Deviation of every metric from the calibrated (good) baseline.

    Angles and percentages are plain differences; ratio metrics are
    reported as percent change from the baseline ratio.
    """
    deviations = {}

    for key in METRIC_KEYS:
        cur = current.get(key)
        base = baseline.get(key)

        if cur is None or base is None:
            deviations[key] = None
        elif key in RATIO_METRICS:
            if abs(base) > 1e-6:
                deviations[key] = ((cur / base) - 1.0) * 100.0
            else:
                deviations[key] = None
        else:
            deviations[key] = cur - base

    return deviations


def threshold_for(key, default):
    """Fixed threshold, raised to the metric's measured jitter if larger."""
    return max(default, NOISE_MULTIPLIER * baseline_noise.get(key, 0.0))


def calculate_instantaneous_score(deviations):
    """Starts at 100 and deducts piecewise linear penalties per metric."""
    total_penalty = 0.0

    for key, (threshold, rate) in PENALTY_CONFIG.items():
        total_penalty += penalty_for_deviation(
            deviations.get(key),
            threshold_for(key, threshold),
            rate
        )

    return max(0.0, 100.0 - total_penalty)


def update_fatigue(posture_score, now):
    global poor_posture_seconds
    global last_posture_time

    if last_posture_time is None:
        last_posture_time = now
        return poor_posture_seconds

    elapsed = max(0.0, now - last_posture_time)
    last_posture_time = now

    if posture_score < POOR_POSTURE_SCORE_THRESHOLD:
        poor_posture_seconds += elapsed
    else:
        poor_posture_seconds = 0.0

    return poor_posture_seconds


def apply_fatigue_multiplier(score, poor_seconds):
    """FatigueMultiplier = 1.0 + 0.1 * (seconds_in_poor_posture / 60)"""
    if score >= POOR_POSTURE_SCORE_THRESHOLD:
        return score

    if poor_seconds <= FATIGUE_START_SECONDS:
        return score

    multiplier = 1.0 + (
        FATIGUE_RATE * (poor_seconds / FATIGUE_START_SECONDS)
    )

    return max(0.0, score / multiplier)


def update_rolling_score(score):
    """Exponential moving average: S(t) = a*S_instant + (1-a)*S(t-1)"""
    global rolling_score

    if score is None:
        return rolling_score

    if rolling_score is None:
        rolling_score = score
    else:
        rolling_score = (
            ROLLING_SCORE_ALPHA * score
            + (1.0 - ROLLING_SCORE_ALPHA) * rolling_score
        )

    return rolling_score


def posture_label(deviations, score):
    """Classify WHAT kind of posture problem this is.

    Everything is relative to the calibrated baseline, so holding the
    calibration posture always reads GOOD POSTURE.
    """
    def val(key):
        value = deviations.get(key)
        return 0.0 if value is None else value

    head = abs(val("head"))
    spine = abs(val("spine"))
    shoulder = abs(val("shoulder"))
    pelvis = abs(val("pelvis"))
    pitch = val("head_pitch")
    lean = val("lean")
    torso = val("torso")
    neck = val("neck")

    leaning_forward = lean > threshold_for("lean", LEAN_FORWARD_THRESHOLD)
    leaning_back = lean < -threshold_for("lean", LEAN_BACK_THRESHOLD)
    head_down = pitch > threshold_for("head_pitch", HEAD_DOWN_THRESHOLD)
    head_back = pitch < -threshold_for("head_pitch", HEAD_DOWN_THRESHOLD)
    compressed = torso < -threshold_for("torso", TORSO_COMPRESS_THRESHOLD)
    stretched = torso > threshold_for("torso", TORSO_STRETCH_THRESHOLD)
    shrugged = neck < -threshold_for("neck", NECK_SHRUG_THRESHOLD)

    # Slouch = rounding forward: at least two of lean-forward,
    # head-down and a shortened torso happening together.
    if sum([leaning_forward, head_down, compressed]) >= 2:
        return "SLOUCHING"

    # Torso got shorter without leaning/head-down: collapsed in the seat.
    if compressed:
        return "SUNKEN BODY"

    if shrugged:
        return "SHOULDERS RAISED"

    # Chest lifted / arched back, usually with the head tipped back.
    if stretched and (leaning_back or head_back):
        return "HYPEREXTENSION"

    if leaning_forward:
        return "LEANING FORWARD"

    if head_down:
        return "HEAD DOWN"

    if leaning_back:
        return "LEANING BACK"

    if head_back:
        return "HEAD TILTED BACK"

    if head > threshold_for("head", HEAD_THRESHOLD):
        return "FORWARD HEAD"

    if spine > threshold_for("spine", SPINE_THRESHOLD):
        return "LEANING SIDEWAYS"

    if shoulder > threshold_for("shoulder", SHOULDER_THRESHOLD):
        return "SHOULDER DROP"

    if pelvis > threshold_for("pelvis", PELVIS_THRESHOLD):
        return "PELVIS ASYMMETRY"

    if score >= 90:
        return "GOOD POSTURE"

    return "NEEDS ATTENTION"


def stabilize_label(raw_label, now):
    """Only switch the displayed label once the new one has persisted."""
    state = label_state

    if state["stable"] == "CALIBRATING":
        state["stable"] = raw_label
        state["pending"] = None
    elif raw_label == state["stable"]:
        state["pending"] = None
    elif raw_label != state["pending"]:
        state["pending"] = raw_label
        state["since"] = now
    elif now - state["since"] >= LABEL_HOLD_SECONDS:
        state["stable"] = raw_label
        state["pending"] = None

    return state["stable"]


# ============================================================
# Camera worker
# ============================================================

def camera_worker():
    global camera
    global tracking_active
    global latest_frame
    global calibration_valid_since
    global calibration_samples
    global smoothed_metrics
    global smoothed_pose_points

    global face_last_seen
    global last_update
    global last_valid_pose_time
    global face_status
    global facing_direction
    global posture_status

    global current_metrics
    global metric_deviations
    global instantaneous_score
    global rolling_score

    print("CAMERA WORKER STARTED", flush=True)

    camera = open_camera()

    if camera is None:
        print("ERROR: Could not open webcam.", flush=True)

        with state_lock:
            tracking_active = False

        return

    print("WEBCAM OPENED SUCCESSFULLY", flush=True)

    try:
        with mp_pose.Pose(
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose, mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        ) as face_detector:

            while not stop_event.is_set():
                success, frame = camera.read()

                if not success:
                    print(
                        "WARNING: Camera frame could not be read.",
                        flush=True
                    )
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                pose_results = pose.process(rgb)
                face_results = face_detector.process(rgb)

                now = time.time()
                detected_face = bool(face_results.detections)

                with state_lock:

                    if detected_face:
                        face_last_seen = now
                        face_status = "FACE DETECTED"
                        face_color = (0, 255, 0)
                    else:
                        face_status = "NO FACE DETECTED"
                        face_color = (0, 0, 255)

                    pose_is_valid = False
                    facing_direction = "UNKNOWN"
                    core_visible = 0.0
                    landmarks = None

                    if pose_results.pose_landmarks:
                        filtered_pose = smooth_pose_landmarks(
                            pose_results.pose_landmarks
                        )
                        landmarks = filtered_pose.landmark

                        mp_draw.draw_landmarks(
                            frame,
                            filtered_pose,
                            mp_pose.POSE_CONNECTIONS
                        )

                        core_indices = [11, 12, 23, 24]
                        core_visible = min(
                            landmarks[i].visibility
                            for i in core_indices
                        )

                        if core_visible >= 0.4:
                            pose_is_valid = True
                            last_valid_pose_time = now
                            world_landmarks = (
                                pose_results.pose_world_landmarks.landmark
                                if pose_results.pose_world_landmarks
                                else None
                            )
                            raw_metrics = calculate_metrics(
                                landmarks, world_landmarks
                            )
                            current_metrics = smooth_metrics(raw_metrics)

                            if pose_results.pose_world_landmarks:
                                world = pose_results.pose_world_landmarks.landmark
                                shoulder_depth = world[11].z - world[12].z
                                hip_depth = world[23].z - world[24].z
                                turn_score = (shoulder_depth + hip_depth) / 2.0

                                if abs(turn_score) >= TURN_THRESHOLD:
                                    facing_direction = (
                                        "LEFT" if turn_score > 0 else "RIGHT"
                                    )
                                elif detected_face:
                                    facing_direction = "FORWARD"
                                elif (
                                    face_last_seen is not None
                                    and now - face_last_seen >= FACE_LOST_SECONDS
                                ):
                                    facing_direction = "BACKWARD"
                                else:
                                    facing_direction = "UNKNOWN"
                            elif detected_face:
                                facing_direction = "FORWARD"
                    else:
                        smoothed_pose_points = None
                        smoothed_metrics = empty_metrics()

                    # Calibration requires continuous valid torso data.
                    if baseline_metrics is None:
                        posture_status = "CALIBRATING"

                        if pose_is_valid:
                            if calibration_valid_since is None:
                                calibration_valid_since = now
                                calibration_samples = []

                            calibration_samples.append(current_metrics.copy())
                            elapsed = now - calibration_valid_since
                            remaining = max(0.0, CALIBRATION_SECONDS - elapsed)

                            put_text(
                                frame,
                                "HOLD A COMFORTABLE, UPRIGHT POSTURE",
                                35,
                                (0, 255, 255)
                            )
                            put_text(
                                frame,
                                f"Calibrating: {remaining:.1f}s",
                                70,
                                (0, 255, 255),
                                0.8
                            )

                            if elapsed >= CALIBRATION_SECONDS:
                                establish_baseline()
                        else:
                            calibration_valid_since = None
                            calibration_samples = []
                            put_text(
                                frame,
                                "Keep shoulders and hips visible to calibrate",
                                35,
                                (0, 165, 255)
                            )

                    # Scoring phase.
                    elif pose_is_valid:
                        metric_deviations = compute_deviations(
                            current_metrics, baseline_metrics
                        )

                        instantaneous_score = (
                            calculate_instantaneous_score(
                                metric_deviations
                            )
                        )

                        current_poor_seconds = update_fatigue(
                            instantaneous_score, now
                        )

                        fatigue_adjusted_score = (
                            apply_fatigue_multiplier(
                                instantaneous_score,
                                current_poor_seconds
                            )
                        )

                        rolling_score = update_rolling_score(
                            fatigue_adjusted_score
                        )

                        posture_status = stabilize_label(
                            posture_label(
                                metric_deviations,
                                instantaneous_score
                            ),
                            now
                        )

                        status_color = (
                            (0, 255, 0)
                            if posture_status == "GOOD POSTURE"
                            else (0, 0, 255)
                        )

                        put_text(
                            frame,
                            f"Score: {rolling_score:.0f}",
                            35,
                            (255, 255, 255),
                            0.7
                        )

                        y = 62
                        for label, key, unit in (
                            ("Head", "head", "deg"),
                            ("Spine", "spine", "deg"),
                            ("Shoulder", "shoulder", "deg"),
                            ("Pelvis", "pelvis", "deg"),
                            ("Pitch", "head_pitch", "%"),
                            ("Lean", "lean", "deg"),
                            ("Torso", "torso", "%"),
                            ("Neck", "neck", "%")
                        ):
                            value = metric_deviations.get(key)
                            text = (
                                f"{label}: {value:+.1f}{unit}"
                                if value is not None
                                else f"{label}: --"
                            )
                            put_text(frame, text, y, (255, 255, 255), 0.5)
                            y += 24

                        put_text(frame, posture_status, 275, status_color, 0.7)

                    put_text(frame, face_status, 305, face_color, 0.65)
                    put_text(
                        frame,
                        f"Facing: {facing_direction}",
                        335,
                        (255, 255, 0),
                        0.65
                    )

                    last_update = now

                success, encoded = cv2.imencode(".jpg", frame)

                if not success:
                    continue

                with frame_lock:
                    latest_frame = encoded.tobytes()

                time.sleep(0.01)

    except Exception as error:
        print(
            "CAMERA WORKER ERROR:",
            repr(error),
            flush=True
        )

    finally:
        if camera is not None:
            camera.release()
            camera = None

        with state_lock:
            tracking_active = False

        with frame_lock:
            latest_frame = None

        print("CAMERA WORKER STOPPED", flush=True)


# ============================================================
# Video stream
# ============================================================

def generate_frames():
    while True:
        with frame_lock:
            frame_bytes = latest_frame

        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        with state_lock:
            active = tracking_active

        time.sleep(0.03 if active else 0.1)


# ============================================================
# Routes
# ============================================================

@app.route("/")
def home():
    return render_template("landing.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/feather_test")
def feather_test():
    connected = connect_to_feather()
    return jsonify({
        "connected": connected,
        "port": SERIAL_PORT
    })


@app.route("/feather_send_test")
def feather_send_test():
    if not send_serial_line("home"):
        return jsonify({"success": False, "error": "Feather not connected"}), 503

    last_servo_angles.update(SERVO_HOME)
    return jsonify({"success": True, "command": "home"})


@app.route("/feather_head_test")
def feather_head_test():
    return send_servo_test(0, 82)


def send_servo_test(servo_index, angle):
    if servo_index not in (0, 1, 2) or not 0 <= angle <= 180:
        return jsonify({"success": False, "error": "Invalid servo or angle"}), 400

    if not connect_to_feather():
        return jsonify({"success": False, "error": "Feather not connected"}), 503

    try:
        with servo_serial_lock:
            servo_serial.reset_input_buffer()
            servo_serial.write(b"home\n")
            servo_serial.flush()
            time.sleep(0.5)
            response = servo_serial.read_all().decode(errors="replace")

            servo_serial.write(f"{servo_index} {angle}\n".encode())
            servo_serial.flush()

        last_servo_angles[servo_index] = angle

        return jsonify({
            "success": True,
            "servo": servo_index,
            "target_angle": angle,
            "feather_response": response
        })
    except (serial.SerialException, OSError) as error:
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/servo_test")
def servo_test():
    try:
        servo_index = int(request.args.get("servo", "0"))
        angle = int(request.args.get("angle", "90"))
    except ValueError:
        return jsonify({"success": False, "error": "Servo and angle must be integers"}), 400

    return send_servo_test(servo_index, angle)


@app.route("/servo_control")
def servo_control():
    """Toggle mimic mode: /servo_control?enabled=1 or ?enabled=0"""
    global SERVO_CONTROL_ENABLED

    enabled = request.args.get("enabled")
    if enabled is not None:
        SERVO_CONTROL_ENABLED = enabled in ("1", "true", "on")

    return jsonify({
        "servo_control_enabled": SERVO_CONTROL_ENABLED,
        "last_servo_angles": last_servo_angles
    })


@app.route("/robot_home")
def robot_home():
    """Manual test: open /robot_home in the browser to send the robot home."""
    ok = robot_go_home("manual /robot_home")
    return jsonify({"success": ok, "home": SERVO_HOME}), (200 if ok else 503)


@app.route("/status")
def status():
    with state_lock:
        return jsonify({
            "posture_status": posture_status,
            "face_status": face_status,
            "facing_direction": facing_direction,

            "baseline": baseline_metrics,
            "current_metrics": current_metrics,
            "deviations": metric_deviations,

            "instantaneous_score": instantaneous_score,
            "rolling_score": rolling_score,
            "poor_posture_seconds": poor_posture_seconds,

            "calibrating": baseline_metrics is None,
            "calibration_seconds": CALIBRATION_SECONDS,
            "posture_tolerance": POSTURE_TOLERANCE,
            "tracking_active": tracking_active,
            "last_update": last_update,

            "servo_control_enabled": SERVO_CONTROL_ENABLED,
            "servo_angles": last_servo_angles
        })


@app.route("/config", methods=["POST"])
def config():
    global CALIBRATION_SECONDS
    global POSTURE_TOLERANCE

    data = request.get_json(silent=True) or {}

    try:
        new_duration = int(
            data.get("calibration_seconds", CALIBRATION_SECONDS)
        )

        new_tolerance = float(
            data.get("posture_tolerance", POSTURE_TOLERANCE)
        )

    except (ValueError, TypeError):
        return jsonify({
            "error": "Settings must be numeric."
        }), 400

    if not 1 <= new_duration <= 30:
        return jsonify({
            "error": "Calibration must be between 1 and 30 seconds."
        }), 400

    if not 1 <= new_tolerance <= 45:
        return jsonify({
            "error": "Tolerance must be between 1 and 45 degrees."
        }), 400

    CALIBRATION_SECONDS = new_duration
    POSTURE_TOLERANCE = new_tolerance
    robot_go_home("settings changed")

    return jsonify({
        "message": "Settings saved.",
        "calibration_seconds": CALIBRATION_SECONDS,
        "posture_tolerance": POSTURE_TOLERANCE
    })


@app.route("/recalibrate", methods=["POST"])
def recalibrate():
    reset_calibration()
    robot_go_home("recalibrate")

    return jsonify({
        "message": "Calibration restarted."
    })


@app.route("/start_tracking", methods=["POST"])
def start_tracking():
    global camera_thread
    global tracking_active
    global latest_frame

    with state_lock:
        already_running = (
            camera_thread is not None
            and camera_thread.is_alive()
        )

    if already_running:
        robot_go_home("start tracking")
        return jsonify({
            "message": "Tracking is already running.",
            "active": True
        })

    reset_calibration()
    stop_event.clear()
    start_servo_worker()
    robot_go_home("start tracking")

    with frame_lock:
        latest_frame = None

    with state_lock:
        tracking_active = True

    camera_thread = threading.Thread(
        target=camera_worker,
        daemon=True
    )

    camera_thread.start()

    return jsonify({
        "message": "Tracking started.",
        "active": True
    })


@app.route("/stop_tracking", methods=["POST"])
def stop_tracking():
    global tracking_active
    global latest_frame

    stop_event.set()

    with state_lock:
        tracking_active = False

    with frame_lock:
        latest_frame = None

    # Return the robot to its calibrated home pose.
    robot_go_home("stop tracking")

    return jsonify({
        "message": "Tracking stopped.",
        "active": False
    })


# ============================================================
# Start application
# ============================================================

if __name__ == "__main__":
    print("Starting ErgoBot backend...", flush=True)

    # Connect up front: the first connect waits 2s for the RP2040 to boot.
    connect_to_feather()
    start_servo_worker()

    app.run(
        host="127.0.0.1",
        port=5055,
        debug=False,
        threaded=True,
        use_reloader=False
    )