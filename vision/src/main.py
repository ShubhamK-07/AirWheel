
import time
import math
from pathlib import Path

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "vision" / "models" / "hand_landmarker.task"

PALM_INDICES = [0, 5, 9, 13, 17]
MAX_STEERING_ANGLE = 45.0
STEERING_SMOOTHING_ALPHA = 0.25

CALIBRATION_STABLE_SECONDS = 3.5
CALIBRATION_TOLERANCE_DEG = 2.0
CALIBRATION_TIMEOUT_SECONDS = 10.0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19),
    (19, 20),
]


def draw_hand(frame, landmarks):
    height, width = frame.shape[:2]

    points = [
        (int(lm.x * width), int(lm.y * height))
        for lm in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame, points[start], points[end], (0, 255, 0), 2
        )

    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def palm_center(landmarks, width, height):
    """Estimate the palm center using five palm landmarks."""
    x = sum(landmarks[i].x for i in PALM_INDICES) / len(PALM_INDICES)
    y = sum(landmarks[i].y for i in PALM_INDICES) / len(PALM_INDICES)

    return int(x * width), int(y * height)


def angle_degrees(left, right):
    dx = right[0] - left[0]
    dy = right[1] - left[1]

    return math.degrees(math.atan2(dy, dx))


def wrap_degrees(angle):
    """Wrap an angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def clamp(value, low, high):
    return max(low, min(value, high))


def main():
    if not MODEL_PATH.is_file():
        print(f"ERROR: Hand model not found: {MODEL_PATH}")
        return 1

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return 1

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Calibration state
    neutral_angle = None
    smoothed_steering = None
    calibration_requested = False
    stable_since = None
    last_angle = None
    calibration_started_at = None

    print("AirWheel steering geometry started.")
    print("C: request/recalibrate neutral position")
    print("Q or Esc: exit")
    print("Calibration is voluntary and can be repeated at any time.")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    print("ERROR: Failed to read webcam frame.")
                    return 1

                # Match a typical mirrored selfie preview.
                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = time.perf_counter_ns() // 1_000_000

                result = landmarker.detect_for_video(
                    image, timestamp_ms
                )

                hands = result.hand_landmarks
                centers = []

                for landmarks in hands:
                    draw_hand(frame, landmarks)
                    centers.append(
                        palm_center(landmarks, width, height)
                    )

                current_angle = None
                steering = None

                if len(centers) >= 2:
                    # Determine left and right from screen coordinates.
                    centers.sort(key=lambda point: point[0])
                    left, right = centers[0], centers[1]

                    current_angle = angle_degrees(left, right)

                    # Draw the measured hand-to-hand line.
                    cv2.line(
                        frame, left, right, (255, 200, 0), 3
                    )
                    cv2.circle(
                        frame, left, 7, (255, 200, 0), -1
                    )
                    cv2.circle(
                        frame, right, 7, (255, 200, 0), -1
                    )
                    
                now = time.monotonic()

                if (
                    calibration_requested
                    and calibration_started_at is not None
                    and now - calibration_started_at >= CALIBRATION_TIMEOUT_SECONDS
                ):
                    calibration_requested = False
                    calibration_started_at = None
                    stable_since = None
                    last_angle = None
                
                    print("[INFO] Calibration timed out. Returning to normal mode.")
                # Handle calibration only after the user requests it.
                if calibration_requested:
                    if current_angle is None:
                        # Keep the request pending until both hands return.
                        stable_since = None
                        last_angle = None

                    else:
                        now = time.monotonic()

                        angle_changed = (
                            last_angle is None
                            or abs(
                                wrap_degrees(
                                    current_angle - last_angle
                                )
                            ) > CALIBRATION_TOLERANCE_DEG
                        )

                        if angle_changed:
                            # Movement resets the stability timer.
                            stable_since = now

                        elif stable_since is not None:
                            stable_duration = now - stable_since

                            if (
                                stable_duration
                                >= CALIBRATION_STABLE_SECONDS
                            ):
                                neutral_angle = current_angle
                                smoothed_steering = 0.0
                                calibration_requested = False
                                calibration_started_at = None
                                stable_since = None

                                print(
                                    "Calibration successful. "
                                    f"Neutral angle: "
                                    f"{neutral_angle:.2f} degrees"
                                )

                        last_angle = current_angle

                else:
                    stable_since = None
                    last_angle = None

                # Calculate and smooth steering relative to neutral.
                if current_angle is not None and neutral_angle is not None:
                    delta = wrap_degrees(current_angle - neutral_angle)
                
                    target_steering = clamp(
                        -delta / MAX_STEERING_ANGLE,
                        -1.0,
                        1.0,
                    )
                
                    if smoothed_steering is None:
                        # Initialize directly to avoid a startup ramp.
                        smoothed_steering = target_steering
                    else:
                        smoothed_steering += STEERING_SMOOTHING_ALPHA * (
                            target_steering - smoothed_steering
                        )
                
                    steering = smoothed_steering
                
                else:
                    steering = None
                
                    # Avoid carrying stale steering into a tracking interruption.
                    if current_angle is None:
                        smoothed_steering = None

                # Status display
                cv2.putText(
                    frame,
                    f"Hands detected: {len(hands)}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                if calibration_requested:
                    if current_angle is None:
                        status = "Calibration requested: show both hands"
                    elif stable_since is None:
                        status = "Calibration: hold wheel steady"
                    else:
                        elapsed = time.monotonic() - stable_since
                        remaining = max(
                            0.0,
                            CALIBRATION_STABLE_SECONDS - elapsed,
                        )
                        status = (
                            f"Calibrating: hold steady "
                            f"({remaining:.1f}s)"
                        )

                    cv2.putText(
                        frame, status, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 200, 255), 2,
                    )

                elif neutral_angle is None:
                    cv2.putText(
                        frame,
                        "Not calibrated - press C to calibrate",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 200, 255), 2,
                    )

                elif current_angle is None:
                    cv2.putText(
                        frame,
                        "Tracking lost - show both hands",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 200, 255), 2,
                    )

                else:
                    cv2.putText(
                        frame,
                        f"Steering: {steering:+.2f}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2,
                    )

                    cv2.putText(
                        frame,
                        f"Angle change: {delta:+.1f} deg",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2,
                    )
                                # Always display the current angle between the palm centers.
                if current_angle is not None:
                    angle_text = f"Current angle: {current_angle:+.1f} deg"
                else:
                    angle_text = "Current angle: --"
                
                cv2.putText(
                    frame,
                    angle_text,
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                
                                # Persistent key-binding help
                cv2.putText(
                    frame,
                    "C: Calibrate/Recalibrate    Q/Esc: Exit",
                    (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1,
                )

                cv2.imshow(
                    "AirWheel - Steering Geometry", frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("c"):
                    calibration_requested = True
                    calibration_started_at = time.monotonic()
                    stable_since = None
                    last_angle = None

                    print("[INFO] Calibration requested. Show both hands and hold steady.")

                elif key == ord("q") or key == 27:
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Camera and hand tracker released.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
