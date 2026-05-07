"""Rule-based video pipeline for turning sampled CCTV frames into event logs."""

import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def timestamp_to_seconds(timestamp):
    return (
        int(timestamp[0:2]) * 3600
        + int(timestamp[3:5]) * 60
        + int(timestamp[6:8])
    )


def get_motion_status(motion_score):
    if motion_score < 0.01:
        return "low motion"
    if motion_score < 0.03:
        return "medium motion"
    return "high motion"


def detect_persons(model, frame, conf_threshold=0.35):
    """
    Detect persons in the frame using YOLO.
    Returns:
    - detected_objects: detected class names
    - persons_count: number of detected persons
    - max_confidence: max object confidence in frame
    - primary_person: largest detected person box (used for motion heuristics)
    """
    results = model(frame, verbose=False)[0]

    # Restricting the detector to people keeps the event logic stable and avoids
    # noisy labels that are not useful for this prototype.
    allowed_classes = {"person"}

    detected_objects = []
    persons_count = 0
    max_confidence = 0.0
    primary_person = None
    max_area = 0.0

    if results.boxes is None:
        return [], 0, 0.0, None

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if confidence < conf_threshold:
            continue

        class_name = results.names[class_id]
        if class_name not in allowed_classes:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1
        area = width * height

        detected_objects.append(class_name)
        max_confidence = max(max_confidence, confidence)

        if class_name == "person":
            persons_count += 1
            if area > max_area:
                max_area = area
                primary_person = {
                    "center_x": (x1 + x2) / 2,
                    "center_y": (y1 + y2) / 2,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / max(height, 1.0),
                    "area": area,
                }

    detected_objects = sorted(set(detected_objects))
    return detected_objects, persons_count, max_confidence, primary_person


def infer_event(
    motion_score,
    motion_status,
    detected_objects,
    persons_count,
    previous_persons_count,
    no_person_gap,
    person_shift,
    stationary_streak,
    primary_person,
    multiperson_streak,
):
    """
    Infer event from actual frame content.
    This avoids using filename-based labels.
    """

    has_person = "person" in detected_objects
    if persons_count == 0 and motion_score < 0.02:
        return "stable scene"

    if persons_count == 0 and motion_score >= 0.02:
        return "background scene movement"

    # Require multi-person detections to persist before calling it strong group activity.
    if persons_count >= 2 and motion_score >= 0.03 and multiperson_streak >= 2:
        return "high-intensity group activity"

    if persons_count >= 2 and (motion_status == "medium motion" or person_shift >= 8):
        return "active group movement"

    if persons_count >= 2:
        return "group presence"

    if has_person and stationary_streak >= 3 and motion_status == "low motion":
        return "stationary pedestrian"

    if has_person and (
        motion_score >= 0.03
        or (motion_status == "high motion" and person_shift >= 10)
        or (person_shift >= 14 and motion_score >= 0.018)
    ):
        return "brisk pedestrian movement"

    if has_person and (
        person_shift >= 6
        or motion_score >= 0.015
        or motion_status == "medium motion"
    ):
        return "pedestrian movement"

    # Only call it an appearance after a meaningful no-person gap.
    if previous_persons_count == 0 and persons_count == 1 and no_person_gap >= 2:
        return "pedestrian appearance"

    if has_person:
        return "pedestrian presence"

    return "movement detected"


def compute_event_confidence(motion_score, object_confidence, persons_count, event_detected):
    """
    Combined confidence heuristic for prototype use.
    """
    confidence = max(0.45 + motion_score, object_confidence)

    if "high-intensity" in event_detected or "brisk" in event_detected:
        confidence += 0.05
    elif "active group" in event_detected:
        confidence += 0.04

    if persons_count >= 2:
        confidence += 0.03

    return round(min(confidence, 0.99), 2)


def build_event_log_rows(clip_rows):
    meaningful_events = {
        "pedestrian appearance",
        "pedestrian presence",
        "stationary pedestrian",
        "pedestrian movement",
        "brisk pedestrian movement",
        "group presence",
        "active group movement",
        "high-intensity group activity",
    }

    event_rows = []
    last_signature = None

    for row in clip_rows:
        if row["Event_Detected"] not in meaningful_events:
            last_signature = None
            continue

        signature = (row["Event_Detected"], row["Persons_Count"])
        if signature == last_signature:
            continue

        event_rows.append(row.copy())
        last_signature = signature

    cleaned_rows = []

    for index, row in enumerate(event_rows):
        next_row = event_rows[index + 1] if index + 1 < len(event_rows) else None
        previous_cleaned = cleaned_rows[-1] if cleaned_rows else None

        # Suppress likely one-frame double-person glitches.
        if row["Event_Detected"] == "group presence" and row["Confidence"] < 0.70:
            continue

        # Drop weak appearance rows that are likely detector flicker.
        if row["Event_Detected"] == "pedestrian appearance" and row["Confidence"] < 0.50:
            continue

        # If a brief appearance immediately turns into a more informative event,
        # keep only the stronger follow-up label.
        if (
            row["Event_Detected"] == "pedestrian appearance"
            and next_row is not None
            and row["Clip_ID"] == next_row["Clip_ID"]
            and next_row["Event_Detected"]
            in {
                "stationary pedestrian",
                "pedestrian movement",
                "brisk pedestrian movement",
            }
            and row["Persons_Count"] == next_row["Persons_Count"]
        ):
            current_seconds = timestamp_to_seconds(row["Timestamp"])
            next_seconds = timestamp_to_seconds(next_row["Timestamp"])
            if next_seconds - current_seconds <= 3:
                continue

        # Treat a brief presence row before a stationary period as a transition artifact.
        if (
            row["Event_Detected"] == "pedestrian presence"
            and next_row is not None
            and row["Clip_ID"] == next_row["Clip_ID"]
            and next_row["Event_Detected"] == "stationary pedestrian"
            and row["Persons_Count"] == next_row["Persons_Count"]
        ):
            current_seconds = timestamp_to_seconds(row["Timestamp"])
            next_seconds = timestamp_to_seconds(next_row["Timestamp"])
            if next_seconds - current_seconds <= 3:
                continue

        # Bridge short background gaps so the same event block is logged once.
        if (
            previous_cleaned is not None
            and row["Clip_ID"] == previous_cleaned["Clip_ID"]
            and row["Event_Detected"] == previous_cleaned["Event_Detected"]
            and row["Persons_Count"] == previous_cleaned["Persons_Count"]
        ):
            previous_seconds = timestamp_to_seconds(previous_cleaned["Timestamp"])
            current_seconds = timestamp_to_seconds(row["Timestamp"])
            if current_seconds - previous_seconds <= 3:
                continue

        cleaned_rows.append(row)

    return cleaned_rows


# -----------------------------
# Paths
# -----------------------------
# This script is intentionally runnable as a one-shot pipeline: import the
# helpers above, then execute the full clip-processing workflow below.
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
frames_root = base_dir / "frames"
output_dir = base_dir / "output"
output_csv = output_dir / "video_event_log.csv"

frames_root.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

video_files = sorted(data_dir.glob("*.mpg"))

if not video_files:
    print(f"No .mpg videos found in {data_dir}")
    raise SystemExit

# Load YOLO model
local_model_path = base_dir.parent / "yolov8n.pt"
model = YOLO(str(local_model_path if local_model_path.exists() else "yolov8n.pt"))

results_rows = []

for video_path in video_files:
    clip_id = video_path.stem
    frames_folder = frames_root / clip_id.lower()
    frames_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Skipping {clip_id}: could not open video.")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print(f"Skipping {clip_id}: invalid FPS value.")
        cap.release()
        continue

    # Sampling roughly one frame per second gives us lightweight event logs
    # without exploding the number of frames we need to process.
    frame_interval = max(1, int(fps))
    frame_count = 0
    saved_count = 0
    saved_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            frame_name = f"frame_{saved_count:03d}.jpg"
            frame_path = frames_folder / frame_name

            cv2.imwrite(str(frame_path), frame)

            saved_frames.append(
                {
                    "frame_id": frame_name.replace(".jpg", ""),
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                }
            )
            saved_count += 1

        frame_count += 1

    cap.release()

    if len(saved_frames) < 2:
        print(f"Skipping {clip_id}: not enough frames extracted.")
        continue

    clip_rows = []
    previous_person_center = None
    stationary_streak = 0
    multiperson_streak = 0
    no_person_gap = 0

    for i, current_info in enumerate(saved_frames):
        current_frame = cv2.imread(str(current_info["frame_path"]))
        if current_frame is None:
            continue

        # Motion is measured against the previous sampled frame so event labels
        # react to actual scene change instead of filename hints.
        if i == 0:
            motion_score = 0.0
        else:
            previous_info = saved_frames[i - 1]
            previous_frame = cv2.imread(str(previous_info["frame_path"]))
            if previous_frame is None:
                continue

            gray1 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
            gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

            diff = cv2.absdiff(gray1, gray2)
            _, threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            threshold = cv2.dilate(threshold, None, iterations=2)

            changed_pixels = cv2.countNonZero(threshold)
            total_pixels = threshold.shape[0] * threshold.shape[1]
            motion_score = changed_pixels / total_pixels

        motion_status = get_motion_status(motion_score)

        (
            detected_objects,
            persons_count,
            object_confidence,
            primary_person,
        ) = detect_persons(model, current_frame)

        if i == 0:
            previous_persons_count = 0
        else:
            previous_persons_count = clip_rows[-1]["Persons_Count"] if clip_rows else 0

        if primary_person and previous_person_center is not None:
            dx = primary_person["center_x"] - previous_person_center[0]
            dy = primary_person["center_y"] - previous_person_center[1]
            person_shift = (dx * dx + dy * dy) ** 0.5
        else:
            person_shift = 0.0

        # Track short temporal patterns so the final labels feel less jittery
        # than frame-by-frame detector output.
        if persons_count > 0 and person_shift < 8:
            stationary_streak += 1
        else:
            stationary_streak = 0

        if persons_count >= 2:
            multiperson_streak += 1
        else:
            multiperson_streak = 0

        if persons_count == 0:
            no_person_gap += 1
        else:
            current_no_person_gap = no_person_gap
            no_person_gap = 0

        event_detected = infer_event(
            motion_score=motion_score,
            motion_status=motion_status,
            detected_objects=detected_objects,
            persons_count=persons_count,
            previous_persons_count=previous_persons_count,
            no_person_gap=current_no_person_gap if persons_count > 0 else no_person_gap,
            person_shift=person_shift,
            stationary_streak=stationary_streak,
            primary_person=primary_person,
            multiperson_streak=multiperson_streak,
        )

        confidence = compute_event_confidence(
            motion_score=motion_score,
            object_confidence=object_confidence,
            persons_count=persons_count,
            event_detected=event_detected,
        )

        row = {
            "Clip_ID": clip_id,
            "Timestamp": format_timestamp(current_info["timestamp"]),
            "Frame_ID": current_info["frame_id"],
            "Event_Detected": event_detected,
            "Persons_Count": persons_count,
            "Confidence": confidence,
        }

        clip_rows.append(row)

        if primary_person:
            previous_person_center = (
                primary_person["center_x"],
                primary_person["center_y"],
            )
        else:
            previous_person_center = None

    event_rows = build_event_log_rows(clip_rows)
    results_rows.extend(event_rows)

    print(
        f"Processed {clip_id}: {len(saved_frames)} sampled frames, "
        f"{len(event_rows)} event rows"
    )

# Once every clip is processed, write the cleaned event rows as the one CSV the
# integration step expects from the video modality.
with open(output_csv, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Clip_ID",
            "Timestamp",
            "Frame_ID",
            "Event_Detected",
            "Persons_Count",
            "Confidence",
        ],
    )
    writer.writeheader()
    writer.writerows(results_rows)

print(f"Done. Event log CSV saved at: {output_csv}")
