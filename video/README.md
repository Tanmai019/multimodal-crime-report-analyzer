# Video Analysis Module

Computer vision pipeline for analyzing surveillance-style video clips and generating structured event intelligence outputs.

This module is part of the larger Multimodal Crime Report Analyzer system and focuses on detecting motion patterns, pedestrian activity, and scene-level events from surveillance-style video footage using YOLOv8-based person detection and rule-based event analysis.

The pipeline converts raw video clips into structured event logs suitable for downstream multimodal integration.

---

# Tech Stack

`Python` `YOLOv8` `OpenCV` `Computer Vision` `Video Analytics` `Ultralytics` `Pandas`

---

# Module Overview

The video analysis pipeline processes surveillance-style video clips and extracts:

- Motion events
- Pedestrian activity
- Group movement patterns
- Frame-level event detection
- Structured event logs

The processed outputs are exported into structured CSV format for integration into the larger multimodal incident analysis system.

---

# Features

- YOLOv8-based person detection
- Motion-based event analysis
- Surveillance event classification
- Frame sampling workflows
- Structured event logging
- Intermediate frame extraction

---

# Supported Input

The module currently processes:

```text
.mpg
```

video files stored inside:

```text
data/
```

Example clips:

- `Browse1.mpg`
- `Fight_Chase.mpg`
- `LeftBag.mpg`
- `Rest_FallOnFloor.mpg`
- `Walk1.mpg`

---

# Output

Generated output:

```text
output/video_event_log.csv
```

---

# Output Columns

The generated CSV contains:

- `Clip_ID`
- `Timestamp`
- `Frame_ID`
- `Event_Detected`
- `Persons_Count`
- `Confidence`

The pipeline also saves sampled frames inside:

```text
frames/
```

---

# Folder Structure

```text
video/
├── data/
├── frames/
├── output/
├── src/
├── requirements.txt
└── README.md
```

---

# Processing Pipeline

For each surveillance clip, the system performs:

1. Video loading
2. Frame sampling
3. Motion analysis
4. YOLOv8 person detection
5. Event classification
6. Structured CSV export
7. Frame snapshot generation

---

# Setup

From the `video/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# YOLOv8 Model Weights

The pipeline uses Ultralytics YOLOv8 person detection.

Recommended setup:

```text
../yolov8n.pt
```

The script:
- first checks for local model weights
- automatically downloads weights if unavailable

---

# Running the Pipeline

From the `video/` folder:

```bash
python3 src/motion_detection.py
```

---

# Pipeline Behavior

The workflow automatically:

- reads `.mpg` clips
- samples frames at approximately 1-second intervals
- detects motion and pedestrians
- infers event labels
- generates structured event logs
- saves sampled frames

---

# Event Labels

Current supported event labels include:

- `pedestrian appearance`
- `pedestrian presence`
- `stationary pedestrian`
- `pedestrian movement`
- `brisk pedestrian movement`
- `group presence`
- `active group movement`
- `high-intensity group activity`

Low-confidence or irrelevant background motion is filtered from the final structured output.

---

# Example Workflow

Input:
- surveillance-style video clips

Processing:
- frame sampling
- object detection
- motion analysis
- event classification

Output:
- structured video event intelligence CSV

---

# Skills Demonstrated

- Video analytics workflows
- Computer vision pipelines
- YOLOv8 object detection
- Motion detection systems
- Event classification
- Surveillance intelligence workflows
- Structured AI pipeline development
- Multimodal system integration

---

# Limitations

- Current event classification is rule-based
- Complex crowded scenes may reduce detection accuracy
- Lighting and camera quality can affect performance
- Motion heuristics are intentionally conservative

---

# Integration

The generated output integrates into the larger multimodal incident analysis system through:

```text
integration/data/incident_map.csv
```

using:

```text
Clip_ID
```

as the linking identifier.

---

# Notes

- The `frames/` directory contains intermediate generated artifacts and can be recreated by rerunning the pipeline.
- If no `.mpg` clips are found, the script exits without generating output.
- Large raw video files were excluded from GitHub for repository optimization.
- GPU acceleration improves YOLOv8 inference performance significantly.
