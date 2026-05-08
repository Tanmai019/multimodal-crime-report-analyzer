# Image Analysis Module

Computer vision pipeline for analyzing incident-related images using object detection, OCR, and scene classification workflows.

This module is part of the larger Multimodal Crime Report Analyzer system and focuses on extracting structured intelligence from images through YOLOv8-based object detection and OCR-driven text extraction.

The current implementation is optimized for fire and smoke scene analysis using a Roboflow-based dataset.

---

# Tech Stack

`Python` `YOLOv8` `OpenCV` `OCR` `Tesseract` `Computer Vision` `Pandas` `Roboflow`

---

# Module Overview

The image analysis pipeline processes incident-related images and generates structured outputs containing:

- Scene classification
- Object detection results
- Bounding box coordinates
- OCR-extracted text
- Confidence scores

The processed outputs are exported into structured CSV format for integration into the larger multimodal incident analysis system.

---

# Features

- YOLOv8-based object detection
- Fire and smoke scene analysis
- OCR-based text extraction
- Scene classification
- Structured CSV export
- Inference and retraining workflows

---

# Current Output Columns

Generated CSV fields:

- `Image_ID`
- `Scene_Type`
- `Objects_Detected`
- `Bounding_Boxes`
- `Text_Extracted`
- `Confidence`

---

# Folder Structure

```text
images/
├── .env.example
├── README.md
├── app.config.yaml
├── data/
├── output/
├── requirements.txt
├── runs/
└── src/
```

---

# Main Files

- `src/main.py` → main training and inference pipeline
- `app.config.yaml` → inference and output configuration
- `.env.example` → environment variable template
- `requirements.txt` → project dependencies
- `runs/detect/fire_model/weights/best.pt` → trained YOLOv8 model weights

---

# Dataset

The module currently uses a local fire and smoke incident dataset:

```text
data/fire-2
```

### Current Dataset Statistics

- Total available images: `8456`
- Default inference limit: `150`

---

# Processing Pipeline

For each input image, the system performs:

1. Image preprocessing
2. Object detection using YOLOv8
3. Scene classification
4. OCR-based text extraction
5. Confidence score estimation
6. Structured CSV export

---

# Setup

From the `images/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

---

# Environment Configuration

`.env`

```env
ROBOFLOW_API_KEY=
DATASET_LOCATION=data/fire-2
TESSERACT_CMD=
```

### Notes

- Keep `DATASET_LOCATION=data/fire-2` to use the local dataset
- `ROBOFLOW_API_KEY` is optional for external dataset downloads
- `TESSERACT_CMD` is only required if Tesseract is outside system PATH

---

# Running the Pipeline

## Inference Only

Recommended workflow for reproducing assignment outputs.

### Run on 150 images

```bash
python3 src/main.py --mode infer --config app.config.yaml --max-images 150
```

---

### Quick Test

```bash
python3 src/main.py --mode infer --config app.config.yaml --max-images 25
```

---

### Process Full Dataset

```bash
python3 src/main.py --mode infer --config app.config.yaml --max-images 0
```

---

## Train + Inference Workflow

```bash
python3 src/main.py --mode all --config app.config.yaml
```

---

# Outputs

Generated files:

- `output/image_analyst_output.csv`
- `output/detection_sample.png`

The CSV contains:
- detected scene labels
- object classes
- OCR text
- confidence scores

---

# Example Workflow

Input:
- incident scene images

Processing:
- YOLOv8 detection
- OCR extraction
- scene classification

Output:
- structured image intelligence CSV

---

# Skills Demonstrated

- Computer vision workflows
- YOLOv8 object detection
- OCR-based image processing
- Scene classification
- Structured AI pipelines
- Dataset-driven model inference
- Multimodal system integration

---

# Quick Output Preview

```bash
head output/image_analyst_output.csv
```

---

# Limitations

- The current model is optimized primarily for fire and smoke scenarios
- OCR accuracy may vary across environments and image quality
- Complex scenes with overlapping objects may reduce detection accuracy
- Confidence scores depend heavily on dataset distribution and lighting conditions

---

# Integration

The generated output integrates into the larger multimodal incident analysis system through:

```text
integration/data/incident_map.csv
```

using:

```text
Image_ID
```

as the linking identifier.

---

# Notes

- Large datasets and generated outputs were excluded from GitHub for repository optimization.
- GPU acceleration improves inference speed significantly.
- This module was developed as part of a larger multimodal AI incident analysis platform.
