# Image Analysis

This folder contains the Image Analyst module for the Multimodal Crime / Incident Report Analyzer assignment.

The pipeline uses YOLOv8 object detection and OCR to analyze incident images and generate a structured CSV output. The current implementation is focused on fire and smoke detection using a Roboflow dataset, along with scene classification and text extraction from images.

## Assignment Mapping

This module covers the Image Analyst role from the assignment:

- detect relevant visual objects in incident images
- classify scene type
- extract visible text using OCR
- export structured results to CSV

Current CSV columns:

- `Image_ID`
- `Scene_Type`
- `Objects_Detected`
- `Bounding_Boxes`
- `Text_Extracted`
- `Confidence`

## Folder Structure

```text
images/
├── .env.example
├── README.md
├── app.config.yaml
├── data/
│   └── fire-2/
├── output/
│   ├── detection_sample.png
│   └── image_analyst_output.csv
├── requirements.txt
├── runs/
│   └── detect/
│       └── fire_model/
│           └── weights/
│               └── best.pt
└── src/
    └── main.py
```

## Main Files

- `src/main.py` - main script for training and inference
- `app.config.yaml` - model, inference, and output settings
- `.env.example` - environment variable template
- `requirements.txt` - Python dependencies
- `data/fire-2/` - local image dataset used by this module
- `runs/detect/fire_model/weights/best.pt` - trained YOLO weights for inference

## Requirements

- Python 3.x
- Tesseract OCR installed if OCR extraction is needed

Install Python dependencies from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## Environment Configuration

The module is configured to use the dataset stored inside this folder.

`.env`:

```env
ROBOFLOW_API_KEY=
DATASET_LOCATION=data/fire-2
TESSERACT_CMD=
```

Notes:

- Keep `DATASET_LOCATION=data/fire-2` to use the local dataset in this repo
- Leave `ROBOFLOW_API_KEY` empty unless you want to download a dataset from Roboflow
- Set `TESSERACT_CMD` only if Tesseract is installed outside your system PATH

## How To Run

From the `images` folder:

```bash
cd /Users/archana/Documents/SP2026/AI_For_Engg/Assignment_3/MultimodalCrimeReportAnalyzer/images
source .venv/bin/activate
```

### Inference Only

This is the recommended run mode for verifying the current assignment output.

To regenerate the current final CSV with 150 images:

```bash
python3 src/main.py --mode infer --config app.config.yaml --max-images 150
```

For a smaller quick test:

```bash
python3 src/main.py --mode infer --config app.config.yaml --max-images 25
```

To process all dataset images:

```bash
python3 src/main.py --mode infer --config app.config.yaml --max-images 0
```

### Train And Infer

If you want to retrain the model and then run inference:

```bash
python3 src/main.py --mode all --config app.config.yaml
```

## Output Files

After a successful run, the module generates:

- `output/image_analyst_output.csv`
- `output/detection_sample.png`

The CSV contains one row per processed image with the detected scene label, object classes, OCR text, and confidence score.

## Current Verification Result

The current setup successfully runs on the local dataset:

- dataset location: `data/fire-2`
- total dataset images available: `8456`
- current default inference limit in `app.config.yaml`: `150`

## Quick Output Check

To preview the CSV in terminal:

```bash
head output/image_analyst_output.csv
```

## Notes

- The current model is best suited for fire and smoke scene analysis
- OCR results may vary slightly across environments because Tesseract output is not always perfectly deterministic
- The final integration pipeline links this output by `Image_ID` through
  `integration/data/incident_map.csv`
