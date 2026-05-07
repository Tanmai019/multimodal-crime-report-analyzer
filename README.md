# Multimodal Crime Report Analyzer

This project is a team-built multimodal crime analysis system for assignment work across audio, document, image, text, video, and final integration stages. Each modality lives in its own folder so it can be developed, tested, and documented independently.

## Assignment Folder Mapping

This repository now matches the assignment folder names directly:

- `audio/`
- `pdf/`
- `images/`
- `video/`
- `text/`
- `integration/`

## Current Modules

- `audio/` processes emergency-call audio and writes structured CSV output
- `pdf/` extracts structured incident details from PDF reports
- `images/` analyzes scene images and writes structured detection CSV output
- `text/` analyzes crime-related text records and writes structured CSV outputs
- `video/` analyzes surveillance-style clips and generates an event log
- `integration/` merges the available modality outputs into one assignment-ready incident report and dashboard

## Repository Structure

```text
MultimodalCrimeReportAnalyzer/
├── README.md
├── audio/
│   ├── README.md
│   ├── data/
│   ├── output/
│   ├── requirements.txt
│   └── src/
├── pdf/
│   ├── README.md
│   ├── data/
│   ├── output/
│   ├── requirements.txt
│   └── src/
├── images/
│   ├── README.md
│   ├── data/
│   ├── output/
│   ├── requirements.txt
│   └── src/
├── integration/
│   ├── app.py
│   ├── README.md
│   ├── data/
│   ├── output/
│   ├── requirements.txt
│   └── src/
├── text/
│   ├── README.md
│   ├── data/
│   ├── output/
│   ├── requirements.txt
│   ├── requirements-full.txt
│   └── src/
├── video/
│   ├── README.md
│   ├── data/
│   ├── frames/
│   ├── output/
│   ├── requirements.txt
│   └── src/
└── yolov8n.pt
```

## Module Guides

Each implemented module has its own README with setup and run instructions:

- [Audio Analysis](audio/README.md)
- [Document Analysis](pdf/README.md)
- [Image Analysis](images/README.md)
- [Text Analysis](text/README.md)
- [Video Analysis](video/README.md)
- [Integration](integration/README.md)

## Quick Start

Choose the module you want to run, move into that folder, create a virtual environment, install dependencies, and follow that module's README.

## Demo Only: Run Streamlit Dashboard

If you only want to show the final dashboard on another laptop, you can skip the
full audio, image, text, PDF, and video pipelines. The repository already
includes a sample integrated CSV at
`integration/output/final_integrated_incident_report.csv`, and the Streamlit app
uses that file by default.

From the repository root:

```bash
cd integration
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Windows activation command:

```bash
.venv\Scripts\activate
```

Then open `http://localhost:8501` in a browser.

If the integrated CSV is missing for any reason, generate it once with:

```bash
python3 src/integrate_reports.py
streamlit run app.py
```

Examples:

Audio analysis:

```bash
cd audio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/audio_analyzer.py --data data --output output/audio_output.csv --max 10 --summary
```

Document analysis:

```bash
cd pdf
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python3 src/document_analysis.py data/LESO2.pdf -o output/incident_extract.csv -v
```

Text analysis:

```bash
cd text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/text_analysis.py --input "data/CrimeReport (1).txt" --no-transformers
```

Image analysis:

```bash
cd images
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/main.py --mode infer --config app.config.yaml --max-images 150
```

Video analysis:

```bash
cd video
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/motion_detection.py
```

Integration:

```bash
cd integration
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/integrate_reports.py
streamlit run app.py
```

The integration module uses a manual mapping file at `integration/data/incident_map.csv`
to link `Call_ID`, `Report_ID`, `Image_ID`, `Clip_ID`, and `Text_ID` to a shared
`Incident_ID`.

## Outputs

Current outputs produced by the implemented modules:

- `audio/output/audio_output.csv`
- `pdf/output/incident_extract.csv`
- `images/output/image_analyst_output.csv`
- `text/output/text_output.csv`
- `video/output/video_event_log.csv`
- `integration/output/final_integrated_incident_report.csv`

## Notes

- `video/` uses the `yolov8n.pt` model file already present in the project root.
- `pdf/` may require Tesseract OCR for scanned PDFs.
- `text/` can run in a lightweight rule-based mode with `--no-transformers`.
- `integration/` includes a Streamlit dashboard at `integration/app.py`.
