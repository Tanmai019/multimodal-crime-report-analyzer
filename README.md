# Multimodal Crime Report Analyzer

AI-powered multimodal incident analysis system integrating text, audio, document, image, and video intelligence into a unified reporting pipeline.

This project combines multiple AI and data processing workflows to analyze crime-related information across different modalities and generate a structured incident-level report through a centralized integration pipeline and Streamlit dashboard.

Developed as a collaborative graduate AI project with emphasis on multimodal machine learning, computer vision, NLP, OCR, and integrated analytics workflows.

---

# Tech Stack

`Python` `YOLOv8` `OpenCV` `spaCy` `Transformers` `OCR` `NLTK` `Pandas` `Streamlit` `Computer Vision` `NLP`

---

# Project Overview

The system processes and analyzes multiple forms of crime-related data including:

- Emergency-call audio
- Incident PDF reports
- Scene images
- Crime-related text
- Surveillance-style video clips

Each modality is analyzed independently and then merged into a unified incident-level intelligence report.

The final integration pipeline generates:
- Structured incident summaries
- Severity categorization
- Cross-modal entity mapping
- Streamlit dashboard visualization

---

# System Architecture

The project is divided into independent modality pipelines:

- `audio/` → audio transcription and emergency-call analysis
- `pdf/` → OCR and structured incident extraction
- `images/` → object detection and scene analysis
- `text/` → NLP-based text analysis and classification
- `video/` → motion analysis and surveillance event extraction
- `integration/` → unified incident aggregation and dashboard generation

The integration layer combines outputs from all modalities into a centralized incident intelligence dataset.

---

# Modalities

## Audio Analysis

Processes emergency-style audio recordings and extracts structured information from transcripts.

### Features
- Audio preprocessing
- Speech transcription
- Structured CSV generation
- Incident-level metadata extraction

---

## Document Analysis

Extracts incident information from PDF reports using OCR and NLP workflows.

### Features
- PDF parsing
- OCR-based extraction
- Named entity extraction
- Structured incident report generation

---

## Image Analysis

Uses computer vision workflows to analyze crime-scene images.

### Features
- YOLOv8-based object detection
- Scene analysis
- OCR extraction from images
- Structured detection outputs

---

## Text Analysis

Processes crime-related textual reports using NLP pipelines.

### Features
- Text preprocessing
- Named Entity Recognition (NER)
- Sentiment analysis
- Topic classification
- Severity labeling

---

## Video Analysis

Analyzes surveillance-style clips for motion and event detection.

### Features
- Motion detection
- Event extraction
- Frame-level analysis
- Structured event logging

---

## Integration Pipeline

Combines all modality outputs into a unified incident-level report.

### Features
- Cross-modal record mapping
- Incident aggregation
- Severity scoring
- Streamlit dashboard generation

---

# Repository Structure

```text
MultimodalCrimeReportAnalyzer/
├── audio/
├── pdf/
├── images/
├── text/
├── video/
├── integration/
├── README.md
└── yolov8n.pt
```

---

# Example Outputs

The implemented pipelines generate structured outputs including:

- Audio incident summaries
- OCR-extracted PDF reports
- Object detection logs
- NLP-based text classification results
- Video event logs
- Final integrated incident reports

The integration module produces:

```text
integration/output/final_integrated_incident_report.csv
```

---

# Streamlit Dashboard

The integration pipeline includes a Streamlit dashboard for viewing unified incident reports and modality outputs.

Run locally:

```bash
cd integration
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

# Skills Demonstrated

- Multimodal AI system design
- Computer vision workflows
- NLP pipelines and text analytics
- OCR-based document processing
- Audio and video analysis
- Streamlit dashboard development
- Data integration pipelines
- Structured incident intelligence generation

---

# Future Improvements

- Real-time multimodal streaming support
- Advanced LLM-based report summarization
- Cross-modal retrieval workflows
- Cloud deployment and scalable APIs
- Improved event correlation and entity linking
- Interactive investigation dashboard enhancements

---

# Notes

- Large raw datasets and media files were excluded from GitHub for repository optimization.
- Some workflows require GPU acceleration for efficient processing.
- The repository focuses on modular AI pipelines and portfolio-friendly project structure.
