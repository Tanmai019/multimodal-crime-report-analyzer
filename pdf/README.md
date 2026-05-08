# Document Analysis Module

PDF-based incident intelligence extraction pipeline for processing structured and scanned incident reports using OCR and NLP workflows.

This module is part of the larger Multimodal Crime Report Analyzer system and focuses on extracting structured incident information from PDF reports using text extraction, OCR fallback processing, and named entity recognition techniques.

The pipeline converts unstructured incident documents into structured CSV outputs suitable for downstream multimodal integration.

---

# Tech Stack

`Python` `PyMuPDF` `Tesseract OCR` `spaCy` `NLP` `OCR` `Pandas` `PDF Processing`

---

# Module Overview

The document analysis pipeline processes incident-related PDF reports and extracts:

- Incident type
- Incident date
- Location information
- Officer information
- Incident summaries

The extracted information is converted into structured CSV format for integration into the multimodal incident intelligence system.

---

# Features

- Native PDF text extraction
- OCR fallback for scanned PDFs
- Named Entity Recognition (NER)
- Rule-based field extraction
- Structured CSV export
- PDF-to-incident intelligence workflow

---

# Extracted Fields

Generated output columns:

- `Report_ID`
- `Incident_Type`
- `Date`
- `Location`
- `Officer`
- `Summary`

---

# Folder Structure

```text
pdf/
├── data/
├── output/
├── src/
├── requirements.txt
└── README.md
```

---

# Processing Pipeline

For each PDF report, the system performs:

1. Native PDF text extraction using PyMuPDF
2. OCR fallback for scanned/image-based documents
3. Named entity extraction using spaCy
4. Rule-based field identification
5. Incident summary generation
6. Structured CSV export

---

# Core Technologies

The pipeline uses:

- `PyMuPDF` for direct PDF text extraction
- `Tesseract OCR` for scanned document processing
- `spaCy` for NLP-based entity extraction
- Rule-based heuristics for structured field mapping

---

# Setup

From the `pdf/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

# OCR Setup

For scanned PDFs, install Tesseract OCR:

```bash
brew install tesseract
```

If Tesseract is not available in system PATH:

```bash
export TESSERACT_CMD="/opt/homebrew/bin/tesseract"
```

---

# Input

Place PDF files inside:

```text
data/
```

Example sample file:

```text
data/LESO2.pdf
```

You may also pass custom PDF paths directly through the command line.

---

# Running the Pipeline

## Example Run

```bash
python3 src/document_analysis.py data/LESO2.pdf -o output/incident_extract.csv -v
```

---

## Generic Workflow

```bash
python3 src/document_analysis.py path/to/report.pdf -o output/incident_extract.csv
```

---

# Optional Arguments

- `--report-id` → override default report identifier
- `--spacy-model` → specify custom spaCy model
- `-v` or `--verbose` → enable debugging logs

If no PDF path is provided, the script prompts for one interactively.

---

# Output

Generated output:

```text
output/incident_extract.csv
```

The pipeline:
- prints extracted results in terminal
- exports structured incident intelligence to CSV

---

# Example Workflow

Input:
- incident PDF reports

Processing:
- text extraction
- OCR fallback
- NLP entity recognition
- structured field extraction

Output:
- structured incident report CSV

---

# Skills Demonstrated

- OCR-based document processing
- PDF parsing workflows
- Named Entity Recognition (NER)
- Rule-based information extraction
- NLP-driven structured extraction
- Incident intelligence generation
- Multimodal AI integration

---

# Limitations

- OCR accuracy depends heavily on scan quality
- Poorly formatted PDFs may reduce extraction reliability
- Some extracted fields may return `Not Found`
- Current implementation processes one PDF at a time

---

# Integration

The generated output integrates into the larger multimodal incident analysis system through:

```text
integration/data/incident_map.csv
```

using:

```text
Report_ID
```

as the linking identifier.

---

# Notes

- The system first attempts native PDF extraction before using OCR fallback workflows.
- Large PDF datasets were excluded from GitHub for repository optimization.
- This module was designed as part of a multimodal AI incident analysis platform.
