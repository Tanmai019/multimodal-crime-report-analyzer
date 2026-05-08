# Text Analysis Module

NLP-driven crime text analysis pipeline for extracting structured incident intelligence from unstructured textual reports.

This module is part of the larger Multimodal Crime Report Analyzer system and focuses on analyzing crime-related text using NLP workflows including named entity recognition, sentiment analysis, topic classification, and severity labeling.

The pipeline converts unstructured text records into structured CSV outputs suitable for downstream multimodal integration and incident intelligence workflows.

---

# Tech Stack

`Python` `spaCy` `Transformers` `NLTK` `NLP` `Hugging Face` `Pandas` `Text Analytics`

---

# Module Overview

The text analysis pipeline processes crime-related textual records and extracts:

- Crime categories
- Location entities
- Sentiment labels
- Incident topics
- Severity classifications
- Named entities

The processed outputs are exported into structured CSV format for integration into the larger multimodal incident analysis system.

---

# Features

- NLP-based text preprocessing
- Named Entity Recognition (NER)
- Sentiment analysis
- Topic classification
- Severity labeling
- Structured CSV export
- Lightweight rule-based fallback support

---

# Supported Input Formats

The module supports:

- `.csv`
- `.txt`
- `.json`
- `.jsonl`

If no input file is specified, the pipeline automatically selects the first supported file found in:

```text
data/
```

Example dataset:

```text
data/CrimeReport (1).txt
```

---

# Output Files

Current integration output:

```text
output/text_output.csv
```

Additional generated outputs:

```text
output/text_analyst_final.csv
output/text_analyst_extended_output.csv
```

---

# Submission Output Columns

The primary submission CSV contains:

- `Text_ID`
- `Crime_Type`
- `Location_Entity`
- `Sentiment`
- `Topic`
- `Severity_Label`

---

# Extended Output Features

The extended output additionally includes:

- cleaned text
- extracted entities
- organization names
- dates
- metadata
- confidence scores

---

# Processing Pipeline

For each text record, the system performs:

1. Text preprocessing
2. Named entity extraction
3. Crime-type classification
4. Sentiment analysis
5. Topic prediction
6. Severity labeling
7. Structured CSV export

---

# Setup

From the `text/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# Optional spaCy Model Setup

For improved named entity recognition:

```bash
python -m spacy download en_core_web_sm
```

---

# Optional Transformer Support

For transformer-based sentiment and topic classification:

```bash
pip install -r requirements-full.txt
```

---

# Fallback Logic

If optional dependencies are unavailable:

- spaCy falls back to regex-based extraction
- transformers fall back to lightweight rule-based logic

This enables the module to run in lightweight environments without requiring large models.

---

# Running the Pipeline

## Lightweight Rule-Based Run

```bash
python3 src/text_analysis.py --input "data/CrimeReport (1).txt" --output output/text_analyst_final.csv --extended-output output/text_analyst_extended_output.csv --no-transformers
```

---

## Default Run

```bash
python3 src/text_analysis.py
```

---

# Optional Arguments

- `--input` → specify custom input file
- `--output` → specify submission CSV path
- `--extended-output` → specify detailed CSV path
- `--no-transformers` → disable transformer-based inference

---

# Example Workflow

Input:
- crime-related textual reports

Processing:
- NLP preprocessing
- entity extraction
- sentiment analysis
- topic classification
- severity estimation

Output:
- structured incident intelligence CSV

---

# Skills Demonstrated

- NLP pipeline development
- Named Entity Recognition (NER)
- Sentiment analysis
- Topic classification
- Rule-based AI workflows
- Structured information extraction
- Transformer-based NLP experimentation
- Multimodal AI integration

---

# Example Outputs

Generated outputs include:

- crime classifications
- location entities
- sentiment labels
- topic predictions
- severity scores
- structured metadata

---

# Limitations

- Rule-based fallback logic may reduce classification accuracy
- Text quality and formatting strongly affect extraction reliability
- Transformer-free mode prioritizes lightweight execution over advanced inference
- Ambiguous incident descriptions may impact severity labeling

---

# Integration

The generated output integrates into the larger multimodal incident analysis system through:

```text
integration/data/incident_map.csv
```

using:

```text
Text_ID
```

as the linking identifier.

---

# Notes

- Large raw datasets were excluded from GitHub for repository optimization.
- Transformer-based workflows require additional dependencies and compute resources.
- This module was developed as part of a larger multimodal AI incident analysis platform.
