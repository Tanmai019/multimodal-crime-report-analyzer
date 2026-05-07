# Text Analysis Module

This module analyzes crime-related text records and produces structured CSV outputs for submission and detailed review.

## Folder Layout

- `src/` contains the text analysis script
- `data/` contains input text datasets
- `output/` contains generated CSV files

## Supported Input Formats

The script accepts:

- `.csv`
- `.txt`
- `.jsonl`
- `.json`

If `--input` is not provided, the script automatically selects the first supported file in `data/`.

Sample file already in this folder:

- `data/CrimeReport (1).txt`

## Output Files

The checked-in text output currently used by the team integration is:

- `output/text_output.csv`

The script can also write these CSV files by default:

- `output/text_analyst_final.csv`
- `output/text_analyst_extended_output.csv`

The submission CSV contains:

- `Text_ID`
- `Crime_Type`
- `Location_Entity`
- `Sentiment`
- `Topic`
- `Severity_Label`

The extended CSV also includes the cleaned text, extracted entities, organizations, dates, confidence scores, and metadata.

## Setup

From the `text/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional spaCy model install for improved NER:

```bash
python -m spacy download en_core_web_sm
```

Optional full install for transformer-based sentiment and topic classification:

```bash
pip install -r requirements-full.txt
```

Without the spaCy model, the script falls back to a blank English pipeline plus regex-based extraction.
Without transformers, it falls back to rule-based sentiment and topic logic.

## Run

From the `text/` folder:

Lightweight run with rule-based fallback:

```bash
python3 src/text_analysis.py --input "data/CrimeReport (1).txt" --output output/text_analyst_final.csv --extended-output output/text_analyst_extended_output.csv --no-transformers
```

Default run using the first supported file in `data/`:

```bash
python3 src/text_analysis.py
```

Optional arguments:

- `--input path/to/file.csv` to choose a specific input file
- `--output path/to/text_analyst_final.csv` to choose the submission output path
- `--extended-output path/to/text_analyst_extended_output.csv` to choose the detailed output path
- `--no-transformers` to skip Hugging Face models and use rule-based fallback

## What It Does

For each input record, the pipeline extracts or predicts:

- crime type
- location entity
- sentiment
- topic
- severity label
- people, organizations, and dates

The script prints a preview table in the terminal after writing the CSV files.

## Integration Note

- The final team integration currently uses `output/text_output.csv`.
- The integration pipeline links this output by `Text_ID` through
  `integration/data/incident_map.csv`.
