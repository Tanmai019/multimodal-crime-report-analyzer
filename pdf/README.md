# Document Analysis Module

This module extracts structured incident information from PDF reports and writes the result as a one-row CSV.

## Folder Layout

- `src/` contains the PDF parsing script
- `data/` contains input PDF files
- `output/` contains generated CSV output

## What It Extracts

The current script outputs these columns:

- `Report_ID`
- `Incident_Type`
- `Date`
- `Location`
- `Officer`
- `Summary`

The pipeline uses:

- PyMuPDF for native PDF text extraction
- Tesseract OCR as a fallback for scanned PDFs
- spaCy NER plus rule-based heuristics for field extraction

## Setup

From the `pdf/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If your PDF is scanned or image-based, install Tesseract OCR too:

```bash
brew install tesseract
```

If Tesseract is installed but not on your shell `PATH`, set:

```bash
export TESSERACT_CMD="/opt/homebrew/bin/tesseract"
```

## Input

Place PDFs in `data/` or pass a full path on the command line.

Sample file already in this folder:

- `data/LESO2.pdf`

## Run

From the `pdf/` folder:

```bash
python3 src/document_analysis.py data/LESO2.pdf -o output/incident_extract.csv -v
```

Generic form:

```bash
python3 src/document_analysis.py path/to/report.pdf -o output/incident_extract.csv
```

Optional arguments:

- `--report-id "custom_id"` to override the default report ID
- `--spacy-model en_core_web_sm` to choose a spaCy model
- `-v` or `--verbose` for debug logs

If you omit the PDF path, the script prompts you for it interactively.

## Output

The script:

- prints the extracted row in the terminal
- saves the CSV to the file passed with `-o`

Recommended output path:

- `output/incident_extract.csv`

## Notes

- The script first tries native text extraction and falls back to OCR when the PDF appears text-sparse.
- When a field is not reliable, the script returns `Not Found`.
- This module is currently designed for one PDF at a time.
- The final integration pipeline links this output by `Report_ID` through
  `integration/data/incident_map.csv`.
