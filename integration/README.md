# Integration Module

Unified multimodal incident intelligence pipeline that combines structured outputs from audio, document, image, text, and video analysis workflows into a centralized incident-level reporting system.

This module serves as the final aggregation layer of the Multimodal Crime Report Analyzer project and generates consolidated incident intelligence reports through cross-modal mapping and Streamlit-based visualization.

---

# Tech Stack

`Python` `Pandas` `Streamlit` `Data Integration` `CSV Processing` `Multimodal AI` `Analytics Dashboard`

---

# Module Overview

The integration pipeline merges outputs generated from:

- Audio analysis
- Document analysis
- Image analysis
- Text analysis
- Video analysis

The system links modality outputs through a shared incident mapping workflow and generates:

- Unified incident summaries
- Severity categorization
- Cross-modal evidence aggregation
- Interactive dashboard visualization

---

# Input Sources

The integration pipeline combines structured outputs from:

```text
audio/output/audio_output.csv
pdf/output/incident_extract.csv
images/output/image_analyst_output.csv
video/output/video_event_log.csv
text/output/text_output.csv
```

---

# Main Deliverables

The module produces:

- `data/incident_map.csv`
- `output/final_integrated_incident_report.csv`
- `app.py` Streamlit dashboard

---

# Integration Workflow

The integration script supports two workflows.

---

## 1. Mapped Integration Mode

This is the primary assignment-ready workflow.

The system uses:

```text
data/incident_map.csv
```

to map modality-specific IDs into a shared:

```text
Incident_ID
```

This enables cross-modal incident aggregation across:
- audio
- image
- text
- video
- document evidence

---

## 2. Prototype Mode

Prototype mode creates a simplified synthetic merged output using available modality files.

This workflow is intended only for experimentation and debugging.

---

# Folder Structure

```text
integration/
├── app.py
├── README.md
├── data/
├── output/
├── requirements.txt
└── src/
```

---

# Main Files

- `app.py` → Streamlit dashboard interface
- `src/integrate_reports.py` → integration pipeline
- `data/incident_map.csv` → cross-modal mapping file
- `output/final_integrated_incident_report.csv` → final structured dataset

---

# Processing Pipeline

The integration workflow performs:

1. Loading modality CSV outputs
2. Normalizing structured fields
3. Aggregating video clip summaries
4. Cross-modal record mapping
5. Incident-level merging
6. Severity inference
7. Final dataset generation
8. Dashboard visualization

---

# Setup

From the `integration/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# Running the Integration Pipeline

From repository root:

```bash
python3 integration/src/integrate_reports.py
```

---

From the `integration/` folder:

```bash
python3 src/integrate_reports.py
```

---

# Generated Output

Default generated dataset:

```text
output/final_integrated_incident_report.csv
```

---

# Typical Workflow

1. Generate outputs from each modality
2. Update `data/incident_map.csv`
3. Run integration pipeline
4. Launch dashboard with Streamlit
5. Explore unified incident intelligence outputs

---

# Streamlit Dashboard

Launch dashboard:

```bash
streamlit run app.py
```

---

# Dashboard Features

The dashboard includes:

- Incident-level filtering
- Severity filtering
- Source-based filtering
- Audio-event filtering
- Keyword search
- Incident statistics
- Severity visualizations
- Source distribution charts
- Interactive incident table
- Detailed grouped incident views

---

# Incident Mapping

The default integration workflow expects:

```text
data/incident_map.csv
```

with:

- `Incident_ID`
- `Call_ID`
- `Report_ID`
- `Image_ID`
- `Clip_ID`
- `Text_ID`

This mapping enables:
- cross-modal linking
- evidence aggregation
- incident-level intelligence generation

---

# Example Custom Integration Run

```bash
python3 src/integrate_reports.py --incident-map data/incident_map.csv
```

---

# Example Prototype Run

```bash
python3 src/integrate_reports.py --prototype
```

---

# Skills Demonstrated

- Multimodal data integration
- Structured incident intelligence generation
- Cross-modal entity mapping
- Data pipeline orchestration
- Dashboard development
- Incident aggregation workflows
- Streamlit analytics applications
- End-to-end AI system integration

---

# Example Output

Final integrated dataset:

```text
integration/output/final_integrated_incident_report.csv
```

The generated report combines:
- audio insights
- OCR outputs
- object detections
- NLP classifications
- video event summaries

into a unified incident-level intelligence view.

---

# Limitations

- Current incident linking relies on manual mapping
- Cross-modal matching is rule-based rather than learned
- Missing modality outputs may reduce incident completeness
- Prototype mode is not intended for production workflows

---

# Future Improvements

- Automated multimodal entity matching
- Real-time incident ingestion
- LLM-based incident summarization
- Cloud deployment and API integration
- Advanced analytics and investigation dashboards
- Cross-modal retrieval and search workflows

---

# Notes

- Large raw modality outputs were excluded from GitHub for repository optimization.
- Video data is aggregated into clip-level summaries before merging.
- The module was designed as the central orchestration layer of the multimodal incident analysis system.
