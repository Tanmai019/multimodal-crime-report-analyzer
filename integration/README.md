# Integration Module

This module combines the current structured outputs from:

- `audio/output/audio_output.csv`
- `pdf/output/incident_extract.csv`
- `images/output/image_analyst_output.csv`
- `video/output/video_event_log.csv`
- `text/output/text_output.csv`

Audio is integrated through the same incident map workflow used by the other
modalities.

The main deliverables in this folder are:

- `data/incident_map.csv` - manual cross-modal mapping file
- `output/final_integrated_incident_report.csv` - final merged structured dataset
- `app.py` - Streamlit dashboard and query interface

## What The Script Does

The integration script supports two workflows:

1. Mapped mode
   Uses a manual incident map CSV so every output row is tied to a real
   `Incident_ID`. This is the default and is the correct assignment-ready flow.

2. Prototype mode
   Creates one synthetic summary row from all five CSVs. Keep this only for
   rough experimentation, not for final submission.

## Folder Layout

- `app.py` - Streamlit dashboard for filtering and presenting incidents
- `src/integrate_reports.py` - main integration script
- `data/incident_map.csv` - starter mapping file for assignment-style integration
- `output/` - generated final CSV files

## Setup

From the `integration/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Integration

From the repository root:

```bash
python3 integration/src/integrate_reports.py
```

Or from the `integration/` folder:

```bash
python3 src/integrate_reports.py
```

Default output:

- `output/final_integrated_incident_report.csv`

Typical workflow:

1. Regenerate or collect each modality CSV.
2. Update `data/incident_map.csv` with the rows you want to treat as the same incident.
3. Run `python3 src/integrate_reports.py`.
4. Open the dashboard with `streamlit run app.py`.

## Run Dashboard

From the `integration/` folder:

```bash
streamlit run app.py
```

The dashboard includes:

- sidebar filters for incident, source, severity, text crime type, and image scene type
- audio event filtering
- a keyword search across key incident fields
- metric cards for incident counts and severity totals
- simple severity and source charts
- a filtered incident table
- a detail panel with grouped modality data for the selected incident

## Incident Map

The default workflow expects `data/incident_map.csv` with:

- `Incident_ID`
- `Call_ID`
- `Report_ID`
- `Image_ID`
- `Clip_ID`
- `Text_ID`

Default starter map included:

- `data/incident_map.csv`

You can edit it as your team finalizes which rows belong to the same incident.

Rows do not need to contain all modalities. Leave unmatched fields blank when an
incident only has audio, document, image, video, or text evidence available.

Example run with a custom map:

```bash
python3 src/integrate_reports.py --incident-map data/incident_map.csv
```

## Notes

- Video is first aggregated from frame-level rows to clip-level summaries before
  merging.
- The default severity label is inferred from combined signals across the mapped
  modalities for each incident.
- If you really want the old single-row summary, run:

```bash
python3 src/integrate_reports.py --prototype
```
