# Audio Analysis Module

This module processes short emergency-call audio clips and writes a structured CSV with transcript, event label, location, sentiment, and urgency score.

## Folder Layout

- `src/` contains the audio analysis script
- `data/` contains input audio files and optional metadata
- `output/` contains generated CSV output

## Supported Input

The script accepts these audio formats:

- `.wav`
- `.mp3`
- `.m4a`
- `.flac`
- `.ogg`

Optional metadata file:

- `data/911_metadata.csv`

If the metadata CSV is present, the script uses it to improve event and location extraction.

## Output

The pipeline writes:

- `output/audio_output.csv`

Current output columns:

- `Call_ID`
- `Transcript`
- `Extracted_Event`
- `Location`
- `Sentiment`
- `Urgency_Score`

## Setup

From the `audio/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Install `ffmpeg` too, because Whisper needs it to decode audio:

```bash
brew install ffmpeg
```

On first run, Whisper and the Hugging Face sentiment model may download model weights automatically.

## Run

From the `audio/` folder:

Quick test on 10 files:

```bash
python3 src/audio_analyzer.py --data data --output output/audio_output.csv --max 10 --summary
```

Full dataset:

```bash
python3 src/audio_analyzer.py --data data --output output/audio_output.csv
```

Faster run with a smaller Whisper model:

```bash
python3 src/audio_analyzer.py --data data --output output/audio_output.csv --max 10 --whisper_model tiny --summary
```

Optional arguments:

- `--max N` to process only the first `N` files
- `--whisper_model tiny|base|small|medium|large` to choose model size
- `--summary` to print event and sentiment counts in the terminal
- `--output path/to/audio_output.csv` to change the output file

## What It Does

For each input audio clip, the pipeline:

- transcribes speech with Whisper
- extracts location hints using spaCy and regex rules
- classifies the event with keyword rules
- predicts calm vs distressed sentiment
- computes a simple urgency score

## Notes

- Audio clips in this dataset are very short, so transcripts can be incomplete.
- If transcription fails, the row may contain `ERROR` in the `Transcript` column.
- Metadata helps improve results, especially for location and emergency context.
- The final integration pipeline links this output by `Call_ID` through
  `integration/data/incident_map.csv`.
