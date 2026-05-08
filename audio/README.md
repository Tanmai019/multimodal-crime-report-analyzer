# Audio Analysis Module

AI-powered emergency audio analysis pipeline for extracting structured incident intelligence from short emergency-call recordings.

This module processes emergency-style audio clips using speech transcription, NLP-based extraction, sentiment analysis, and rule-based classification to generate structured incident-level outputs for downstream multimodal integration.

---

# Tech Stack

`Python` `Whisper` `spaCy` `Transformers` `NLTK` `Speech Recognition` `Pandas` `Audio Processing`

---

# Module Overview

The pipeline analyzes short emergency-call audio recordings and extracts:

- Speech transcripts
- Incident event labels
- Location references
- Sentiment classification
- Urgency scoring

The processed outputs are converted into structured CSV format for integration into the larger multimodal incident analysis system.

---

# Features

- Automatic speech transcription using Whisper
- Named entity extraction for location identification
- Rule-based emergency event classification
- Sentiment analysis for caller distress detection
- Urgency score generation
- Structured CSV output generation

---

# Supported Input Formats

The module supports:

- `.wav`
- `.mp3`
- `.m4a`
- `.flac`
- `.ogg`

Optional metadata support:

```text
data/911_metadata.csv
```

If metadata is available, it is used to improve:
- event extraction
- contextual interpretation
- location identification

---

# Output

Generated output:

```text
output/audio_output.csv
```

### Output Columns

- `Call_ID`
- `Transcript`
- `Extracted_Event`
- `Location`
- `Sentiment`
- `Urgency_Score`

---

# Processing Pipeline

For each audio clip, the system performs:

1. Speech transcription using Whisper
2. Location extraction using spaCy and regex workflows
3. Emergency-event classification using keyword rules
4. Sentiment prediction for distress detection
5. Urgency score computation
6. Structured CSV export

---

# Folder Structure

```text
audio/
├── data/
├── output/
├── src/
├── requirements.txt
└── README.md
```

---

# Setup

From the `audio/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Install `ffmpeg` for Whisper audio decoding:

```bash
brew install ffmpeg
```

On first execution, Whisper and Hugging Face models may automatically download pretrained weights.

---

# Running the Pipeline

## Quick Test

```bash
python3 src/audio_analyzer.py --data data --output output/audio_output.csv --max 10 --summary
```

---

## Full Dataset Processing

```bash
python3 src/audio_analyzer.py --data data --output output/audio_output.csv
```

---

## Faster Lightweight Inference

```bash
python3 src/audio_analyzer.py --data data --output output/audio_output.csv --max 10 --whisper_model tiny --summary
```

---

# Optional Arguments

- `--max N` → process first `N` files only
- `--whisper_model` → select Whisper model size
- `--summary` → print event and sentiment statistics
- `--output` → specify custom output path

Supported Whisper models:

- `tiny`
- `base`
- `small`
- `medium`
- `large`

---

# Example Workflow

Input:
- emergency-call audio clips

Processing:
- transcription
- NLP extraction
- sentiment analysis
- urgency estimation

Output:
- structured incident intelligence CSV

---

# Skills Demonstrated

- Speech-to-text processing
- NLP entity extraction
- Audio analytics workflows
- Sentiment analysis
- Structured data generation
- Rule-based AI systems
- Multimodal pipeline integration

---

# Limitations

- Very short audio clips may produce incomplete transcripts
- Background noise can reduce transcription quality
- Rule-based event classification may miss nuanced scenarios
- Sentiment analysis is limited by transcript quality

---

# Integration

The generated output integrates into the larger multimodal incident analysis system through:

```text
integration/data/incident_map.csv
```

using:

```text
Call_ID
```

as the primary linking identifier.

---

# Notes

- This module was designed as part of a multimodal AI incident analysis pipeline.
- GPU acceleration can significantly improve Whisper inference speed.
- Large raw audio datasets were excluded from GitHub for repository optimization.
