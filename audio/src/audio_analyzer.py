"""Audio analysis pipeline for the assignment's emergency-call modality.

The code leans on strong defaults and conservative fallbacks so the CSV stays
usable even when short clips or noisy transcripts make extraction difficult.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import spacy
import torch
import whisper
from transformers import pipeline as hf_pipeline

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_CSV = os.path.join("output", "audio_output.csv")
DEFAULT_WHISPER_MODEL = "base"

EVENT_KEYWORDS: Dict[str, List[str]] = {
    "Building fire / trapped persons": [
        "fire", "smoke", "burning", "flames", "blaze", "trapped"
    ],
    "Road accident": [
        "accident", "crash", "collision", "collided", "vehicle", "car", "highway", "hit"
    ],
    "Robbery / theft": [
        "robbery", "robbed", "stole", "stolen", "theft", "bank", "burglary"
    ],
    "Assault / violence": [
        "attack", "assault", "fight", "knife", "stabbing", "beating", "bleeding"
    ],
    "Public disturbance": [
        "disturbance", "noise complaint", "loud", "party", "yelling", "riot"
    ],
    "Shooting": [
        "gun", "shoot", "shooting", "gunshot", "shot", "fired"
    ],
    "Medical emergency": [
        "ambulance", "medical", "unconscious", "heart", "breathing", "choking", "overdose", "seizure"
    ],
    "Domestic disturbance": [
        "domestic", "husband", "wife", "boyfriend", "girlfriend", "abuse"
    ],
    "Missing / kidnapped person": [
        "missing", "kidnap", "kidnapped", "abducted", "taken"
    ],
}

URGENCY_KEYWORDS = [
    "help", "hurry", "please", "emergency", "immediately", "right away",
    "trapped", "dying", "bleeding", "children", "gun", "weapon",
    "fire", "unconscious", "critical", "dead", "shot", "not breathing",
    "can't breathe", "cannot breathe", "send help", "ambulance"
]

ADDRESS_PATTERN = re.compile(
    r"\b\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|court|ct)\b",
    flags=re.IGNORECASE,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audio Analyst: transcribe 911 audio calls and generate structured CSV output."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Folder containing .wav files and optional 911_metadata.csv (default: ./data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path (default: ./output/audio_output.csv)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Process only the first N audio files for quick testing",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics after processing",
    )
    return parser


def resolve_path(script_dir: Path, user_path: str) -> Path:
    path = Path(user_path)
    if path.is_absolute():
        return path
    return (script_dir / path).resolve()


def find_audio_files(data_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(data_dir.rglob(f"*{ext}"))
    return sorted([p for p in files if p.is_file()])


def load_metadata(data_dir: Path) -> Optional[pd.DataFrame]:
    matches = list(data_dir.rglob("911_metadata.csv"))
    if not matches:
        print("[INFO] No 911_metadata.csv found. Continuing with audio only.")
        return None

    metadata_path = matches[0]
    df = pd.read_csv(metadata_path)
    print(f"[INFO] Found metadata: {metadata_path} ({len(df)} rows)")

    filename_col = None
    for col in df.columns:
        if col.lower() == "filename":
            filename_col = col
            break

    if filename_col is None:
        print("[WARN] Metadata found but no 'filename' column exists. Ignoring metadata.")
        return None

    df = df.copy()
    # Matching on basename keeps the lookup resilient even if the metadata file
    # stores longer paths than the local workspace does.
    df["__basename__"] = df[filename_col].astype(str).apply(lambda x: os.path.basename(x).strip())
    return df


class AudioAnalyzer:
    def __init__(
        self,
        data_dir: Path,
        output_csv: Path,
        max_files: Optional[int],
        whisper_model_size: str,
    ):
        self.data_dir = data_dir
        self.output_csv = output_csv
        self.max_files = max_files
        self.whisper_model_size = whisper_model_size

        self.whisper_model = None
        self.nlp = None
        self.sentiment_pipe = None
        self.metadata_df: Optional[pd.DataFrame] = None

    def load_models(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Device: {device.upper()}")

        # Load everything up front so the per-file loop stays simple and we pay
        # the model startup cost only once.
        print(f"[INFO] Loading Whisper model: {self.whisper_model_size}")
        self.whisper_model = whisper.load_model(self.whisper_model_size, device=device)
        print("[INFO] Whisper loaded")

        print("[INFO] Loading spaCy model: en_core_web_sm")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. Run: python -m spacy download en_core_web_sm"
            ) from exc
        print("[INFO] spaCy loaded")

        print("[INFO] Loading Hugging Face sentiment model")
        hf_device = 0 if torch.cuda.is_available() else -1
        self.sentiment_pipe = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=hf_device,
        )
        print("[INFO] Sentiment model loaded")

    def get_metadata_row(self, audio_path: Path) -> Optional[pd.Series]:
        if self.metadata_df is None:
            return None
        matches = self.metadata_df[self.metadata_df["__basename__"] == audio_path.name]
        if matches.empty:
            return None
        return matches.iloc[0]

    def transcribe_audio(self, audio_path: Path) -> str:
        try:
            result = self.whisper_model.transcribe(
                str(audio_path),
                language="en",
                fp16=torch.cuda.is_available(),
            )
            text = result.get("text", "").strip()
            return text if text else "ERROR"
        except Exception as exc:
            print(f"[WARN] Transcription failed for {audio_path.name}: {exc}")
            return "ERROR"

    def classify_event(self, text: str, meta_title: str = "", meta_description: str = "") -> str:
        combined = " ".join([text or "", meta_title or "", meta_description or ""]).lower()
        best_event = "Unknown"
        best_score = 0

        # A simple keyword vote is easier to reason about for class demos than a
        # heavier classifier, and the metadata gives weak transcripts a boost.
        for event_type, keywords in EVENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > best_score:
                best_score = score
                best_event = event_type

        return best_event

    def extract_location(self, text: str, meta_state: str = "") -> str:
        doc = self.nlp(text[:5000])
        locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
        locations.extend(ADDRESS_PATTERN.findall(text))

        # If metadata knows the state, surface it first so the output is not
        # empty when the transcript itself is too short to name a place.
        if meta_state and str(meta_state).strip() and str(meta_state).lower() != "nan":
            locations.insert(0, str(meta_state).strip())

        unique_locations = list(dict.fromkeys([loc.strip() for loc in locations if str(loc).strip()]))
        return ", ".join(unique_locations) if unique_locations else "Unknown"

    def analyze_sentiment_and_urgency(
        self,
        text: str,
        meta_deaths: str = "",
        meta_potential_death: str = "",
    ) -> Tuple[str, Optional[float]]:
        if not text or text == "ERROR":
            return "Unknown", None

        try:
            result = self.sentiment_pipe(text[:512])[0]
            label = str(result.get("label", "")).upper()
            score = float(result.get("score", 0.5))
        except Exception as exc:
            print(f"[WARN] Sentiment analysis failed: {exc}")
            return "Unknown", None

        # Urgency is a blend of emotional tone, emergency keywords, and any
        # high-risk metadata fields that came with the call.
        keyword_hits = sum(1 for kw in URGENCY_KEYWORDS if kw in text.lower())
        keyword_score = min(keyword_hits / 5.0, 1.0)

        if label == "NEGATIVE":
            sentiment_label = "Distressed"
            model_urgency = score
        elif label == "POSITIVE":
            sentiment_label = "Calm"
            model_urgency = 1.0 - score
        else:
            sentiment_label = "Unknown"
            model_urgency = 0.5

        urgency = (0.6 * keyword_score) + (0.4 * model_urgency)

        if str(meta_deaths).strip() not in {"", "0", "0.0", "nan", "None"}:
            urgency = min(urgency + 0.20, 1.0)
        if str(meta_potential_death).strip() not in {"", "0", "0.0", "nan", "None"}:
            urgency = min(urgency + 0.10, 1.0)

        urgency = round(min(urgency, 1.0), 2)

        if urgency >= 0.50:
            sentiment_label = "Distressed"
        elif sentiment_label == "Unknown":
            sentiment_label = "Calm"

        return sentiment_label, urgency

    def process_one(self, call_id: str, audio_path: Path) -> Dict[str, object]:
        meta = self.get_metadata_row(audio_path)
        meta_title = str(meta.get("title", "")) if meta is not None else ""
        meta_description = str(meta.get("description", "")) if meta is not None else ""
        meta_state = str(meta.get("state", "")) if meta is not None else ""
        meta_deaths = str(meta.get("deaths", "")) if meta is not None else ""
        meta_potential_death = str(meta.get("potential_death", "")) if meta is not None else ""

        transcript = self.transcribe_audio(audio_path)

        # Very short transcripts are usually not trustworthy enough to assign a
        # specific event label, so we keep them as Unknown instead of guessing.
        if not transcript or transcript == "ERROR" or len(transcript.split()) < 3:
            event = "Unknown"
        else:
            event = self.classify_event(
                transcript,
                meta_title=meta_title,
                meta_description=meta_description,
            )

        location = self.extract_location(transcript, meta_state=meta_state)
        sentiment, urgency = self.analyze_sentiment_and_urgency(
            transcript,
            meta_deaths=meta_deaths,
            meta_potential_death=meta_potential_death,
        )

        return {
            "Call_ID": call_id,
            "Transcript": transcript,
            "Extracted_Event": event,
            "Location": location,
            "Sentiment": sentiment,
            "Urgency_Score": urgency,
        }

    def run(self) -> pd.DataFrame:
        if not self.data_dir.exists() or not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data folder not found: {self.data_dir}")

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_df = load_metadata(self.data_dir)

        audio_files = find_audio_files(self.data_dir)
        if not audio_files:
            raise FileNotFoundError(f"No supported audio files found in: {self.data_dir}")

        if self.max_files is not None:
            audio_files = audio_files[: self.max_files]

        print("=" * 72)
        print("AUDIO ANALYST PIPELINE")
        print("=" * 72)
        print(f"[INFO] Data folder : {self.data_dir}")
        print(f"[INFO] Audio files : {len(audio_files)}")
        if self.max_files is not None:
            print(f"[INFO] Max files   : {self.max_files}")
        print(f"[INFO] Output CSV  : {self.output_csv}")

        self.load_models()

        records: List[Dict[str, object]] = []
        for idx, audio_path in enumerate(audio_files, start=1):
            # Stable synthetic IDs make the integration step predictable even if
            # the source filenames are messy.
            call_id = f"C{idx:03d}"
            print(f"[PROCESS] {call_id} | {audio_path.name}")
            records.append(self.process_one(call_id, audio_path))

        df_out = pd.DataFrame(
            records,
            columns=[
                "Call_ID",
                "Transcript",
                "Extracted_Event",
                "Location",
                "Sentiment",
                "Urgency_Score",
            ],
        )
        df_out.to_csv(self.output_csv, index=False)

        print("=" * 72)
        print(f"[INFO] Done. Processed {len(df_out)} file(s)")
        print(f"[INFO] Output saved to: {self.output_csv}")
        print("=" * 72)
        return df_out


def print_summary(df: pd.DataFrame) -> None:
    print("\nEvent distribution")
    print(df["Extracted_Event"].value_counts(dropna=False).to_string())

    print("\nSentiment distribution")
    print(df["Sentiment"].value_counts(dropna=False).to_string())

    print("\nUrgency statistics")
    print(pd.to_numeric(df["Urgency_Score"], errors="coerce").describe().round(3).to_string())


def main() -> None:
    args = build_arg_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    data_dir = resolve_path(script_dir, args.data)
    output_csv = resolve_path(script_dir, args.output)

    analyzer = AudioAnalyzer(
        data_dir=data_dir,
        output_csv=output_csv,
        max_files=args.max,
        whisper_model_size=args.whisper_model,
    )
    df_result = analyzer.run()

    if args.summary:
        print_summary(df_result)


if __name__ == "__main__":
    main()
