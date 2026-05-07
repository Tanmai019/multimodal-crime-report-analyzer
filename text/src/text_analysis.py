"""Text analysis pipeline for social posts, reports, and other written sources.

The module tries richer NLP first, then falls back to lightweight heuristics so
the assignment can still run on machines without every optional dependency.
"""

import argparse
import ast
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS

try:
    from transformers import pipeline as hf_pipeline
except Exception:  # pragma: no cover - optional dependency fallback
    hf_pipeline = None

warnings.filterwarnings("ignore")

MODULE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = MODULE_DIR / "data"
OUTPUT_DIR = MODULE_DIR / "output"
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "text_analyst_final.csv"
DEFAULT_EXTENDED_OUTPUT_FILE = OUTPUT_DIR / "text_analyst_extended_output.csv"

TEXT_COLUMN_CANDIDATES = (
    "text",
    "raw_text",
    "content",
    "description",
    "details",
    "report",
    "summary",
    "post",
    "body",
)

SOURCE_COLUMN_CANDIDATES = (
    "source",
    "platform",
    "channel",
    "author",
    "username",
    "user",
)

DATE_COLUMN_CANDIDATES = (
    "created_at",
    "timestamp",
    "date",
    "time",
    "reported_at",
)

TOPIC_LABELS = [
    "accident",
    "fire",
    "theft",
    "disturbance",
    "robbery",
    "shooting",
    "assault",
    "kidnapping",
    "drug-related incident",
    "murder",
    "fraud",
    "general crime report",
]

CRIME_KEYWORDS = {
    "Robbery": ["robbery", "robbed", "heist", "bank robbery", "stole cash"],
    "Theft": ["theft", "stolen", "steal", "burglary", "shoplifting", "larceny"],
    "Fire": ["fire", "burning", "smoke", "blaze", "explosion", "grass fire"],
    "Accident": ["accident", "crash", "collision", "wreck", "hit-and-run"],
    "Disturbance": ["disturbance", "fight", "riot", "chaos", "public disorder"],
    "Shooting": ["shooting", "shot", "gunfire", "bullet", "fired shots"],
    "Assault": ["assault", "attacked", "beating", "stabbed", "violence"],
    "Kidnapping": ["kidnap", "kidnapping", "abducted", "hostage"],
    "Murder": ["murder", "killed", "dead", "homicide", "attempt murder"],
    "Drug Crime": ["drug", "heroin", "marijuana", "narcotics", "trafficking"],
    "Fraud": ["fraud", "scam", "forgery", "identity theft"],
}

TOPIC_TO_CRIME = {
    "accident": "Accident",
    "fire": "Fire",
    "theft": "Theft",
    "disturbance": "Disturbance",
    "robbery": "Robbery",
    "shooting": "Shooting",
    "assault": "Assault",
    "kidnapping": "Kidnapping",
    "murder": "Murder",
    "drug-related incident": "Drug Crime",
    "fraud": "Fraud",
    "general crime report": "General Crime",
}

CRIME_TO_TOPIC = {value: key for key, value in TOPIC_TO_CRIME.items()}

PRIORITY_TOPICS = {
    "shooting": "Shooting",
    "murder": "Murder",
    "kidnapping": "Kidnapping",
}

HIGH_SEVERITY_TERMS = {
    "shooting",
    "gunfire",
    "murder",
    "killed",
    "dead",
    "explosion",
    "arson",
    "kidnapping",
    "hostage",
    "attempt murder",
    "injured",
    "homicide",
}

MEDIUM_SEVERITY_TERMS = {
    "robbery",
    "theft",
    "burglary",
    "assault",
    "fire",
    "drug",
    "disturbance",
    "crash",
    "accident",
    "violence",
    "fight",
}

NEGATIVE_SENTIMENT_TERMS = HIGH_SEVERITY_TERMS | MEDIUM_SEVERITY_TERMS | {
    "arrest",
    "suspect",
    "victim",
    "threat",
    "emergency",
    "crime",
}

POSITIVE_SENTIMENT_TERMS = {
    "safe",
    "rescued",
    "resolved",
    "contained",
    "recovered",
    "stable",
    "cleared",
}

BAD_LOCATION_TERMS = {
    "Police",
    "Crime",
    "Robbery",
    "Murder",
    "Shooting",
    "Disturbance",
    "Fire",
    "Accident",
    "Assault",
}


@dataclass
class ModelBundle:
    stop_words: set
    nlp: Any
    has_ner: bool
    sentiment_pipeline: Any = None
    topic_pipeline: Any = None


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        cleaned = normalize_whitespace(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def setup_models(use_transformers: bool = True) -> ModelBundle:
    # Model setup degrades gracefully: the pipeline should still produce a useful
    # CSV even if spaCy models or transformer weights are unavailable.
    try:
        nlp = spacy.load("en_core_web_sm")
        has_ner = True
    except OSError:
        warnings.warn(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Falling back to a blank English pipeline with regex-based entities."
        )
        nlp = spacy.blank("en")
        has_ner = False

    sentiment_pipeline = None
    topic_pipeline = None
    if use_transformers and hf_pipeline is not None:
        try:
            sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            )
            topic_pipeline = hf_pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
            )
        except Exception as exc:
            warnings.warn(
                "Transformer models could not be loaded. "
                "Using rule-based fallback for sentiment/topic. "
                f"Reason: {exc}"
            )
    elif use_transformers:
        warnings.warn(
            "transformers is not available. Using rule-based fallback "
            "for sentiment/topic."
        )

    return ModelBundle(
        stop_words=set(SPACY_STOP_WORDS),
        nlp=nlp,
        has_ner=has_ner,
        sentiment_pipeline=sentiment_pipeline,
        topic_pipeline=topic_pipeline,
    )


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"RT\s+@\w+:", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"&amp;", "and", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return normalize_whitespace(text)


def preprocess_for_tokens(text: str, stop_words: set) -> str:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    filtered = [token for token in tokens if token not in stop_words]
    return " ".join(filtered)


def parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None

    # Many classroom datasets are "almost JSON" rather than clean JSONL, so we
    # try strict JSON first and then a safer Python-literal fallback.
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    try:
        obj = ast.literal_eval(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def build_record(
    record_no: int,
    raw_text: str,
    source: str,
    created_at: str = "",
) -> Dict[str, Any]:
    return {
        "Record_No": record_no,
        "Source": source or "unknown",
        "Raw_Text": raw_text,
        "Created_At": created_at,
    }


def coerce_structured_record(
    obj: Dict[str, Any],
    record_no: int,
    default_source: str,
) -> Optional[Dict[str, Any]]:
    raw_text = None
    for key in ("text", "report_text", "content", "message", "description", "body"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            raw_text = value
            break

    if raw_text is None:
        return None

    source = default_source
    if isinstance(obj.get("user"), dict):
        source = (
            obj["user"].get("screen_name")
            or obj["user"].get("name")
            or default_source
        )
    elif isinstance(obj.get("source"), str) and obj["source"].strip():
        source = obj["source"].strip()

    created_at = ""
    for key in ("created_at", "timestamp", "date"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            created_at = value.strip()
            break

    return build_record(record_no, raw_text.strip(), source, created_at)


def extract_structured_records(raw_text: str, default_source: str) -> List[Dict[str, Any]]:
    stripped = raw_text.lstrip()
    if not stripped:
        return []

    records: List[Dict[str, Any]] = []

    # If the file already looks structured, preserve that structure before
    # falling back to paragraph-style record splitting.
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                parsed = parsed.get("records", [parsed])
            if isinstance(parsed, list):
                for idx, obj in enumerate(parsed, start=1):
                    if not isinstance(obj, dict):
                        continue
                    record = coerce_structured_record(obj, idx, default_source)
                    if record:
                        records.append(record)
                if records:
                    return records
        except Exception:
            pass

    line_records: List[Dict[str, Any]] = []
    for idx, line in enumerate(raw_text.splitlines(), start=1):
        obj = parse_json_line(line)
        if not obj:
            continue
        record = coerce_structured_record(obj, idx, default_source)
        if record:
            line_records.append(record)

    return line_records


def extract_plain_text_records(raw_text: str, default_source: str) -> List[Dict[str, Any]]:
    paragraphs = [
        normalize_whitespace(block)
        for block in re.split(r"\n\s*\n+", raw_text)
        if normalize_whitespace(block)
    ]
    if not paragraphs:
        return []

    if len(paragraphs) == 1:
        return [build_record(1, paragraphs[0], default_source)]

    return [
        build_record(idx, paragraph, default_source)
        for idx, paragraph in enumerate(paragraphs, start=1)
        if len(paragraph) > 20
    ]


def pick_best_text_column(df: pd.DataFrame) -> Optional[str]:
    lowered = {column.lower(): column for column in df.columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]

    object_columns = [
        column
        for column in df.columns
        if df[column].dtype == "object"
    ]
    if not object_columns:
        return None

    # When column names are messy, use the longest object column as a practical
    # guess for "the real text" field.
    best_column = None
    best_score = -1.0
    for column in object_columns:
        sample = df[column].dropna().astype(str).head(25)
        if sample.empty:
            continue
        avg_length = sample.str.len().mean()
        if avg_length > best_score:
            best_score = avg_length
            best_column = column
    return best_column


def pick_optional_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    lowered = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def extract_csv_records(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if df.empty:
        return pd.DataFrame()

    text_column = pick_best_text_column(df)
    if text_column is None:
        return pd.DataFrame()

    source_column = pick_optional_column(df, SOURCE_COLUMN_CANDIDATES)
    date_column = pick_optional_column(df, DATE_COLUMN_CANDIDATES)

    records: List[Dict[str, Any]] = []
    default_source = filepath.stem
    for idx, row in df.iterrows():
        raw_value = row.get(text_column)
        if pd.isna(raw_value):
            continue

        raw_text = str(raw_value).strip()
        if not raw_text:
            continue

        source = default_source
        if source_column is not None:
            source_value = row.get(source_column)
            if not pd.isna(source_value):
                source = str(source_value).strip() or default_source

        created_at = ""
        if date_column is not None:
            date_value = row.get(date_column)
            if not pd.isna(date_value):
                created_at = str(date_value).strip()

        records.append(build_record(idx + 1, raw_text, source, created_at))

    return pd.DataFrame(records)


def extract_raw_records(filepath: Path) -> pd.DataFrame:
    if filepath.suffix.lower() == ".csv":
        return extract_csv_records(filepath)

    raw_text = filepath.read_text(encoding="utf-8", errors="ignore")
    default_source = filepath.stem

    records = extract_structured_records(raw_text, default_source)
    if not records:
        records = extract_plain_text_records(raw_text, default_source)

    return pd.DataFrame(records)


def clean_location_candidate(text: str) -> str:
    text = normalize_whitespace(text.strip(" ,.-:;"))
    return re.sub(r"^(the)\s+", "", text, flags=re.IGNORECASE)


def fallback_people(text: str) -> List[str]:
    matches = re.findall(
        r"\b(?:Officer|Ofc\.|Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        text,
    )
    return unique_preserve_order(matches)


def fallback_locations(text: str) -> List[str]:
    candidates: List[str] = []
    patterns = [
        r"\b(?:in|at|near)\s+([A-Z][a-zA-Z]+(?:[\s,/.-][A-Z0-9][a-zA-Z0-9./-]*){0,4})",
        r"\b(\d{2,5}\s+[A-Z][A-Za-z0-9.\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr))\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            candidate = clean_location_candidate(match)
            candidate = re.sub(r"/+", ", ", candidate)
            if 1 <= len(candidate.split()) <= 6 and len(candidate) > 2:
                candidates.append(candidate)
    return unique_preserve_order(candidates)


def fallback_organizations(text: str) -> List[str]:
    matches = re.findall(
        r"\b([A-Z][A-Za-z&.\s]+(?:Police Department|Sheriff'?s Office|Department|University|Hospital))\b",
        text,
    )
    return unique_preserve_order(matches)


def fallback_dates(text: str) -> List[str]:
    patterns = [
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
    ]
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return unique_preserve_order(matches)


def extract_entities(text: str, models: ModelBundle) -> Dict[str, str]:
    people: List[str] = []
    locations: List[str] = []
    organizations: List[str] = []
    dates: List[str] = []

    doc = models.nlp(text)
    if models.has_ner:
        people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        locations = [
            ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC", "FAC"}
        ]
        organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        dates = [ent.text for ent in doc.ents if ent.label_ in {"DATE", "TIME"}]

    # Regex fallbacks are intentionally simple, but they keep the pipeline usable
    # when NER is unavailable or the text is too noisy for spaCy to help.
    if not people:
        people = fallback_people(text)
    if not locations:
        locations = fallback_locations(text)
    if not organizations:
        organizations = fallback_organizations(text)
    if not dates:
        dates = fallback_dates(text)

    people = unique_preserve_order(people)
    locations = unique_preserve_order(locations)
    organizations = unique_preserve_order(organizations)
    dates = unique_preserve_order(dates)

    locations = [loc for loc in locations if loc not in BAD_LOCATION_TERMS]

    entity_parts = []
    if people:
        entity_parts.append("People: " + ", ".join(people))
    if locations:
        entity_parts.append("Locations: " + ", ".join(locations))
    if organizations:
        entity_parts.append("Organizations: " + ", ".join(organizations))
    if dates:
        entity_parts.append("Dates: " + ", ".join(dates))

    return {
        "People": ", ".join(people),
        "Location_Entity": ", ".join(locations),
        "Organizations": ", ".join(organizations),
        "Dates": ", ".join(dates),
        "Entities": " | ".join(entity_parts),
    }


def keyword_match_counts(text: str) -> Dict[str, int]:
    text_lower = text.lower()
    return {
        label: sum(1 for keyword in keywords if keyword in text_lower)
        for label, keywords in CRIME_KEYWORDS.items()
    }


def infer_crime_type(text: str) -> str:
    counts = keyword_match_counts(text)
    best_label = "General Crime"
    best_score = 0
    for label, score in counts.items():
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def heuristic_sentiment(text: str) -> Tuple[str, float]:
    text_lower = text.lower()
    negative_hits = sum(text_lower.count(term) for term in NEGATIVE_SENTIMENT_TERMS)
    positive_hits = sum(text_lower.count(term) for term in POSITIVE_SENTIMENT_TERMS)

    if negative_hits > positive_hits and negative_hits > 0:
        return "Negative", min(0.55 + negative_hits * 0.06, 0.99)
    if positive_hits > negative_hits and positive_hits > 0:
        return "Positive", min(0.55 + positive_hits * 0.06, 0.95)
    return "Neutral", 0.5


def analyze_sentiment(text: str, models: ModelBundle) -> Tuple[str, float]:
    if not text.strip():
        return "Neutral", 0.0

    if models.sentiment_pipeline is not None:
        try:
            result = models.sentiment_pipeline(text[:512])[0]
            label = result["label"].upper()
            score = float(result["score"])
            if label == "NEGATIVE":
                return "Negative", score
            if label == "POSITIVE":
                return "Positive", score
            return "Neutral", score
        except Exception:
            pass

    return heuristic_sentiment(text)


def heuristic_topic(text: str) -> Tuple[str, float]:
    crime_type = infer_crime_type(text)
    if crime_type == "General Crime":
        return "general crime report", 0.35

    counts = keyword_match_counts(text)
    best_score = counts.get(crime_type, 0)
    topic = CRIME_TO_TOPIC.get(crime_type, "general crime report")
    confidence = min(0.5 + best_score * 0.12, 0.95)
    return topic, confidence


def classify_topic(text: str, models: ModelBundle) -> Tuple[str, float]:
    if not text.strip():
        return "general crime report", 0.0

    if models.topic_pipeline is not None:
        try:
            result = models.topic_pipeline(text[:512], TOPIC_LABELS)
            return result["labels"][0], float(result["scores"][0])
        except Exception:
            pass

    return heuristic_topic(text)


def reconcile_crime_type(crime_type: str, topic_label: str, topic_score: float) -> str:
    # Let obviously critical topics override weaker keyword matches so the final
    # label stays sensible for high-stakes events.
    topic_lower = topic_label.lower()
    if topic_lower in PRIORITY_TOPICS:
        return PRIORITY_TOPICS[topic_lower]
    if crime_type == "General Crime":
        return TOPIC_TO_CRIME.get(topic_lower, "General Crime")
    if topic_score >= 0.88 and topic_lower in TOPIC_TO_CRIME:
        return TOPIC_TO_CRIME[topic_lower]
    return crime_type


def reconcile_topic(topic_label: str, crime_type: str) -> str:
    topic_lower = topic_label.lower()
    if crime_type == "Shooting" and topic_lower == "disturbance":
        return "shooting"
    if crime_type == "Murder" and topic_lower != "murder":
        return "murder"
    if crime_type == "Kidnapping" and topic_lower != "kidnapping":
        return "kidnapping"
    return topic_label


def assign_severity(text: str, sentiment: str, topic: str, crime_type: str) -> str:
    text_lower = text.lower()
    topic_lower = topic.lower()

    if any(term in text_lower for term in HIGH_SEVERITY_TERMS):
        return "High"
    if topic_lower in {"shooting", "kidnapping", "murder"}:
        return "High"
    if crime_type in {"Shooting", "Kidnapping", "Murder"}:
        return "High"
    if any(term in text_lower for term in MEDIUM_SEVERITY_TERMS):
        return "Medium"
    if topic_lower in {
        "robbery",
        "theft",
        "assault",
        "disturbance",
        "fire",
        "accident",
        "drug-related incident",
    }:
        return "Medium"
    if sentiment == "Negative":
        return "Medium"
    return "Low"


def build_text_id(index: int) -> str:
    return f"TXT_{index:03d}"


def discover_input_file(data_dir: Path) -> Path:
    supported_suffixes = (".csv", ".txt", ".jsonl", ".json")
    candidates = [
        path
        for path in sorted(data_dir.iterdir())
        if path.is_file() and path.suffix.lower() in supported_suffixes
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No input file found in {data_dir}. "
            "Add a .csv, .txt, .jsonl, or .json file or pass --input."
        )
    return candidates[0]


def run_pipeline(
    input_path: Path,
    output_path: Path,
    extended_output_path: Path,
    models: ModelBundle,
) -> pd.DataFrame:
    df = extract_raw_records(input_path)
    if df.empty:
        raise ValueError(
            "No usable records were extracted. "
            "The input can be CSV, JSON-lines, JSON, or plain text."
        )

    # Clean once up front so entity extraction, sentiment, topic, and severity
    # all work from the same normalized text.
    df["Cleaned_Text"] = df["Raw_Text"].apply(clean_text)
    df["Processed_Text"] = df["Cleaned_Text"].apply(
        lambda text: preprocess_for_tokens(text, models.stop_words)
    )
    df = df[df["Cleaned_Text"].str.len() > 0].copy()
    df.reset_index(drop=True, inplace=True)

    output_rows: List[Dict[str, Any]] = []
    for index, row in df.iterrows():
        cleaned_text = row["Cleaned_Text"]
        entities = extract_entities(cleaned_text, models)
        sentiment_label, sentiment_score = analyze_sentiment(cleaned_text, models)
        topic_label, topic_score = classify_topic(cleaned_text, models)
        crime_type = infer_crime_type(cleaned_text)
        crime_type = reconcile_crime_type(crime_type, topic_label, topic_score)
        topic_label = reconcile_topic(topic_label, crime_type)
        severity = assign_severity(cleaned_text, sentiment_label, topic_label, crime_type)

        output_rows.append(
            {
                "Text_ID": build_text_id(index + 1),
                "Source": row["Source"],
                "Raw_Text": row["Raw_Text"],
                "Sentiment": sentiment_label,
                "Entities": entities["Entities"],
                "Topic": topic_label,
                "Crime_Type": crime_type,
                "Location_Entity": entities["Location_Entity"],
                "Severity_Label": severity,
                "People": entities["People"],
                "Organizations": entities["Organizations"],
                "Dates": entities["Dates"],
                "Sentiment_Score": round(sentiment_score, 4),
                "Topic_Confidence": round(topic_score, 4),
                "Created_At": row["Created_At"],
            }
        )

    result_df = pd.DataFrame(output_rows)
    submission_cols = [
        "Text_ID",
        "Crime_Type",
        "Location_Entity",
        "Sentiment",
        "Topic",
        "Severity_Label",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    extended_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write both variants: a compact submission CSV and a fuller analyst CSV
    # that keeps the extra evidence useful during debugging and demos.
    result_df[submission_cols].to_csv(output_path, index=False)
    result_df.to_csv(extended_output_path, index=False)
    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze crime-related text files and generate structured CSV output."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the input file. Supports .csv, .txt, .jsonl, or .json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to save the submission CSV.",
    )
    parser.add_argument(
        "--extended-output",
        type=Path,
        default=DEFAULT_EXTENDED_OUTPUT_FILE,
        help="Path to save the full output CSV.",
    )
    parser.add_argument(
        "--no-transformers",
        action="store_true",
        help="Skip Hugging Face pipelines and use rule-based sentiment/topic fallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input is not None else discover_input_file(DATA_DIR)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    models = setup_models(use_transformers=not args.no_transformers)

    print(f"Reading input from: {input_path}")
    result_df = run_pipeline(
        input_path=input_path,
        output_path=args.output,
        extended_output_path=args.extended_output,
        models=models,
    )

    preview_cols = [
        "Text_ID",
        "Crime_Type",
        "Location_Entity",
        "Sentiment",
        "Topic",
        "Severity_Label",
    ]
    print(f"Submission file saved as: {args.output}")
    print(f"Extended file saved as: {args.extended_output}")
    print("\nPreview:")
    print(result_df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
