#!/usr/bin/env python3
"""Integrate audio, document, image, video, and text CSV outputs into one report.

This script supports two useful modes:

1. Prototype mode (default)
   Summarizes the available modality CSVs into a single incident row.
   This is the safest mode when the datasets are unrelated and do not share a
   natural cross-modal incident key yet.

2. Mapped mode (optional)
   If you provide an incident map CSV with a shared Incident_ID, the script
   merges the matched audio, document, image, video, and text rows into one row per
   incident.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO_CSV = REPO_ROOT / "audio_analysis" / "output" / "audio_output.csv"
DEFAULT_DOCUMENT_CSV = REPO_ROOT / "document_analysis" / "output" / "incident_extract.csv"
DEFAULT_IMAGE_CSV = REPO_ROOT / "image_analysis" / "output" / "image_analyst_output.csv"
DEFAULT_VIDEO_CSV = REPO_ROOT / "video_analysis" / "output" / "video_event_log.csv"
DEFAULT_TEXT_CSV = REPO_ROOT / "text_analysis" / "output" / "text_output.csv"
DEFAULT_INCIDENT_MAP = REPO_ROOT / "integration" / "data" / "incident_map.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "integration" / "output" / "final_integrated_incident_report.csv"

MISSING_TOKENS = {"", "nan", "none", "not found", "unknown", "n/a", "na"}
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}

FINAL_COLUMNS = [
    "Incident_ID",
    "Source",
    "Audio_Event",
    "Audio_Location",
    "Audio_Sentiment",
    "Audio_Urgency_Score",
    "PDF_Doc_Type",
    "PDF_Date",
    "PDF_Location",
    "PDF_Officer",
    "PDF_Summary",
    "Image_Scene_Type",
    "Image_Objects",
    "Image_Text_Extracted",
    "Image_Max_Confidence",
    "Video_Event",
    "Video_Time",
    "Video_Max_Persons",
    "Video_Max_Confidence",
    "Text_Crime_Type",
    "Text_Location",
    "Text_Sentiment",
    "Text_Topic",
    "Text_Source_Severity",
    "Severity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate audio, document, image, video, and text outputs into one "
            "incident-ready CSV."
        )
    )
    parser.add_argument("--audio-csv", type=Path, default=DEFAULT_AUDIO_CSV)
    parser.add_argument("--document-csv", type=Path, default=DEFAULT_DOCUMENT_CSV)
    parser.add_argument("--image-csv", type=Path, default=DEFAULT_IMAGE_CSV)
    parser.add_argument("--video-csv", type=Path, default=DEFAULT_VIDEO_CSV)
    parser.add_argument("--text-csv", type=Path, default=DEFAULT_TEXT_CSV)
    parser.add_argument(
        "--incident-map",
        type=Path,
        default=DEFAULT_INCIDENT_MAP,
        help=(
            "CSV that links source IDs to a shared Incident_ID. "
            "Expected columns: Incident_ID, Call_ID, Report_ID, Image_ID, Clip_ID, Text_ID. "
            "Defaults to integration/data/incident_map.csv."
        ),
    )
    parser.add_argument(
        "--prototype",
        action="store_true",
        help="Build one synthetic summary row from all source CSVs. Use only for rough demos.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV not found: {path}")
    return pd.read_csv(path).fillna("")


def clean_value(value: object) -> str:
    # Treat the usual placeholder tokens the same way so downstream merging does
    # not have to care whether a source wrote "Unknown", "None", or blanks.
    text = str(value).strip()
    return "" if text.lower() in MISSING_TOKENS else text


def normalize_token_list(values: Iterable[object], *, split_commas: bool = False) -> list[str]:
    # This version keeps first-seen order and removes duplicates, which is handy
    # for human-readable display fields in the final CSV.
    tokens: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = clean_value(value)
        if not text:
            continue

        parts = [text]
        if split_commas:
            parts = [part.strip() for part in text.split(",")]

        for part in parts:
            cleaned = clean_value(part)
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            tokens.append(cleaned)

    return tokens


def countable_tokens(values: Iterable[object], *, split_commas: bool = False) -> list[str]:
    # Unlike normalize_token_list, this one preserves repeats so we can compute
    # frequency-based summaries such as the dominant event label.
    tokens: list[str] = []
    for value in values:
        text = clean_value(value)
        if not text:
            continue

        parts = [text]
        if split_commas:
            parts = [part.strip() for part in text.split(",")]

        for part in parts:
            cleaned = clean_value(part)
            if cleaned:
                tokens.append(cleaned)

    return tokens


def join_unique(values: Iterable[object], *, sep: str = " | ", split_commas: bool = False) -> str:
    tokens = normalize_token_list(values, split_commas=split_commas)
    return sep.join(tokens)


def dominant_value(values: Iterable[object], fallback: str = "") -> str:
    tokens = countable_tokens(values)
    if not tokens:
        return fallback
    counts = Counter(tokens)
    return counts.most_common(1)[0][0]


def top_values(
    values: Iterable[object],
    *,
    limit: int = 3,
    split_commas: bool = False,
    sep: str = " | ",
) -> str:
    counts = Counter(countable_tokens(values, split_commas=split_commas))
    if not counts:
        return ""
    return sep.join(value for value, _ in counts.most_common(limit))


def strongest_severity(values: Iterable[object]) -> str:
    best_label = ""
    best_rank = 0
    for value in values:
        label = clean_value(value).title()
        rank = SEVERITY_RANK.get(label.lower(), 0)
        if rank > best_rank:
            best_rank = rank
            best_label = label
    return best_label


def normalize_confidence(values: Iterable[object]) -> str:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if numeric.empty:
        return ""
    return f"{numeric.max():.2f}"


def normalize_person_count(values: Iterable[object]) -> str:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if numeric.empty:
        return ""
    return str(int(numeric.max()))


def ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    # Each modality script is allowed to evolve a little, so we backfill missing
    # columns here instead of letting a slight schema difference break integration.
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def prepare_document_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_columns(
        frame,
        ["Report_ID", "Incident_Type", "Date", "Location", "Officer", "Summary"],
    )
    return frame.rename(
        columns={
            "Incident_Type": "PDF_Doc_Type",
            "Date": "PDF_Date",
            "Location": "PDF_Location",
            "Officer": "PDF_Officer",
            "Summary": "PDF_Summary",
        }
    )[
        ["Report_ID", "PDF_Doc_Type", "PDF_Date", "PDF_Location", "PDF_Officer", "PDF_Summary"]
    ]


def prepare_audio_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_columns(
        frame,
        ["Call_ID", "Transcript", "Extracted_Event", "Location", "Sentiment", "Urgency_Score"],
    )
    return frame.rename(
        columns={
            "Extracted_Event": "Audio_Event",
            "Location": "Audio_Location",
            "Sentiment": "Audio_Sentiment",
            "Urgency_Score": "Audio_Urgency_Score",
        }
    )[
        ["Call_ID", "Audio_Event", "Audio_Location", "Audio_Sentiment", "Audio_Urgency_Score"]
    ]


def prepare_image_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_columns(
        frame,
        [
            "Image_ID",
            "Scene_Type",
            "Objects_Detected",
            "Bounding_Boxes",
            "Text_Extracted",
            "Confidence",
        ],
    )
    return frame.rename(
        columns={
            "Scene_Type": "Image_Scene_Type",
            "Objects_Detected": "Image_Objects",
            "Bounding_Boxes": "Image_Bounding_Boxes",
            "Text_Extracted": "Image_Text_Extracted",
            "Confidence": "Image_Confidence",
        }
    )[
        [
            "Image_ID",
            "Image_Scene_Type",
            "Image_Objects",
            "Image_Bounding_Boxes",
            "Image_Text_Extracted",
            "Image_Confidence",
        ]
    ]


def prepare_video_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_columns(
        frame,
        ["Clip_ID", "Timestamp", "Frame_ID", "Event_Detected", "Persons_Count", "Confidence"],
    ).copy()
    frame["Persons_Count"] = pd.to_numeric(frame["Persons_Count"], errors="coerce")
    frame["Confidence"] = pd.to_numeric(frame["Confidence"], errors="coerce")

    # Video arrives as frame-level rows, but integration works much better when
    # each clip is summarized to one incident-friendly record.
    rows: list[dict[str, str]] = []
    for clip_id, group in frame.groupby("Clip_ID", sort=False):
        rows.append(
            {
                "Clip_ID": clean_value(clip_id),
                "Video_Event": dominant_value(group["Event_Detected"], fallback="Unknown"),
                "Video_Time": min(normalize_token_list(group["Timestamp"])) if normalize_token_list(group["Timestamp"]) else "",
                "Video_Max_Persons": normalize_person_count(group["Persons_Count"]),
                "Video_Confidence": normalize_confidence(group["Confidence"]),
            }
        )

    return pd.DataFrame(rows, columns=["Clip_ID", "Video_Event", "Video_Time", "Video_Max_Persons", "Video_Confidence"])


def prepare_text_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_columns(
        frame,
        [
            "Text_ID",
            "Crime_Type",
            "Location_Entity",
            "Sentiment",
            "Topic",
            "Severity_Label",
        ],
    )
    return frame.rename(
        columns={
            "Crime_Type": "Text_Crime_Type",
            "Location_Entity": "Text_Location",
            "Sentiment": "Text_Sentiment",
            "Topic": "Text_Topic",
            "Severity_Label": "Text_Source_Severity",
        }
    )[
        [
            "Text_ID",
            "Text_Crime_Type",
            "Text_Location",
            "Text_Sentiment",
            "Text_Topic",
            "Text_Source_Severity",
        ]
    ]


def summarize_document(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        return {
            "PDF_Doc_Type": "",
            "PDF_Date": "",
            "PDF_Location": "",
            "PDF_Officer": "",
            "PDF_Summary": "",
        }

    return {
        "PDF_Doc_Type": dominant_value(frame["PDF_Doc_Type"], fallback="Unknown"),
        "PDF_Date": join_unique(frame["PDF_Date"]),
        "PDF_Location": join_unique(frame["PDF_Location"]),
        "PDF_Officer": join_unique(frame["PDF_Officer"]),
        "PDF_Summary": join_unique(frame["PDF_Summary"]),
    }


def summarize_audio(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        return {
            "Audio_Event": "",
            "Audio_Location": "",
            "Audio_Sentiment": "",
            "Audio_Urgency_Score": "",
        }

    return {
        "Audio_Event": top_values(frame["Audio_Event"], limit=3),
        "Audio_Location": top_values(frame["Audio_Location"], limit=3),
        "Audio_Sentiment": dominant_value(frame["Audio_Sentiment"], fallback="Unknown"),
        "Audio_Urgency_Score": normalize_confidence(frame["Audio_Urgency_Score"]),
    }


def summarize_image(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        return {
            "Image_Scene_Type": "",
            "Image_Objects": "",
            "Image_Text_Extracted": "",
            "Image_Max_Confidence": "",
        }

    return {
        "Image_Scene_Type": dominant_value(frame["Image_Scene_Type"], fallback="Unknown"),
        "Image_Objects": top_values(frame["Image_Objects"], limit=5, split_commas=True),
        "Image_Text_Extracted": top_values(frame["Image_Text_Extracted"], limit=3),
        "Image_Max_Confidence": normalize_confidence(frame["Image_Confidence"]),
    }


def summarize_video(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        return {
            "Video_Event": "",
            "Video_Time": "",
            "Video_Max_Persons": "",
            "Video_Max_Confidence": "",
        }

    return {
        "Video_Event": top_values(frame["Video_Event"], limit=3),
        "Video_Time": join_unique(sorted(normalize_token_list(frame["Video_Time"]))),
        "Video_Max_Persons": normalize_person_count(frame["Video_Max_Persons"]),
        "Video_Max_Confidence": normalize_confidence(frame["Video_Confidence"]),
    }


def summarize_text(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        return {
            "Text_Crime_Type": "",
            "Text_Location": "",
            "Text_Sentiment": "",
            "Text_Topic": "",
            "Text_Source_Severity": "",
        }

    return {
        "Text_Crime_Type": dominant_value(frame["Text_Crime_Type"], fallback="Unknown"),
        "Text_Location": top_values(frame["Text_Location"], limit=5),
        "Text_Sentiment": dominant_value(frame["Text_Sentiment"], fallback="Unknown"),
        "Text_Topic": dominant_value(frame["Text_Topic"], fallback="Unknown"),
        "Text_Source_Severity": strongest_severity(frame["Text_Source_Severity"]),
    }


def detect_sources(record: dict[str, str]) -> str:
    sources: list[str] = []
    if any(record.get(field, "") for field in ["PDF_Doc_Type", "PDF_Summary", "PDF_Officer"]):
        sources.append("Document")
    if any(record.get(field, "") for field in ["Image_Scene_Type", "Image_Objects", "Image_Text_Extracted"]):
        sources.append("Image")
    if any(record.get(field, "") for field in ["Video_Event", "Video_Time"]):
        sources.append("Video")
    if any(record.get(field, "") for field in ["Text_Crime_Type", "Text_Location", "Text_Sentiment"]):
        sources.append("Text")
    if any(record.get(field, "") for field in ["Audio_Event", "Audio_Location", "Audio_Sentiment"]):
        sources.append("Audio")
    return " + ".join(sources)


def classify_severity(record: dict[str, str]) -> str:
    # Severity is a transparent heuristic on purpose: it is easier to explain in
    # the report/demo than a black-box classifier over a tiny prototype dataset.
    combined = " ".join(
        [
            record.get("Audio_Event", ""),
            record.get("Audio_Sentiment", ""),
            record.get("PDF_Doc_Type", ""),
            record.get("PDF_Summary", ""),
            record.get("Image_Scene_Type", ""),
            record.get("Image_Objects", ""),
            record.get("Video_Event", ""),
            record.get("Text_Crime_Type", ""),
            record.get("Text_Topic", ""),
            record.get("Text_Source_Severity", ""),
        ]
    ).lower()

    high_keywords = {
        "fire",
        "shooting",
        "murder",
        "fight",
        "collapse",
        "collapsing",
        "trapped",
        "weapon",
        "armed",
    }
    medium_keywords = {
        "disturbance",
        "drug",
        "robbery",
        "theft",
        "accident",
        "high-intensity",
    }

    audio_urgency = pd.to_numeric(
        pd.Series([record.get("Audio_Urgency_Score", "")]),
        errors="coerce",
    ).dropna()
    urgency_score = float(audio_urgency.iloc[0]) if not audio_urgency.empty else 0.0
    audio_sentiment = record.get("Audio_Sentiment", "").lower()

    if record.get("Text_Source_Severity", "").lower() == "high" or any(
        keyword in combined for keyword in high_keywords
    ) or (audio_sentiment == "distressed" and urgency_score >= 0.7):
        return "High"
    if record.get("Text_Source_Severity", "").lower() == "medium" or any(
        keyword in combined for keyword in medium_keywords
    ) or urgency_score >= 0.4 or audio_sentiment == "distressed":
        return "Medium"
    return "Low"


def finalize_records(records: list[dict[str, str]]) -> pd.DataFrame:
    finalized: list[dict[str, str]] = []
    for record in records:
        row = {
            "Incident_ID": record.get("Incident_ID", ""),
            "Source": record.get("Source", ""),
            "Audio_Event": record.get("Audio_Event", ""),
            "Audio_Location": record.get("Audio_Location", ""),
            "Audio_Sentiment": record.get("Audio_Sentiment", ""),
            "Audio_Urgency_Score": record.get("Audio_Urgency_Score", ""),
            "PDF_Doc_Type": record.get("PDF_Doc_Type", ""),
            "PDF_Date": record.get("PDF_Date", ""),
            "PDF_Location": record.get("PDF_Location", ""),
            "PDF_Officer": record.get("PDF_Officer", ""),
            "PDF_Summary": record.get("PDF_Summary", ""),
            "Image_Scene_Type": record.get("Image_Scene_Type", ""),
            "Image_Objects": record.get("Image_Objects", ""),
            "Image_Text_Extracted": record.get("Image_Text_Extracted", ""),
            "Image_Max_Confidence": record.get("Image_Max_Confidence", ""),
            "Video_Event": record.get("Video_Event", ""),
            "Video_Time": record.get("Video_Time", ""),
            "Video_Max_Persons": record.get("Video_Max_Persons", ""),
            "Video_Max_Confidence": record.get("Video_Max_Confidence", ""),
            "Text_Crime_Type": record.get("Text_Crime_Type", ""),
            "Text_Location": record.get("Text_Location", ""),
            "Text_Sentiment": record.get("Text_Sentiment", ""),
            "Text_Topic": record.get("Text_Topic", ""),
            "Text_Source_Severity": record.get("Text_Source_Severity", ""),
        }
        # Source and Severity are derived last so they reflect whatever evidence
        # survived the normalization/aggregation steps above.
        row["Source"] = detect_sources(row)
        row["Severity"] = classify_severity(row)
        finalized.append(row)

    return pd.DataFrame(finalized, columns=FINAL_COLUMNS)


def build_prototype_incident(
    audio_rows: pd.DataFrame,
    document_rows: pd.DataFrame,
    image_rows: pd.DataFrame,
    video_rows: pd.DataFrame,
    text_rows: pd.DataFrame,
) -> pd.DataFrame:
    record = {"Incident_ID": "INC_001"}
    record.update(summarize_audio(audio_rows))
    record.update(summarize_document(document_rows))
    record.update(summarize_image(image_rows))
    record.update(summarize_video(video_rows))
    record.update(summarize_text(text_rows))
    return finalize_records([record])


def require_mapping_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    expected = ["Incident_ID", "Call_ID", "Report_ID", "Image_ID", "Clip_ID", "Text_ID"]
    for column in expected:
        if column not in frame.columns:
            frame[column] = ""
    if frame["Incident_ID"].astype(str).str.strip().eq("").all():
        raise ValueError("Incident map must include at least one non-empty Incident_ID.")
    return frame[expected]


def aggregate_incident_group(group: pd.DataFrame) -> dict[str, str]:
    return {
        "Incident_ID": clean_value(group["Incident_ID"].iloc[0]),
        "Audio_Event": join_unique(group["Audio_Event"]),
        "Audio_Location": join_unique(group["Audio_Location"]),
        "Audio_Sentiment": join_unique(group["Audio_Sentiment"]),
        "Audio_Urgency_Score": normalize_confidence(group["Audio_Urgency_Score"]),
        "PDF_Doc_Type": join_unique(group["PDF_Doc_Type"]),
        "PDF_Date": join_unique(group["PDF_Date"]),
        "PDF_Location": join_unique(group["PDF_Location"]),
        "PDF_Officer": join_unique(group["PDF_Officer"]),
        "PDF_Summary": join_unique(group["PDF_Summary"]),
        "Image_Scene_Type": join_unique(group["Image_Scene_Type"]),
        "Image_Objects": join_unique(group["Image_Objects"], split_commas=True),
        "Image_Text_Extracted": join_unique(group["Image_Text_Extracted"]),
        "Image_Max_Confidence": normalize_confidence(group["Image_Confidence"]),
        "Video_Event": join_unique(group["Video_Event"]),
        "Video_Time": join_unique(group["Video_Time"]),
        "Video_Max_Persons": normalize_person_count(group["Video_Max_Persons"]),
        "Video_Max_Confidence": normalize_confidence(group["Video_Confidence"]),
        "Text_Crime_Type": join_unique(group["Text_Crime_Type"]),
        "Text_Location": join_unique(group["Text_Location"]),
        "Text_Sentiment": join_unique(group["Text_Sentiment"]),
        "Text_Topic": join_unique(group["Text_Topic"]),
        "Text_Source_Severity": strongest_severity(group["Text_Source_Severity"]),
    }


def build_mapped_incidents(
    incident_map: pd.DataFrame,
    audio_rows: pd.DataFrame,
    document_rows: pd.DataFrame,
    image_rows: pd.DataFrame,
    video_rows: pd.DataFrame,
    text_rows: pd.DataFrame,
) -> pd.DataFrame:
    # The incident map is the explicit ground truth for cross-modal matching in
    # this assignment; we do not try to "guess" joins automatically.
    mapping = require_mapping_columns(incident_map)
    merged = mapping.merge(audio_rows, on="Call_ID", how="left")
    merged = merged.merge(document_rows, on="Report_ID", how="left")
    merged = merged.merge(image_rows, on="Image_ID", how="left")
    merged = merged.merge(video_rows, on="Clip_ID", how="left")
    merged = merged.merge(text_rows, on="Text_ID", how="left")

    records: list[dict[str, str]] = []
    for _, group in merged.groupby("Incident_ID", sort=False):
        records.append(aggregate_incident_group(group.fillna("")))

    return finalize_records(records)


def main() -> None:
    args = parse_args()

    audio_rows = prepare_audio_rows(read_csv(args.audio_csv, "Audio"))
    document_rows = prepare_document_rows(read_csv(args.document_csv, "Document"))
    image_rows = prepare_image_rows(read_csv(args.image_csv, "Image"))
    video_rows = prepare_video_rows(read_csv(args.video_csv, "Video"))
    text_rows = prepare_text_rows(read_csv(args.text_csv, "Text"))

    # Default to map-based integration. Prototype mode still exists, but only as
    # a fallback for rough experiments when a mapping file is not ready yet.
    if args.prototype:
        final_frame = build_prototype_incident(
            audio_rows,
            document_rows,
            image_rows,
            video_rows,
            text_rows,
        )
    else:
        if not args.incident_map or not args.incident_map.exists():
            raise FileNotFoundError(
                "Incident map not found. Create integration/data/incident_map.csv "
                "or pass --incident-map PATH. Use --prototype only for rough demos."
            )
        incident_map = read_csv(args.incident_map, "Incident map")
        final_frame = build_mapped_incidents(
            incident_map,
            audio_rows,
            document_rows,
            image_rows,
            video_rows,
            text_rows,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_frame.to_csv(args.output, index=False)

    print(f"Wrote {len(final_frame)} integrated incident row(s) to {args.output}")
    print(final_frame.to_string(index=False))


if __name__ == "__main__":
    main()
