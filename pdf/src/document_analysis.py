#!/usr/bin/env python3
"""
PDF incident report extraction pipeline (PyMuPDF + optional Tesseract OCR + spaCy NER).

Dependencies (install separately):
  pip install pymupdf pytesseract pillow pandas spacy
  python -m spacy download en_core_web_sm

System: Tesseract OCR must be installed and on PATH (or set TESSERACT_CMD).
  Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
import pandas as pd
import pytesseract
import spacy
from PIL import Image

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS_FOR_NATIVE = 80
MIN_CHARS_PER_PAGE_HINT = 25
MAX_LOCATION_LEN = 80
MAX_OFFICER_TOKEN_LEN = 40
SUMMARY_MIN_LEN = 50
SUMMARY_MAX_LEN = 500

TRAINING_SIGNALS = (
    "training proposal",
    "training exercise",
    "tabletop",
    "drill",
    "simulation",
    "simulated",
    "practice scenario",
    "training scenario",
    "proposed training",
    "training session",
    "exercise only",
    "not a real incident",
    "fictional",
    "hypothetical scenario",
)

REAL_INCIDENT_SIGNALS = (
    "actual incident",
    "occurred on",
    "reported to",
    "victim",
    "suspect",
    "arrest",
    "use of force",
    "911",
    "dispatch",
)

TRAINING_EQUIPMENT_SIGNALS = (
    "training proposal",
    "proposed training",
    "military equipment training",
    "mrap",
    "training on",
    "equipment training",
    "training scenario",
    "tabletop exercise",
    "training drill",
    "training session",
    "simulation exercise",
    "training document",
    "this is a training",
)

HEADER_LINE_PREFIXES = (
    "from:",
    "to:",
    "subject:",
    "re:",
    "date:",
    "cc:",
    "bcc:",
)

JOB_TITLE_WORDS = {
    "maintenance",
    "officer",
    "sergeant",
    "lieutenant",
    "captain",
    "chief",
    "department",
    "dispatch",
    "supervisor",
    "technician",
    "admin",
    "clerk",
}

NOT_FOUND = "Not Found"


def configure_tesseract() -> None:
    """Use TESSERACT_CMD if set (common on Windows)."""
    cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def load_nlp(model_name: str = "en_core_web_sm") -> Any:
    """Load spaCy model; raises with a clear message if missing."""
    try:
        return spacy.load(model_name)
    except OSError as e:
        raise RuntimeError(
            f"spaCy model {model_name!r} not found. Install with: "
            f"python -m spacy download {model_name}"
        ) from e


def extract_text(
    pdf_path: str,
    ocr_min_total_chars: int = MIN_TEXT_CHARS_FOR_NATIVE,
    ocr_min_chars_per_page: float = MIN_CHARS_PER_PAGE_HINT,
) -> str:
    """
    Read the full PDF with PyMuPDF. If native text is sparse, OCR every page
    and concatenate (full document, no truncation).
    """
    configure_tesseract()
    doc = fitz.open(pdf_path)
    try:
        parts: List[str] = []
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            parts.append(page.get_text("text"))
        native = "\n".join(parts)
        native_clean = native.strip()
        n_pages = max(len(doc), 1)
        avg_len = len(native_clean) / n_pages

        # Most assignment PDFs are text-based, so we prefer the native layer and
        # only pay the OCR cost when the extracted text looks suspiciously thin.
        use_ocr = len(native_clean) < ocr_min_total_chars or avg_len < ocr_min_chars_per_page
        if not use_ocr:
            return native

        ocr_chunks: List[str] = []
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            try:
                ocr_chunks.append(pytesseract.image_to_string(img))
            except Exception as ex:
                logger.exception(
                    "OCR failed on page %s; install Tesseract and/or set TESSERACT_CMD: %s",
                    page_index,
                    ex,
                )
                raise RuntimeError(
                    "OCR is required for this PDF but Tesseract failed. "
                    "Install Tesseract OCR and ensure it is on PATH, "
                    "or set the TESSERACT_CMD environment variable to the executable path."
                ) from ex
        return "\n".join(ocr_chunks)
    finally:
        doc.close()


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _is_probable_name(text: str) -> bool:
    t = text.strip()
    if len(t) < 2 or len(t) > MAX_OFFICER_TOKEN_LEN:
        return False
    if not re.search(r"[A-Za-z]", t):
        return False
    upper_ratio = sum(1 for c in t if c.isupper()) / max(len(t), 1)
    if t.isupper() and len(t.split()) <= 2 and upper_ratio > 0.85:
        if t.upper() in JOB_TITLE_WORDS or len(t) <= 4:
            return False
    words = t.split()
    if len(words) > 5:
        return False
    for w in words:
        wl = w.strip(".,;:")
        if wl.upper() in JOB_TITLE_WORDS and wl.isupper():
            return False
    return True


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 <= b0 or b1 <= a0)


def _merge_person_spans(doc: Any) -> List[Tuple[int, int, str]]:
    """
    Merge adjacent PERSON spans; also merge a following GPE/LOC/FAC that is likely a
    mis-tagged surname (single capitalized token) — fixes 'Zac' + 'Rostan' as LOC.
    """
    ents = sorted(doc.ents, key=lambda e: e.start_char)
    merged: List[Tuple[int, int, str]] = []
    i = 0
    n = len(ents)
    while i < n:
        e = ents[i]
        if e.label_ != "PERSON":
            i += 1
            continue
        start, end = e.start_char, e.end_char
        j = i + 1
        while j < n:
            ne = ents[j]
            gap = doc.text[end : ne.start_char]
            if gap.strip():
                break
            if ne.label_ == "PERSON":
                end = ne.end_char
                j += 1
                continue
            if ne.label_ in ("GPE", "LOC", "FAC"):
                nt = _normalize_ws(ne.text)
                if (
                    len(nt.split()) == 1
                    and len(nt) > 1
                    and nt[0].isupper()
                    and not any(ch.isdigit() for ch in nt)
                ):
                    end = ne.end_char
                    j += 1
                    continue
            break
        name = _normalize_ws(doc.text[start:end])
        if name:
            merged.append((start, end, name))
        i = j if j > i else i + 1
    return merged


def _person_token_set(person_names: List[str]) -> set:
    s = set()
    for p in person_names:
        for w in p.split():
            s.add(w.strip(".,;:").lower())
    return s


def extract_entities(text: str, nlp: Any) -> Dict[str, Any]:
    """
    Full-document NER: merged person names, locations excluding person overlap
    and surname tokens, orgs.
    """
    doc = nlp(text)
    merged_persons = _merge_person_spans(doc)
    person_ranges = [(a, b) for a, b, _ in merged_persons]
    person_names = [t for _, _, t in merged_persons]
    tok_set = _person_token_set(person_names)

    locations: List[str] = []
    locations_lab: List[Tuple[str, str]] = []
    orgs: List[str] = []

    for ent in doc.ents:
        label = ent.label_
        span = _normalize_ws(ent.text)
        if not span:
            continue
        if label == "ORG":
            orgs.append(span)
            continue
        if label not in ("GPE", "LOC", "FAC"):
            continue

        # spaCy sometimes tags surnames or nearby facility names in awkward ways;
        # these filters try to keep person tokens from leaking into locations.
        if any(_spans_overlap(ent.start_char, ent.end_char, ps, pe) for ps, pe in person_ranges):
            continue

        st = span.strip(".,;:")
        if st.lower() in tok_set and len(st.split()) <= 2:
            continue

        is_person_token = False
        for pname in person_names:
            name_tokens = {w.strip(".,;:").lower() for w in pname.split()}
            if st.lower() in name_tokens:
                is_person_token = True
                break
        if is_person_token:
            continue

        locations_lab.append((span, label))
        locations.append(span)

    return {
        "persons": person_names,
        "locations": locations,
        "locations_lab": locations_lab,
        "person_spans": merged_persons,
        "orgs": orgs,
        "doc": doc,
    }


def _score_location_candidate(s: str) -> float:
    s = _normalize_ws(s)
    if not s:
        return -1e9
    score = 0.0
    sl = s.lower()
    if "government" in sl or "united states" in sl or "department of" in sl:
        score -= 8.0
    if "police" in sl or "sheriff" in sl or "pd" in sl.split():
        score -= 2.0
    score -= min(len(s), MAX_LOCATION_LEN) * 0.02
    words = s.split()
    if 1 <= len(words) <= 4:
        score += 2.0
    return score


def _location_label_weight(lab: str) -> float:
    if lab == "GPE":
        return 3.0
    if lab == "LOC":
        return 2.0
    if lab == "FAC":
        return 1.0
    return 0.0


def _build_last_to_full(person_names: List[str]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for name in person_names:
        parts = name.split()
        if len(parts) >= 2:
            last = parts[-1].strip(".,;:")
            if last:
                m[last.lower()] = _normalize_ws(name)
    return m


def _resolve_officer(person_names: List[str], text: str) -> str:
    names = list(dict.fromkeys(_normalize_ws(n) for n in person_names if n))
    multi = [n for n in names if len(n.split()) >= 2 and _is_probable_name(n)]
    last_map = _build_last_to_full(names)

    lines = [ln for ln in text.replace("\r", "\n").split("\n") if ln.strip()]
    officer_lines = []
    for ln in lines:
        lo = ln.lower()
        if re.search(
            r"\b(officer|ofc\.?|prepared by|reporting officer|submitted by|author)\b",
            lo,
        ):
            officer_lines.append(ln)

    def best_match(cands: List[str], pool: List[str]) -> str:
        for line in pool:
            ll = line.lower()
            for c in sorted(cands, key=lambda x: (-len(x.split()), -len(x))):
                if not _is_probable_name(c):
                    continue
                parts = c.lower().split()
                if c.lower() in ll:
                    return c
                if len(parts) >= 2 and parts[-1] in ll and parts[0] in ll:
                    return c
        return ""

    if multi:
        # When the document explicitly mentions an officer line, trust that
        # first; otherwise fall back to the best-looking full person name.
        hit = best_match(multi, officer_lines)
        if hit:
            return hit
        hit = best_match(multi, lines)
        if hit:
            return hit
        return sorted(multi, key=lambda x: (-len(x.split()), -len(x)))[0]

    for match in re.finditer(
        r"(?:ofc\.?|officer)\s+([A-Z][a-z]+)(?:\s+([A-Z][a-z]+))?",
        text,
        re.I,
    ):
        g1, g2 = match.group(1), match.group(2)
        if g2:
            cand = f"{g1} {g2}"
            if _is_probable_name(cand):
                return cand
        if g1 and g1.lower() in last_map:
            return last_map[g1.lower()]

    return NOT_FOUND


def clean_text_v2(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("•", "").replace("â€¢", "")
    text = text.encode("ascii", "ignore").decode()
    return text.strip()


def _pick_location_filtered(
    text: str,
    locations_lab: List[Tuple[str, str]],
) -> str:
    seen = set()
    scored: List[Tuple[float, str]] = []
    for span, lab in locations_lab:
        loc = _normalize_ws(span)
        if len(loc) > MAX_LOCATION_LEN:
            continue
        key = loc.lower()
        if key in seen:
            continue
        seen.add(key)
        s = _score_location_candidate(loc) + _location_label_weight(lab)
        scored.append((s, loc))

    if scored:
        scored.sort(key=lambda x: (-x[0], len(x[1])))

    for score, loc in scored:
        loc_clean = clean_text_v2(loc)

        # Department names are often valid entities but poor incident locations,
        # so we keep biasing toward cleaner place-like strings here.
        if any(word in loc_clean.lower() for word in ["police", "sheriff", "department", "office"]):
            continue

        if len(loc_clean.split()) == 1 and loc_clean.lower() in ["mount", "city", "county"]:
            continue

        if score < -3.0:
            return NOT_FOUND

        return loc_clean

    # If NER cannot find a reliable location, fall back to simple field-style
    # patterns that often appear in reports.
    m = re.search(
        r"(?:location|address)\s*[:\-]\s*([A-Za-z][A-Za-z0-9\s,.-]{2,55})",
        text,
        re.I,
    )
    if m:
        cand = _normalize_ws(m.group(1).split("\n")[0])
        if cand and _score_location_candidate(cand) > -5.0:
            return cand

    m2 = re.search(
        r"\b(?:at|near|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*,\s*[A-Z]{2}\b",
        text,
    )
    if m2:
        return _normalize_ws(m2.group(1))

    return NOT_FOUND


def _classify_incident_type_v2(text: str) -> str:
    t = text.lower()
    # This dataset contains a lot of training-style documents, so the heuristic
    # intentionally leans toward "training" when those signals dominate.
    if "mrap" in t:
        return "Military Equipment Training"
    if "military equipment" in t and "training" in t:
        return "Military Equipment Training"
    if (
        "training proposal" in t
        or "proposed training" in t
        or ("training" in t and "proposal" in t)
    ):
        return "Training Proposal"

    train_equip_hits = sum(1 for k in TRAINING_EQUIPMENT_SIGNALS if k in t)
    train_hits = sum(1 for k in TRAINING_SIGNALS if k in t)
    real_hits = sum(1 for k in REAL_INCIDENT_SIGNALS if k in t)

    if train_equip_hits > 0 and real_hits <= train_hits + 1:
        return "Training Proposal"
    if train_hits > real_hits + 1:
        return "Training Proposal"
    if real_hits > train_hits + 1:
        return "Incident Report"
    return "Incident Report"


def _extract_date_string(text: str) -> str:
    patterns = [
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
    ]
    for line in text.replace("\r", "\n").split("\n"):
        ll = line.strip()
        if re.match(r"^date\s*:", ll, re.I):
            rest = re.sub(r"^date\s*:\s*", "", ll, flags=re.I)
            for pat in patterns:
                m = re.search(pat, rest, re.I)
                if m:
                    return m.group(0).strip()
            if len(rest) > 4:
                return _normalize_ws(rest)[:40]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return m.group(0).strip()
    return ""


def _first_meaningful_summary_sentence(text: str) -> str:
    raw_lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n")]
    buf: List[str] = []
    for ln in raw_lines:
        ll = ln.lower()
        if not ln:
            continue
        if any(ll.startswith(h) for h in HEADER_LINE_PREFIXES):
            continue
        if len(ln) < 20 and not ln.endswith("."):
            continue
        buf.append(ln)
    blob = _normalize_ws(" ".join(buf))
    if not blob:
        blob = _normalize_ws(text)

    sentences = re.split(r"(?<=[.!?])\s+", blob)
    for s in sentences:
        s = _normalize_ws(s)
        if len(s) < SUMMARY_MIN_LEN:
            continue
        sl = s.lower()
        if any(sl.startswith(h.rstrip(":")) for h in HEADER_LINE_PREFIXES):
            continue
        if s[0:1].isdigit() and "/" in s[0:15]:
            continue
        out = s[:SUMMARY_MAX_LEN]
        return out
    chunks = sorted(raw_lines, key=len, reverse=True)
    for c in chunks:
        if len(c) >= 40:
            return c[:SUMMARY_MAX_LEN]
    return blob[:SUMMARY_MAX_LEN] if blob else ""


def extract_incident_info(
    full_text: str,
    entities: Dict[str, Any],
    report_id: str,
) -> Dict[str, str]:
    """
    Combine NER + heuristics; use 'Not Found' when a field is not reliable.
    """
    text = full_text
    persons = list(entities.get("persons") or [])
    locations_lab = list(entities.get("locations_lab") or [])

    incident_type = _classify_incident_type_v2(text)
    date_str = _extract_date_string(text)
    if not date_str.strip():
        date_str = NOT_FOUND

    location = clean_text_v2(_pick_location_filtered(text, locations_lab))
    officer = _resolve_officer(persons, text)

    summary = _first_meaningful_summary_sentence(text)
    if not summary.strip():
        summary = NOT_FOUND

    return {
        "Report_ID": report_id,
        "Incident_Type": incident_type,
        "Date": date_str,
        "Location": location,
        "Officer": officer,
        "Summary": summary,
    }


def build_dataframe(rows: List[Dict[str, str]]) -> pd.DataFrame:
    cols = ["Report_ID", "Incident_Type", "Date", "Location", "Officer", "Summary"]
    return pd.DataFrame(rows, columns=cols)


def run_pipeline(
    pdf_path: str,
    report_id: Optional[str] = None,
    nlp: Optional[Any] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    End-to-end: extract text -> NER -> structured fields -> one-row DataFrame.
    """
    if nlp is None:
        nlp = load_nlp()

    rid = report_id if report_id is not None else unicodedata.normalize("NFKC", os.path.basename(pdf_path))
    # Keep the orchestration thin here: each helper does one job, and this
    # wrapper simply turns them into the one-row CSV the assignment expects.
    full_text = extract_text(pdf_path)
    ent = extract_entities(full_text, nlp)
    info = extract_incident_info(full_text, ent, report_id=rid)
    df = build_dataframe([info])
    return df, full_text


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract structured incident fields from a PDF report.",
    )
    p.add_argument(
        "pdf",
        type=str,
        nargs="?",
        default=None,
        help="Path to input PDF file",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="incident_extract.csv",
        help="Output CSV path (default: incident_extract.csv)",
    )
    p.add_argument(
        "--report-id",
        type=str,
        default=None,
        help="Override Report_ID column (default: PDF filename)",
    )
    p.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model name (default: en_core_web_sm)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    pdf_arg = args.pdf
    if not pdf_arg:
        pdf_arg = input("Path to PDF: ").strip().strip('"').strip("'")
    pdf_path = str(Path(pdf_arg).expanduser().resolve())

    if not Path(pdf_path).is_file():
        logger.error("PDF not found: %s", pdf_path)
        return 1

    out_path = str(Path(args.output).expanduser().resolve())

    try:
        nlp = load_nlp(args.spacy_model)
    except RuntimeError as e:
        logger.error("%s", e)
        return 1

    try:
        df, full_text = run_pipeline(
            pdf_path,
            report_id=args.report_id,
            nlp=nlp,
        )
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        return 1

    logger.info("Extracted character count: %s", len(full_text))
    print(df.to_string(index=False))

    try:
        df.to_csv(out_path, index=False)
    except OSError as e:
        logger.error("Could not write CSV %s: %s", out_path, e)
        return 1

    logger.info("Saved: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
