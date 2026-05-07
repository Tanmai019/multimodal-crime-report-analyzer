"""Streamlit dashboard for browsing the final integrated incident CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "output" / "final_integrated_incident_report.csv"

SEARCH_COLUMNS = [
    "Incident_ID",
    "Source",
    "Audio_Event",
    "Audio_Location",
    "Audio_Sentiment",
    "PDF_Doc_Type",
    "PDF_Location",
    "PDF_Officer",
    "PDF_Summary",
    "Image_Scene_Type",
    "Image_Objects",
    "Image_Text_Extracted",
    "Video_Event",
    "Text_Crime_Type",
    "Text_Location",
    "Text_Topic",
    "Severity",
]

DETAIL_GROUPS = {
    "Audio": ["Audio_Event", "Audio_Location", "Audio_Sentiment", "Audio_Urgency_Score"],
    "Document": ["PDF_Doc_Type", "PDF_Date", "PDF_Location", "PDF_Officer", "PDF_Summary"],
    "Image": ["Image_Scene_Type", "Image_Objects", "Image_Text_Extracted", "Image_Max_Confidence"],
    "Video": ["Video_Event", "Video_Time", "Video_Max_Persons", "Video_Max_Confidence"],
    "Text": ["Text_Crime_Type", "Text_Location", "Text_Sentiment", "Text_Topic", "Text_Source_Severity"],
}


st.set_page_config(
    page_title="Multimodal Incident Dashboard",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_incident_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Cache the CSV so filter tweaks feel instant instead of rereading the file
    # on every small interaction in the dashboard.
    frame = pd.read_csv(path).fillna("")
    for column in frame.columns:
        frame[column] = frame[column].astype(str)
    return frame


def unique_options(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(value for value in frame[column].unique().tolist() if value)


def filter_incidents(frame: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")

        selected_incidents = st.multiselect(
            "Incident ID",
            options=unique_options(frame, "Incident_ID"),
        )
        selected_sources = st.multiselect(
            "Source",
            options=unique_options(frame, "Source"),
        )
        selected_audio_event = st.multiselect(
            "Audio Event",
            options=unique_options(frame, "Audio_Event"),
        )
        selected_severity = st.multiselect(
            "Severity",
            options=unique_options(frame, "Severity"),
        )
        selected_crime_type = st.multiselect(
            "Text Crime Type",
            options=unique_options(frame, "Text_Crime_Type"),
        )
        selected_scene_type = st.multiselect(
            "Image Scene Type",
            options=unique_options(frame, "Image_Scene_Type"),
        )
        keyword = st.text_input("Search keyword", placeholder="fire, shooting, location...")

    filtered = frame.copy()

    if selected_incidents:
        filtered = filtered[filtered["Incident_ID"].isin(selected_incidents)]
    if selected_sources:
        filtered = filtered[filtered["Source"].isin(selected_sources)]
    if selected_audio_event:
        filtered = filtered[filtered["Audio_Event"].isin(selected_audio_event)]
    if selected_severity:
        filtered = filtered[filtered["Severity"].isin(selected_severity)]
    if selected_crime_type:
        filtered = filtered[filtered["Text_Crime_Type"].isin(selected_crime_type)]
    if selected_scene_type:
        filtered = filtered[filtered["Image_Scene_Type"].isin(selected_scene_type)]

    if keyword:
        needle = keyword.strip().lower()
        combined = pd.Series("", index=filtered.index, dtype="string")
        # Build one searchable text blob per row so the keyword box feels
        # flexible without adding a dozen separate search inputs.
        for column in SEARCH_COLUMNS:
            if column in filtered.columns:
                combined = combined.str.cat(filtered[column].astype(str), sep=" ")
        filtered = filtered[combined.str.lower().str.contains(needle, na=False)]

    return filtered.reset_index(drop=True)


def render_metrics(frame: pd.DataFrame) -> None:
    high_count = int((frame.get("Severity", pd.Series(dtype="string")) == "High").sum())
    medium_count = int((frame.get("Severity", pd.Series(dtype="string")) == "Medium").sum())
    source_count = len(unique_options(frame, "Source"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Incidents", len(frame))
    col2.metric("High Severity", high_count)
    col3.metric("Medium Severity", medium_count)
    col4.metric("Source Groups", source_count)


def render_summary_charts(frame: pd.DataFrame) -> None:
    left, right = st.columns(2)

    severity_counts = (
        frame["Severity"].value_counts().rename_axis("Severity").reset_index(name="Count")
        if "Severity" in frame.columns
        else pd.DataFrame(columns=["Severity", "Count"])
    )
    source_counts = (
        frame["Source"].value_counts().rename_axis("Source").reset_index(name="Count")
        if "Source" in frame.columns
        else pd.DataFrame(columns=["Source", "Count"])
    )

    # A quick chart pair makes the dashboard useful in demos before anyone even
    # touches the detail table.
    with left:
        st.subheader("Severity Overview")
        if severity_counts.empty:
            st.info("No severity data available.")
        else:
            st.bar_chart(severity_counts.set_index("Severity"))

    with right:
        st.subheader("Source Overview")
        if source_counts.empty:
            st.info("No source data available.")
        else:
            st.bar_chart(source_counts.set_index("Source"))


def render_incident_table(frame: pd.DataFrame) -> None:
    st.subheader("Incident Table")

    preview_columns = [
        "Incident_ID",
        "Source",
        "Severity",
        "Audio_Event",
        "Text_Crime_Type",
        "Image_Scene_Type",
        "Video_Event",
        "PDF_Doc_Type",
    ]
    visible_columns = [column for column in preview_columns if column in frame.columns]
    st.dataframe(frame[visible_columns], use_container_width=True, hide_index=True)


def render_incident_details(frame: pd.DataFrame) -> None:
    st.subheader("Incident Details")

    if frame.empty:
        st.info("No incidents match the current filters.")
        return

    incident_ids = frame["Incident_ID"].tolist()
    selected_incident = st.selectbox("Select incident", options=incident_ids)
    row = frame.loc[frame["Incident_ID"] == selected_incident].iloc[0]

    summary_cols = st.columns(4)
    summary_cols[0].metric("Incident", row.get("Incident_ID", ""))
    summary_cols[1].metric("Sources", row.get("Source", ""))
    summary_cols[2].metric("Severity", row.get("Severity", ""))
    # Prefer the most incident-like field available so mixed-modality rows still
    # show a readable headline.
    primary_event = (
        row.get("Audio_Event", "")
        or row.get("Text_Crime_Type", "")
        or row.get("Image_Scene_Type", "")
        or row.get("Video_Event", "")
        or row.get("PDF_Doc_Type", "")
    )
    summary_cols[3].metric("Primary Event", primary_event)

    for group_name, columns in DETAIL_GROUPS.items():
        with st.expander(group_name, expanded=(group_name == "Text")):
            group_data = {
                column: row.get(column, "")
                for column in columns
                if column in frame.columns
            }
            # Empty modality sections are hidden behind a short caption so the
            # detail panel stays readable even when many fields are blank.
            non_empty = {key: value for key, value in group_data.items() if value}
            if non_empty:
                detail_frame = pd.DataFrame(
                    {"Field": list(non_empty.keys()), "Value": list(non_empty.values())}
                )
                st.table(detail_frame)
            else:
                st.caption(f"No {group_name.lower()} data mapped for this incident.")


def main() -> None:
    st.title("Multimodal Incident Dashboard")
    st.caption("Filter, inspect, and present the integrated incident dataset for the assignment demo.")

    # Keeping the path editable makes it easy to point the dashboard at a freshly
    # regenerated CSV without changing the code.
    csv_path = st.text_input("Integrated CSV path", value=str(DEFAULT_CSV))

    try:
        frame = load_incident_data(csv_path)
    except FileNotFoundError:
        st.error(f"Integrated CSV not found at `{csv_path}`.")
        st.info("Run `python3 integration/src/integrate_reports.py` first to generate the file.")
        return
    except Exception as exc:
        st.error(f"Could not load the CSV: {exc}")
        return

    filtered = filter_incidents(frame)
    render_metrics(filtered)
    render_summary_charts(filtered)
    render_incident_table(filtered)
    render_incident_details(filtered)


if __name__ == "__main__":
    main()
