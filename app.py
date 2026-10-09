import html
import json
import sqlite3
import io
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from database import create_database
from kb.retrieve import retrieve
from orchestrator import evaluate_response
from agents.base import set_llm_provider

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

import sys

print("PYTHON:", sys.executable)

from google import genai

print("GENAI:", genai.__file__)
# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Response Validation System",
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Single Evaluation"


# ============================================================
# SINGLE EVALUATION STATE
# ============================================================

if "single_question_data" not in st.session_state:
    st.session_state.single_question_data = ""

if "single_ai_response_data" not in st.session_state:
    st.session_state.single_ai_response_data = ""

if "single_reference_answer_data" not in st.session_state:
    st.session_state.single_reference_answer_data = ""

if "single_reference_document_name" not in st.session_state:
    st.session_state.single_reference_document_name = None

if "single_reference_document_bytes" not in st.session_state:
    st.session_state.single_reference_document_bytes = None

if "single_evaluation_data" not in st.session_state:
    st.session_state.single_evaluation_data = None

if "single_llm_provider_data" not in st.session_state:
    st.session_state.single_llm_provider_data = "Gemini"

# Streamlit can discard widget state when a page is not rendered.
# Keep persistent copies and restore the widget keys when the
# Single Evaluation page is opened again.
if "single_question_widget" not in st.session_state:
    st.session_state.single_question_widget = (
        st.session_state.single_question_data
    )

if "single_ai_response_widget" not in st.session_state:
    st.session_state.single_ai_response_widget = (
        st.session_state.single_ai_response_data
    )

if "single_reference_widget" not in st.session_state:
    st.session_state.single_reference_widget = (
        st.session_state.single_reference_answer_data
    )


# ============================================================
# BATCH EVALUATION STATE
# ============================================================

if "batch_csv_name" not in st.session_state:
    st.session_state.batch_csv_name = None

if "batch_csv_bytes" not in st.session_state:
    st.session_state.batch_csv_bytes = None

if "batch_results_data" not in st.session_state:
    st.session_state.batch_results_data = None

if "batch_saved_id" not in st.session_state:
    st.session_state.batch_saved_id = None

if "batch_llm_provider_data" not in st.session_state:
    st.session_state.batch_llm_provider_data = "Gemini"


# ============================================================
# HISTORY STATE
# ============================================================

if "history_type_data" not in st.session_state:
    st.session_state.history_type_data = "Single Evaluations"


# ============================================================
# DATABASE
# ============================================================

create_database()

DATABASE = "evaluation.db"


# ============================================================
# HISTORY DATABASE SCHEMA
# ============================================================

def ensure_history_schema():

    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_batches (
            batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            total_records INTEGER NOT NULL,
            successful INTEGER NOT NULL,
            failed INTEGER NOT NULL,
            skipped INTEGER NOT NULL,
            avg_relevance REAL,
            avg_accuracy REAL,
            avg_completeness REAL,
            avg_hallucination REAL,
            pass_count INTEGER,
            needs_improvement_count INTEGER,
            fail_count INTEGER,
            hallucination_count INTEGER,
            status TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_submissions (
            submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            reference_answer TEXT,
            source_document TEXT,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)

    columns = {
        row[1]
        for row in cur.execute(
            "PRAGMA table_info(evaluation_submissions)"
        ).fetchall()
    }

    additions = {
        "evaluation_type": "TEXT DEFAULT 'Legacy'",
        "batch_id": "INTEGER",
        "relevance_result": "TEXT",
        "accuracy_result": "TEXT",
        "completeness_result": "TEXT",
        "hallucination_result": "TEXT",
        "verdict_result": "TEXT",
        "overall_score": "REAL",
        "risk_level": "TEXT",
        "error_message": "TEXT",
    }

    for column, definition in additions.items():

        if column not in columns:

            cur.execute(
                f"ALTER TABLE evaluation_submissions "
                f"ADD COLUMN {column} {definition}"
            )

    conn.commit()
    conn.close()


ensure_history_schema()


# ============================================================
# JSON HELPERS
# ============================================================

def json_text(value):

    if value is None:
        return None

    return json.dumps(
        value,
        ensure_ascii=False,
    )


def json_value(value):

    if value in (None, ""):
        return {}

    if isinstance(value, (dict, list)):
        return value

    try:

        return json.loads(value)

    except Exception:

        return value


# ============================================================
# SAVE SINGLE HISTORY
# ============================================================

def save_single_history(
    question,
    ai_response,
    reference_answer,
    source_document,
    relevance,
    accuracy,
    completeness,
    hallucination,
    verdict,
    overall_score,
    risk_level,
):

    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    cur.execute("""
        INSERT INTO evaluation_submissions (
            question,
            ai_response,
            reference_answer,
            source_document,
            timestamp,
            status,
            evaluation_type,
            batch_id,
            relevance_result,
            accuracy_result,
            completeness_result,
            hallucination_result,
            verdict_result,
            overall_score,
            risk_level,
            error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        question,
        ai_response,
        reference_answer,
        source_document,
        timestamp,
        "completed",
        "Single",
        None,
        json_text(relevance),
        json_text(accuracy),
        json_text(completeness),
        json_text(hallucination),
        json_text(verdict),
        float(overall_score),
        risk_level,
        "",
    ))

    submission_id = cur.lastrowid

    conn.commit()
    conn.close()

    return submission_id


# ============================================================
# SAVE BATCH HISTORY
# ============================================================

def save_batch_history(batch_results):

    if not batch_results:
        return None

    total_records = len(batch_results)

    df = pd.DataFrame(batch_results)

    successful_count = int(
        (
            df["Status"] == "Success"
        ).sum()
    )

    failed_count = int(
        df["Status"]
        .astype(str)
        .str.startswith("Failed")
        .sum()
    )

    skipped_count = int(
        df["Status"]
        .astype(str)
        .str.startswith("Skipped")
        .sum()
    )

    invalid_count = int(
        df["Status"]
        .astype(str)
        .str.startswith("Invalid")
        .sum()
    )

    successful_df = df[
        df["Status"] == "Success"
    ]

    if not successful_df.empty:

        avg_relevance = float(
            successful_df["Relevance"].mean()
        )

        avg_accuracy = float(
            successful_df["Accuracy"].mean()
        )

        avg_completeness = float(
            successful_df["Completeness"].mean()
        )

        avg_hallucination = float(
            successful_df["Hallucination"].mean()
        )

        pass_count = int(
            (
                successful_df["Verdict"]
                == "Pass"
            ).sum()
        )

        needs_count = int(
            (
                successful_df["Verdict"]
                == "Needs Improvement"
            ).sum()
        )

        fail_count = int(
            (
                successful_df["Verdict"]
                == "Fail"
            ).sum()
        )

        hallucination_count = int(
            (
                successful_df["Hallucination"]
                < 5
            ).sum()
        )

    else:

        avg_relevance = 0.0
        avg_accuracy = 0.0
        avg_completeness = 0.0
        avg_hallucination = 0.0

        pass_count = 0
        needs_count = 0
        fail_count = 0
        hallucination_count = 0

    # ========================================================
    # BATCH STATUS
    # ========================================================

    if (
        failed_count == 0
        and skipped_count == 0
        and invalid_count == 0
    ):

        batch_status = "completed"

    else:

        batch_status = "completed_with_issues"

    # ========================================================
    # DATABASE
    # ========================================================

    conn = sqlite3.connect(DATABASE)

    try:

        cur = conn.cursor()

        timestamp = datetime.now().isoformat(
            timespec="seconds"
        )

        cur.execute("""
            INSERT INTO evaluation_batches (
                timestamp,
                total_records,
                successful,
                failed,
                skipped,
                avg_relevance,
                avg_accuracy,
                avg_completeness,
                avg_hallucination,
                pass_count,
                needs_improvement_count,
                fail_count,
                hallucination_count,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            total_records,
            successful_count,
            failed_count + invalid_count,
            skipped_count,
            avg_relevance,
            avg_accuracy,
            avg_completeness,
            avg_hallucination,
            pass_count,
            needs_count,
            fail_count,
            hallucination_count,
            batch_status,
        ))

        batch_id = cur.lastrowid

        # ====================================================
        # SAVE EVERY RECORD
        # ====================================================

        for row in batch_results:

            cur.execute("""
                INSERT INTO evaluation_submissions (
                    question,
                    ai_response,
                    reference_answer,
                    source_document,
                    timestamp,
                    status,
                    evaluation_type,
                    batch_id,
                    relevance_result,
                    accuracy_result,
                    completeness_result,
                    hallucination_result,
                    verdict_result,
                    overall_score,
                    risk_level,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get(
                    "Question",
                    "",
                ),

                row.get(
                    "AI Response",
                    "",
                ),

                row.get(
                    "Reference Answer",
                    "",
                ),

                row.get(
                    "Reference Document"
                ),

                timestamp,

                row.get(
                    "Status",
                    "Unknown",
                ),

                "Batch",

                batch_id,

                json_text(
                    row.get(
                        "Relevance Result",
                        {},
                    )
                ),

                json_text(
                    row.get(
                        "Accuracy Result",
                        {},
                    )
                ),

                json_text(
                    row.get(
                        "Completeness Result",
                        {},
                    )
                ),

                json_text(
                    row.get(
                        "Hallucination Result",
                        {},
                    )
                ),

                json_text(
                    row.get(
                        "Verdict Result",
                        {},
                    )
                ),

                float(
                    row.get(
                        "Overall Score",
                        0,
                    )
                    or 0
                ),

                row.get(
                    "Risk Level",
                    "Unknown",
                ),

                row.get(
                    "Error",
                    "",
                ),
            ))

        conn.commit()

        return batch_id

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()


# ============================================================
# HISTORY READ FUNCTIONS
# ============================================================

def get_single_history():

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT *
        FROM evaluation_submissions
        WHERE evaluation_type = 'Single'
          AND status = 'completed'
        ORDER BY timestamp DESC
    """).fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


def get_batch_history():

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT *
        FROM evaluation_batches
        WHERE status IN (
            'completed',
            'completed_with_issues'
        )
        ORDER BY timestamp DESC
    """).fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


def get_batch_records(batch_id):

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT *
        FROM evaluation_submissions
        WHERE evaluation_type = 'Batch'
          AND batch_id = ?
        ORDER BY submission_id ASC
    """, (
        batch_id,
    )).fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# RISK
# ============================================================

def get_risk_level(hallucination_data):

    score = float(
        hallucination_data.get(
            "score",
            5,
        )
        or 0
    )

    flagged = hallucination_data.get(
        "flagged_claims",
        [],
    ) or []

    status = str(
        hallucination_data.get(
            "status",
            "",
        )
    ).lower()

    if (
        score <= 2
        or len(flagged) >= 2
        or "detected" in status
    ):

        return {
            "level": "High",
            "class": "risk-high",
            "fill": "risk-bar-fill-high",
            "icon": "🔴",
            "text": "High Risk",
        }

    if (
        score <= 3.5
        or len(flagged) == 1
    ):

        return {
            "level": "Medium",
            "class": "risk-medium",
            "fill": "risk-bar-fill-medium",
            "icon": "🟡",
            "text": "Medium Risk",
        }

    return {
        "level": "Low",
        "class": "risk-low",
        "fill": "risk-bar-fill-low",
        "icon": "🟢",
        "text": "Low Risk",
    }


# ============================================================
# TEXT HELPERS
# ============================================================

def safe_text(value):

    if value is None:
        return "Not available."

    return str(value)


def format_verdict_summary(verdict_data):

    if not verdict_data:

        return (
            "No summary reasoning provided.",
            "Validated",
            0.0,
        )

    if isinstance(
        verdict_data,
        str,
    ):

        try:

            verdict_data = json.loads(
                verdict_data
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):

            return (
                str(verdict_data),
                "Verdict Analysis",
                0.0,
            )

    if isinstance(
        verdict_data,
        dict,
    ):

        overall_score = float(
            verdict_data.get(
                "overall_score",
                verdict_data.get(
                    "score",
                    0,
                ),
            )
            or 0
        )

        verdict_status = str(
            verdict_data.get(
                "verdict",
                verdict_data.get(
                    "status",
                    "Evaluation Complete",
                ),
            )
        )

        reasoning = verdict_data.get(
            "reasoning",
            "",
        )

        summary = verdict_data.get(
            "summary",
            "",
        )

        details = verdict_data.get(
            "details",
            "",
        )

        text_parts = []

        if reasoning:

            text_parts.append(
                str(reasoning)
            )

        if (
            summary
            and summary != reasoning
        ):

            text_parts.append(
                f"Summary: {summary}"
            )

        if details:

            text_parts.append(
                f"Details: {details}"
            )

        final_text = (
            "\n\n".join(text_parts)
            if text_parts
            else
            "Evaluation completed successfully."
        )

        return (
            final_text,
            verdict_status,
            overall_score,
        )

    return (
        str(verdict_data),
        "Completed",
        0.0,
    )


# ============================================================
# AGENT RESULTS
# ============================================================

def render_agent_results(
    relevance,
    accuracy,
    completeness,
    hallucination,
    risk_info,
):

    st.subheader(
        "🧩 Agent Workspace"
    )

    st.caption(
        "Four independent validation agents analyze "
        "the response before the Verdict Agent "
        "produces the final decision."
    )

    agents = [
        (
            "🎯",
            "Relevance",
            relevance,
        ),
        (
            "✓",
            "Accuracy",
            accuracy,
        ),
        (
            "◈",
            "Completeness",
            completeness,
        ),
        (
            "⚠️",
            "Hallucination",
            hallucination,
        ),
    ]

    cols = st.columns(4)

    for col, (
        icon,
        name,
        data,
    ) in zip(
        cols,
        agents,
    ):

        with col:

            score = float(
                data.get(
                    "score",
                    0,
                )
                or 0
            )

            score = max(
                0.0,
                min(
                    5.0,
                    score,
                ),
            )

            st.markdown(
                f"### {icon} {name}"
            )

            st.caption(
                "Independent validation agent"
            )

            st.metric(
                "Score",
                f"{score:.1f} / 5.0",
            )

            st.progress(
                score / 5.0
            )

            if score >= 4:

                st.success("Good")

            elif score >= 2.5:

                st.warning("Moderate")

            else:

                st.error("Low")

    st.divider()

    st.subheader(
        "🔎 Agent Details"
    )

    with st.expander(
        "🎯 Relevance Agent"
    ):

        st.write(
            relevance.get(
                "reasoning",
                "No reasoning returned.",
            )
        )

    with st.expander(
        "✓ Accuracy Agent"
    ):

        st.write(
            accuracy.get(
                "reasoning",
                "No reasoning returned.",
            )
        )

        evidence = accuracy.get(
            "supporting_evidence",
            [],
        ) or []

        if evidence:

            st.markdown(
                "**Supporting Evidence**"
            )

            for item in evidence:

                st.info(
                    str(item)
                )

        else:

            st.caption(
                "No supporting evidence returned."
            )

    with st.expander(
        "◈ Completeness Agent"
    ):

        st.write(
            completeness.get(
                "reasoning",
                "No reasoning returned.",
            )
        )

        missing = completeness.get(
            "missing_components",
            [],
        ) or []

        if missing:

            st.markdown(
                "**Missing Components**"
            )

            for item in missing:

                st.warning(
                    str(item)
                )

        else:

            st.success(
                "No missing components identified."
            )

    with st.expander(
        "⚠️ Hallucination Detection Agent"
    ):

        st.write(
            hallucination.get(
                "reasoning",
                "No reasoning returned.",
            )
        )

        flagged = hallucination.get(
            "flagged_claims",
            [],
        ) or []

        if not flagged:

            st.success(
                "No ungrounded or contradictory "
                "claims detected."
            )

        else:

            st.warning(
                f"{len(flagged)} flagged claim(s) detected."
            )

            for i, claim in enumerate(
                flagged,
                1,
            ):

                if isinstance(
                    claim,
                    dict,
                ):

                    claim_text = str(
                        claim.get(
                            "claim",
                            "Not available.",
                        )
                    )

                    classification = str(
                        claim.get(
                            "classification",
                            "Flagged",
                        )
                    )

                    reason = str(
                        claim.get(
                            "reason",
                            "No additional explanation returned.",
                        )
                    )

                else:

                    claim_text = str(
                        claim
                    )

                    classification = "Flagged"

                    reason = (
                        "No additional explanation returned."
                    )

                st.markdown(
                    f"**Flagged Claim #{i} — "
                    f"{classification}**"
                )

                st.write(
                    f"**Claim:** {claim_text}"
                )

                st.write(
                    f"**Reason:** {reason}"
                )

                if i < len(flagged):

                    st.divider()


# ============================================================
# FINAL RESULT
# ============================================================

# ============================================================
# SINGLE EVALUATION PDF REPORT
# ============================================================

def generate_single_pdf(
    question,
    ai_response,
    reference_answer,
    evaluation_results,
    saved_id=None,
):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "SingleTitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=14,
    )

    heading_style = ParagraphStyle(
        "SingleHeadingStyle",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "SingleBodyStyle",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        spaceAfter=6,
    )

    small_style = ParagraphStyle(
        "SingleSmallStyle",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        spaceAfter=4,
    )

    story = []

    # ========================================================
    # GET RESULTS
    # ========================================================

    accuracy = (
        evaluation_results.get(
            "accuracy",
            {},
        )
        or {}
    )

    completeness = (
        evaluation_results.get(
            "completeness",
            {},
        )
        or {}
    )

    relevance = (
        evaluation_results.get(
            "relevance",
            {},
        )
        or {}
    )

    hallucination = (
        evaluation_results.get(
            "hallucination",
            {},
        )
        or {}
    )

    verdict = (
        evaluation_results.get(
            "verdict",
            {},
        )
        or {}
    )

    risk_info = get_risk_level(
        hallucination
    )

    (
        verdict_text,
        verdict_name,
        overall_score,
    ) = format_verdict_summary(
        verdict
    )

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "AI Response Validation System",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Single Evaluation Report",
            heading_style,
        )
    )

    if saved_id is not None:

        story.append(
            Paragraph(
                f"<b>Evaluation ID:</b> {saved_id}",
                body_style,
            )
        )

    # ========================================================
    # INPUT
    # ========================================================

    story.append(
        Paragraph(
            "Evaluation Input",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Question:</b><br/>"
            f"{html.escape(str(question))}",
            body_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>AI Response:</b><br/>"
            f"{html.escape(str(ai_response))}",
            body_style,
        )
    )

    reference_text = (
        reference_answer
        if reference_answer
        else "No reference answer provided."
    )

    story.append(
        Paragraph(
            f"<b>Reference Answer:</b><br/>"
            f"{html.escape(str(reference_text))}",
            body_style,
        )
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    story.append(
        Paragraph(
            "Final Evaluation Result",
            heading_style,
        )
    )

    final_data = [
        ["Metric", "Result"],
        [
            "System Verdict",
            str(verdict_name),
        ],
        [
            "Weighted Score",
            f"{overall_score:.1f} / 5.0",
        ],
        [
            "Hallucination Risk",
            str(risk_info["text"]),
        ],
    ]

    final_table = Table(
        final_data,
        colWidths=[180, 250],
    )

    final_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(
        final_table
    )

    # ========================================================
    # VERDICT SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "Verdict Summary",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            html.escape(
                str(verdict_text)
            ),
            body_style,
        )
    )

    # ========================================================
    # AGENT SCORES
    # ========================================================

    story.append(
        Paragraph(
            "Agent Scores",
            heading_style,
        )
    )

    agent_data = [
        [
            "Agent",
            "Score",
        ],
        [
            "Relevance",
            f"{float(relevance.get('score', 0)):.1f} / 5",
        ],
        [
            "Accuracy",
            f"{float(accuracy.get('score', 0)):.1f} / 5",
        ],
        [
            "Completeness",
            f"{float(completeness.get('score', 0)):.1f} / 5",
        ],
        [
            "Hallucination",
            f"{float(hallucination.get('score', 0)):.1f} / 5",
        ],
    ]

    agent_table = Table(
        agent_data,
        colWidths=[200, 100],
    )

    agent_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        agent_table
    )

    # ========================================================
    # AGENT REASONING
    # ========================================================

    story.append(
        Paragraph(
            "Agent Details",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Relevance Agent:</b><br/>"
            + html.escape(
                str(
                    relevance.get(
                        "reasoning",
                        "No reasoning returned.",
                    )
                )
            ),
            body_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Accuracy Agent:</b><br/>"
            + html.escape(
                str(
                    accuracy.get(
                        "reasoning",
                        "No reasoning returned.",
                    )
                )
            ),
            body_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Completeness Agent:</b><br/>"
            + html.escape(
                str(
                    completeness.get(
                        "reasoning",
                        "No reasoning returned.",
                    )
                )
            ),
            body_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Hallucination Detection Agent:</b><br/>"
            + html.escape(
                str(
                    hallucination.get(
                        "reasoning",
                        "No reasoning returned.",
                    )
                )
            ),
            body_style,
        )
    )

    # ========================================================
    # SUPPORTING EVIDENCE
    # ========================================================

    evidence = (
        accuracy.get(
            "supporting_evidence",
            [],
        )
        or []
    )

    if evidence:

        story.append(
            Paragraph(
                "Supporting Evidence",
                heading_style,
            )
        )

        for item in evidence:

            story.append(
                Paragraph(
                    "• "
                    + html.escape(
                        str(item)
                    ),
                    small_style,
                )
            )

    # ========================================================
    # MISSING COMPONENTS
    # ========================================================

    missing = (
        completeness.get(
            "missing_components",
            [],
        )
        or []
    )

    if missing:

        story.append(
            Paragraph(
                "Missing Components",
                heading_style,
            )
        )

        for item in missing:

            story.append(
                Paragraph(
                    "• "
                    + html.escape(
                        str(item)
                    ),
                    small_style,
                )
            )

    # ========================================================
    # FLAGGED CLAIMS
    # ========================================================

    flagged = (
        hallucination.get(
            "flagged_claims",
            [],
        )
        or []
    )

    if flagged:

        story.append(
            Paragraph(
                "Flagged Hallucination Claims",
                heading_style,
            )
        )

        for i, claim in enumerate(
            flagged,
            1,
        ):

            if isinstance(
                claim,
                dict,
            ):

                claim_text = str(
                    claim.get(
                        "claim",
                        "Not available.",
                    )
                )

                classification = str(
                    claim.get(
                        "classification",
                        "Flagged",
                    )
                )

                reason = str(
                    claim.get(
                        "reason",
                        "No additional explanation returned.",
                    )
                )

            else:

                claim_text = str(
                    claim
                )

                classification = "Flagged"

                reason = (
                    "No additional explanation returned."
                )

            story.append(
                Paragraph(
                    f"<b>Claim #{i} — "
                    f"{html.escape(classification)}</b>",
                    body_style,
                )
            )

            story.append(
                Paragraph(
                    f"<b>Claim:</b> "
                    f"{html.escape(claim_text)}",
                    small_style,
                )
            )

            story.append(
                Paragraph(
                    f"<b>Reason:</b> "
                    f"{html.escape(reason)}",
                    small_style,
                )
            )

    # ========================================================
    # CREATE PDF
    # ========================================================

    doc.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# FINAL RESULT
# ============================================================

def render_final_result(
    evaluation_results,
    saved_id=None,
    question="",
    ai_response="",
    reference_answer="",
):

    accuracy = (
        evaluation_results.get(
            "accuracy",
            {},
        )
        or {}
    )

    completeness = (
        evaluation_results.get(
            "completeness",
            {},
        )
        or {}
    )

    relevance = (
        evaluation_results.get(
            "relevance",
            {},
        )
        or {}
    )

    hallucination = (
        evaluation_results.get(
            "hallucination",
            {},
        )
        or {}
    )

    verdict = (
        evaluation_results.get(
            "verdict",
            {},
        )
        or {}
    )

    risk_info = get_risk_level(
        hallucination
    )

    (
        verdict_text,
        verdict_name,
        overall_score,
    ) = format_verdict_summary(
        verdict
    )

    st.subheader(
        "⚖️ Final Evaluation Verdict"
    )

    st.caption(
        "The final verdict is generated from the "
        "combined results of all validation agents."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "System Verdict",
            verdict_name,
        )

    with c2:

        st.metric(
            "Weighted Score",
            f"{overall_score:.1f} / 5.0",
        )

        st.progress(
            max(
                0.0,
                min(
                    1.0,
                    overall_score / 5.0,
                ),
            )
        )

    with c3:

        st.metric(
            "Hallucination Risk",
            f"{risk_info['icon']} "
            f"{risk_info['text']}",
        )

        risk_progress = {
            "Low": 1 / 3,
            "Medium": 2 / 3,
            "High": 1.0,
        }.get(
            risk_info["level"],
            0.0,
        )

        st.progress(
            risk_progress
        )

    st.divider()

    st.subheader(
        "📝 Verdict Summary"
    )

    st.info(
        verdict_text
    )

    render_agent_results(
        relevance,
        accuracy,
        completeness,
        hallucination,
        risk_info,
    )

    if saved_id is not None:

        st.success(
            f"Saved to History • "
            f"Evaluation ID: {saved_id}"
        )

    # ========================================================
    # PDF DOWNLOAD
    # ========================================================

    pdf_data = generate_single_pdf(
        question=question,
        ai_response=ai_response,
        reference_answer=reference_answer,
        evaluation_results=evaluation_results,
        saved_id=saved_id,
    )

    st.download_button(
        label="📄 Download Single Evaluation PDF",
        data=pdf_data,
        file_name=(
            f"single_evaluation_"
            f"{saved_id if saved_id is not None else 'report'}.pdf"
        ),
        mime="application/pdf",
        key=(
            f"single_pdf_"
            f"{saved_id if saved_id is not None else 'report'}"
        ),
        type="primary",
        use_container_width=True,
    )


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap'
    );

    html,
    body,
    [class*="css"] {
        font-family:
            'Plus Jakarta Sans',
            sans-serif;
    }

    .stApp {
        background-color: #0b0f19;

        background-image:
            radial-gradient(
                at 15% 15%,
                rgba(99,102,241,.12) 0px,
                transparent 45%
            ),
            radial-gradient(
                at 85% 85%,
                rgba(20,184,166,.10) 0px,
                transparent 45%
            );

        color: #f8fafc;
    }

    /* =========================================================
       SINGLE EVALUATION UI
       ========================================================= */

    .ui-section-title { color:#F8FAFC; font-size:18px; font-weight:800; margin-top:24px; margin-bottom:6px; letter-spacing:-0.2px; }
    .ui-section-subtitle { color:#94A3B8; font-size:13px; font-weight:500; margin-bottom:18px; line-height:1.5; }
    .pipeline-title { margin-top:28px; }

    .validation-pipeline { display:flex; align-items:center; width:100%; gap:7px; margin:12px 0 30px 0; }
    .pipeline-item { flex:1; }
    .pipeline-card { min-height:72px; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:10px 7px; border-radius:14px; background:linear-gradient(145deg,rgba(30,41,70,.92),rgba(15,23,42,.96)); border:1px solid rgba(99,102,241,.25); box-shadow:0 8px 25px rgba(0,0,0,.18); transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease; }
    .pipeline-card:hover { transform:translateY(-4px); border-color:rgba(99,102,241,.75); box-shadow:0 10px 30px rgba(99,102,241,.18); }
    .pipeline-icon { font-size:20px; margin-bottom:5px; }
    .pipeline-label { color:#CBD5E1; font-size:11px; font-weight:700; text-align:center; white-space:nowrap; }
    .pipeline-arrow { color:#6366F1; font-size:18px; font-weight:800; flex-shrink:0; }

    .input-title { margin-top:10px; }
    .reference-title { margin-top:25px; }
    .field-heading { color:#CBD5E1; font-size:14px; font-weight:700; margin-bottom:8px; }
    .optional-label { color:#64748B; font-size:11px; font-weight:500; margin-left:5px; }

    .stTextArea textarea { background:linear-gradient(145deg,#111827,#0F172A) !important; color:#F8FAFC !important; border:1px solid #334155 !important; border-radius:14px !important; padding:15px !important; font-size:14px !important; line-height:1.55 !important; transition:border-color .2s ease,box-shadow .2s ease; }
    .stTextArea textarea:focus { border-color:#6366F1 !important; box-shadow:0 0 0 2px rgba(99,102,241,.15) !important; }

    div[data-baseweb="select"] > div { background:#111827 !important; border:1px solid #334155 !important; border-radius:12px !important; color:#F8FAFC !important; min-height:48px !important; transition:border-color .2s ease,box-shadow .2s ease; }
    div[data-baseweb="select"] > div:hover { border-color:#6366F1 !important; }
    div[data-baseweb="select"] span { color:#F8FAFC !important; }

    [data-testid="stFileUploader"] { background:linear-gradient(145deg,#111827,#0F172A) !important; border:1px dashed #475569 !important; border-radius:14px !important; padding:8px !important; transition:border-color .2s ease,background .2s ease; }
    [data-testid="stFileUploader"]:hover { border-color:#6366F1 !important; }
    .upload-help { color:#64748B; font-size:11px; margin-top:-5px; margin-bottom:8px; }

    .ready-status { display:flex; align-items:center; justify-content:center; gap:8px; margin-top:24px; margin-bottom:10px; color:#94A3B8; font-size:13px; font-weight:600; }
    .ready-dot { width:8px; height:8px; border-radius:50%; background:#22C55E; box-shadow:0 0 10px rgba(34,197,94,.7); animation:readyPulse 2s ease-in-out infinite; }
    @keyframes readyPulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:.55; transform:scale(.82); } }

    .stButton > button[kind="primary"] { min-height:52px !important; font-size:15px !important; font-weight:800 !important; border-radius:14px !important; background:linear-gradient(90deg,#4F46E5,#2563EB,#0891B2) !important; box-shadow:0 8px 25px rgba(79,70,229,.22); transition:transform .2s ease,box-shadow .2s ease,filter .2s ease; }
    .stButton > button[kind="primary"]:hover { transform:translateY(-2px); filter:brightness(1.08); box-shadow:0 12px 32px rgba(79,70,229,.35); }

    /* =========================================================
       BATCH EVALUATION UI
       ========================================================= */

    .batch-section-title {
        color:#F8FAFC;
        font-size:18px;
        font-weight:800;
        margin-top:24px;
        margin-bottom:6px;
    }

    .batch-section-subtitle {
        color:#94A3B8;
        font-size:13px;
        line-height:1.55;
        margin-bottom:14px;
    }

    .batch-section-subtitle b {
        color:#CBD5E1;
    }

    .batch-workflow {
        display:flex;
        align-items:center;
        width:100%;
        gap:8px;
        margin:12px 0 24px 0;
    }

    .batch-workflow-item {
        flex:1;
    }

    .batch-workflow-card {
        min-height:66px;
        display:flex;
        align-items:center;
        justify-content:center;
        gap:8px;
        padding:10px 8px;
        border-radius:14px;
        background:linear-gradient(145deg,rgba(30,41,70,.92),rgba(15,23,42,.96));
        border:1px solid rgba(99,102,241,.25);
        box-shadow:0 8px 25px rgba(0,0,0,.18);
        transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease;
    }

    .batch-workflow-card:hover {
        transform:translateY(-3px);
        border-color:rgba(99,102,241,.70);
        box-shadow:0 10px 28px rgba(99,102,241,.16);
    }

    .batch-workflow-icon {
        font-size:19px;
    }

    .batch-workflow-label {
        color:#CBD5E1;
        font-size:12px;
        font-weight:700;
        text-align:center;
    }

    .batch-workflow-arrow {
        color:#6366F1;
        font-size:18px;
        font-weight:800;
        flex-shrink:0;
    }

    .batch-ready-status {
        display:flex;
        align-items:center;
        justify-content:center;
        gap:8px;
        margin:18px 0 8px 0;
        color:#94A3B8;
        font-size:13px;
        font-weight:700;
    }

    .batch-summary-title {
        color:#CBD5E1;
        font-size:16px;
        font-weight:800;
        margin-top:22px;
        margin-bottom:10px;
    }

    .batch-summary-title + div[data-testid="stHorizontalBlock"] {
        margin-bottom:8px;
    }

    /* Make Streamlit file uploader match the dark theme */
    [data-testid="stFileUploader"] section {
        background:linear-gradient(145deg,#111827,#0F172A) !important;
        border:1px dashed #475569 !important;
        border-radius:14px !important;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1280px;
    }

    label,
    .stTextArea label,
    p {
        color:#cbd5e1 !important;
        font-weight:600;
    }

    .stTextArea textarea {
        background:#111827 !important;
        color:#f8fafc !important;
        border:1px solid #1f293d !important;
        border-radius:14px !important;
    }

    [data-testid="stFileUploader"] {
        background:#111827;
        border:1px dashed #334155;
        border-radius:14px;
        padding:10px;
    }

    .stButton > button {
        background:
            linear-gradient(
                135deg,
                #4f46e5 0%,
                #06b6d4 100%
            ) !important;

        color:#fff !important;
        font-weight:700 !important;
        border:none !important;
        border-radius:14px !important;
        padding:12px 20px !important;
    }
    /* =========================================================
    PDF DOWNLOAD BUTTON
    ========================================================= */

    div[data-testid="stDownloadButton"] button {
        background:#3B2A63 !important;
        color:#F3EAFF !important;
        border:1px solid #7055B5 !important;
        border-radius:14px !important;
        font-weight:700 !important;
        font-size:15px !important;
        padding:12px 20px !important;
        box-shadow:0 4px 14px rgba(112,85,181,.25) !important;
    }

    div[data-testid="stDownloadButton"] button:hover {
        background:#463474 !important;
        color:#FFFFFF !important;
        border-color:#8066C5 !important;
        box-shadow:0 5px 16px rgba(112,85,181,.35) !important;
        transform:translateY(-1px) !important;
    }

    div[data-testid="stDownloadButton"] button:focus,
    div[data-testid="stDownloadButton"] button:focus-visible {
        outline:none !important;
        border-color:#8B6FD1 !important;
        box-shadow:
            0 0 0 2px rgba(112,85,181,.20),
            0 4px 14px rgba(112,85,181,.25) !important;
    }

    div[data-testid="stHorizontalBlock"]
    .stButton > button {
        background:
            rgba(17,24,39,.75) !important;

        color:#94a3b8 !important;

        border:
            1px solid
            rgba(255,255,255,.08) !important;

        border-radius:14px !important;

        font-weight:700 !important;

        padding:12px 16px !important;

        box-shadow:none !important;
    }

    div[data-testid="stHorizontalBlock"]
    .stButton > button:hover {
        background:
            linear-gradient(
                135deg,
                rgba(79,70,229,.25),
                rgba(6,182,212,.20)
            ) !important;

        color:#f8fafc !important;

        border-color:
            rgba(99,102,241,.55) !important;

        transform:
            translateY(-2px) !important;
    }

    /* =========================================================
       NAVIGATION — INACTIVE TABS
       ========================================================= */

    div[data-testid="stHorizontalBlock"]
    .stButton > button[kind="secondary"],
    div[data-testid="stHorizontalBlock"]
    .stButton > button[data-testid="stBaseButton-secondary"] {
        background:#17233A !important;
        color:#AFC0D8 !important;
        border:1px solid #2B3D5C !important;
        box-shadow:none !important;
    }

    div[data-testid="stHorizontalBlock"]
    .stButton > button[kind="secondary"]:hover,
    div[data-testid="stHorizontalBlock"]
    .stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background:#203252 !important;
        color:#E2E8F0 !important;
        border-color:#3B5278 !important;
        box-shadow:none !important;
        transform:translateY(-1px) !important;
    }

    /* =========================================================
       NAVIGATION — CURRENT / ACTIVE TAB
       ========================================================= */

    div[data-testid="stHorizontalBlock"]
    .stButton > button[kind="primary"],
    div[data-testid="stHorizontalBlock"]
    .stButton > button[data-testid="stBaseButton-primary"] {
        background:#3B2A63 !important;
        color:#E9DFFF !important;
        border:1px solid #7055B5 !important;
        box-shadow:0 4px 14px rgba(112,85,181,.25) !important;
    }

    div[data-testid="stHorizontalBlock"]
    .stButton > button[kind="primary"]:hover,
    div[data-testid="stHorizontalBlock"]
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background:#463474 !important;
        color:#F3EAFF !important;
        border-color:#8066C5 !important;
        box-shadow:0 5px 16px rgba(112,85,181,.30) !important;
        transform:translateY(-1px) !important;
    }

    /* Active tab gets a slightly stronger visual focus */
    div[data-testid="stHorizontalBlock"]
    .stButton > button[kind="primary"]:focus,
    div[data-testid="stHorizontalBlock"]
    .stButton > button[data-testid="stBaseButton-primary"]:focus,
    div[data-testid="stHorizontalBlock"]
    .stButton > button[kind="primary"]:focus-visible,
    div[data-testid="stHorizontalBlock"]
    .stButton > button[data-testid="stBaseButton-primary"]:focus-visible {
        outline:none !important;
        border-color:#8B6FD1 !important;
        box-shadow:
            0 0 0 2px rgba(112,85,181,.20),
            0 4px 14px rgba(112,85,181,.25) !important;
    }

    .section-title {
        font-size:20px;
        font-weight:700;
        color:#f8fafc;
        margin-top:28px;
        margin-bottom:18px;
    }

    .risk-low {
        background:
            rgba(16,185,129,.15);
        color:#34d399;
        border:
            1px solid
            rgba(52,211,153,.4);
    }

    .risk-medium {
        background:
            rgba(245,158,11,.15);
        color:#fbbf24;
        border:
            1px solid
            rgba(251,191,36,.4);
    }

    .risk-high {
        background:
            rgba(239,68,68,.15);
        color:#f87171;
        border:
            1px solid
            rgba(248,113,113,.4);
    }

    .streamlit-expanderHeader {
        background:#111827 !important;
        border-radius:12px !important;
        color:#f1f5f9 !important;
    }

    
    /* ========================================================
       ANIMATED AI VALIDATION LOGO
       ======================================================== */

    .header-logo-container {
        display:flex;
        align-items:center;
        gap:18px;
        margin:8px 0 18px 0;
    }

    .animated-logo-wrapper {
        width:82px;
        height:82px;
        min-width:82px;
        display:flex;
        align-items:center;
        justify-content:center;
        position:relative;
    }

    .animated-ai-logo {
        width:82px;
        height:82px;
        overflow:visible;
        filter:
            drop-shadow(0 0 6px rgba(56,189,248,.45))
            drop-shadow(0 0 14px rgba(99,102,241,.25));
        animation:logoFloat 4s ease-in-out infinite;
    }

    @keyframes logoFloat {
        0%,100% {
            transform:translateY(0) scale(1);
        }
        50% {
            transform:translateY(-3px) scale(1.025);
        }
    }

    .logo-rotate {
        animation:rotateLogoRing 12s linear infinite;
        transform-box:fill-box;
        transform-origin:center;
    }

    @keyframes rotateLogoRing {
        from { transform:rotate(0deg); }
        to { transform:rotate(360deg); }
    }

    .scan-line {
        animation:scanAI 2.8s ease-in-out infinite;
        transform-box:fill-box;
        transform-origin:center;
    }

    @keyframes scanAI {
        0% {
            transform:translateY(-15px);
            opacity:0;
        }
        15% { opacity:1; }
        50% {
            transform:translateY(30px);
            opacity:1;
        }
        85% { opacity:1; }
        100% {
            transform:translateY(-15px);
            opacity:0;
        }
    }

    .node-pulse {
        animation:nodeGlow 2s ease-in-out infinite;
    }

    @keyframes nodeGlow {
        0%,100% {
            opacity:.45;
            filter:drop-shadow(0 0 2px currentColor);
        }
        50% {
            opacity:1;
            filter:drop-shadow(0 0 8px currentColor);
        }
    }

    .circuit-particles circle {
        animation:particlePulse 3s ease-in-out infinite;
        transform-box:fill-box;
        transform-origin:center;
    }

    .circuit-particles circle:nth-child(2) {
        animation-delay:.7s;
    }

    .circuit-particles circle:nth-child(3) {
        animation-delay:1.4s;
    }

    .circuit-particles circle:nth-child(4) {
        animation-delay:2.1s;
    }

    @keyframes particlePulse {
        0%,100% {
            opacity:.3;
            transform:scale(.8);
        }
        50% {
            opacity:1;
            transform:scale(1.35);
        }
    }

    .header-title-text {
        margin:0 !important;
        padding:0 !important;
        font-size:2rem !important;
        line-height:1.2 !important;
        font-weight:800 !important;
        letter-spacing:-.02em !important;
        color:#f8fafc !important;
    }

    .header-subtitle-text {
        color:#94a3b8;
        font-size:14px;
        font-weight:500;
        margin-top:5px;
    }

    @media (prefers-reduced-motion: reduce) {
        .animated-ai-logo,
        .logo-rotate,
        .scan-line,
        .node-pulse,
        .circuit-particles circle {
            animation:none !important;
        }
    }


/* ============================================================
   HISTORY PAGE POLISH
   ============================================================ */

/* Dark search input */
.stTextInput input {
    background: #111827 !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    min-height: 46px !important;
    padding: 0 14px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.stTextInput input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}

.stTextInput input::placeholder {
    color: #64748B !important;
}

/* Purple selected radio instead of Streamlit red */
.stRadio [role="radiogroup"] label input {
    accent-color: #6366F1 !important;
}

/* History KPI cards */
.history-kpi {
    background: linear-gradient(145deg, rgba(17,24,39,0.95), rgba(15,23,42,0.98));
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 14px;
    padding: 14px 16px;
    min-height: 82px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.16);
}

.history-kpi-value {
    color: #F8FAFC;
    font-size: 24px;
    font-weight: 800;
    line-height: 1.1;
}

.history-kpi-label {
    color: #94A3B8;
    font-size: 11px;
    font-weight: 600;
    margin-top: 5px;
}

.history-section-label {
    color: #CBD5E1;
    font-size: 14px;
    font-weight: 700;
    margin: 18px 0 8px 0;
}

</style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PROFESSIONAL HEADER + LEFT NAVIGATION
# ============================================================

# Global UI animation + sidebar styling.  The evaluation/backend
# logic below is intentionally unchanged.
st.markdown(
    """
    <style>
    /* --------------------------------------------------------
       PROFESSIONAL APP SHELL
       -------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090d18 0%, #0d1222 55%, #0a0f1b 100%);
        border-right: 1px solid #252d4a;
    }

    [data-testid="stSidebarContent"] {
        padding: 1.2rem 0.9rem 1.5rem 0.9rem;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 48px;
        margin: 5px 0;
        border-radius: 12px;
        border: 1px solid transparent;
        background: rgba(17, 24, 39, 0.70);
        color: #cbd5e1;
        text-align: left;
        padding: 0 14px;
        font-weight: 600;
        transition: all .22s ease;
        position: relative;
        overflow: hidden;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateX(4px);
        background: rgba(79, 70, 229, .14);
        border-color: rgba(129, 140, 248, .35);
        color: #f8fafc;
        box-shadow: 0 8px 24px rgba(79, 70, 229, .16);
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, rgba(99,102,241,.28), rgba(168,85,247,.20));
        border: 1px solid rgba(129,140,248,.48);
        color: #ffffff;
        box-shadow: 0 8px 26px rgba(99,102,241,.18);
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"]::before,
    [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]::before {
        content: "";
        position: absolute;
        left: 0;
        top: 9px;
        bottom: 9px;
        width: 3px;
        border-radius: 0 5px 5px 0;
        background: #a78bfa;
        box-shadow: 0 0 12px rgba(167,139,250,.8);
    }

    /* Stagger the navigation entrance. */
    [data-testid="stSidebar"] .stButton:nth-of-type(1) { animation: navIn .45s ease both .05s; }
    [data-testid="stSidebar"] .stButton:nth-of-type(2) { animation: navIn .45s ease both .11s; }
    [data-testid="stSidebar"] .stButton:nth-of-type(3) { animation: navIn .45s ease both .17s; }
    [data-testid="stSidebar"] .stButton:nth-of-type(4) { animation: navIn .45s ease both .23s; }

    @keyframes navIn {
        from { opacity: 0; transform: translateX(-14px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Header reveal */
    .app-header {
        width: 100%;
        min-height: 190px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 34px;
        padding: 12px 18px 18px 18px;
        box-sizing: border-box;
        animation: headerReveal .65s ease both;
    }

    @keyframes headerReveal {
        from { opacity: 0; transform: translateY(-14px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .app-logo {
        width: 180px;
        height: 180px;
        flex: 0 0 180px;
        overflow: visible;
        animation: logoFloat 4s ease-in-out infinite, logoGlow 3s ease-in-out infinite;
    }

    @keyframes logoFloat {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-7px) scale(1.025); }
    }

    @keyframes logoGlow {
        0%, 100% { filter: drop-shadow(0 0 10px rgba(56,189,248,.35)) drop-shadow(0 0 18px rgba(99,102,241,.18)); }
        50% { filter: drop-shadow(0 0 18px rgba(56,189,248,.55)) drop-shadow(0 0 30px rgba(168,85,247,.25)); }
    }

    .logo-rings {
        transform-origin: 80px 80px;
        animation: ringSpin 13s linear infinite;
    }

    @keyframes ringSpin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .logo-scan {
        animation: scanMove 2.7s ease-in-out infinite;
    }

    @keyframes scanMove {
        0%, 100% { transform: translateY(-13px); opacity: 0; }
        18% { opacity: 1; }
        50% { transform: translateY(28px); opacity: 1; }
        82% { opacity: 1; }
    }

    .logo-node {
        animation: nodePulse 2s ease-in-out infinite;
    }

    .logo-node:nth-of-type(2) { animation-delay: .35s; }
    .logo-node:nth-of-type(3) { animation-delay: .7s; }
    .logo-node:nth-of-type(4) { animation-delay: 1.05s; }

    @keyframes nodePulse {
        0%, 100% { opacity: .35; transform: scale(.82); }
        50% { opacity: 1; transform: scale(1.22); }
    }

    .header-copy {
        text-align: center;
        animation: copyReveal .75s ease both .12s;
    }

    @keyframes copyReveal {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .app-title {
        color: #f8fafc;
        font-size: 40px;
        line-height: 1.12;
        font-weight: 850;
        letter-spacing: -.025em;
        margin: 0;
    }

    .app-subtitle {
        color: #94a3b8;
        font-size: 15px;
        font-weight: 500;
        margin-top: 10px;
    }

    .system-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-top: 16px;
        padding: 7px 12px;
        border: 1px solid rgba(52,211,153,.25);
        border-radius: 999px;
        background: rgba(16,185,129,.07);
        color: #a7f3d0;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .02em;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 0 rgba(52,211,153,.7);
        animation: statusPulse 1.8s infinite;
    }

    @keyframes statusPulse {
        0% { box-shadow: 0 0 0 0 rgba(52,211,153,.65); }
        70% { box-shadow: 0 0 0 8px rgba(52,211,153,0); }
        100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
    }

    .sidebar-brand {
        padding: 5px 6px 20px 6px;
        animation: copyReveal .55s ease both;
    }

    .sidebar-brand-title {
        color: #f8fafc;
        font-size: 16px;
        font-weight: 800;
        letter-spacing: -.01em;
    }

    .sidebar-brand-subtitle {
        color: #64748b;
        font-size: 11px;
        margin-top: 4px;
    }

    .sidebar-section-label {
        color: #64748b;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: 8px 6px 7px 6px;
    }

    .sidebar-divider {
        height: 1px;
        background: #252d4a;
        margin: 18px 6px;
    }

    .sidebar-status {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #94a3b8;
        font-size: 12px;
        padding: 0 6px;
    }

    .sidebar-status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #34d399;
        animation: statusPulse 1.8s infinite;
    }

    /* Subtle page and interaction animations. */
    .main .block-container {
        animation: pageReveal .55s ease both;
    }

    @keyframes pageReveal {
        from { opacity: 0; transform: translateY(7px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stButton > button,
    [data-testid="stDownloadButton"] button {
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease !important;
    }

    .stButton > button:active,
    [data-testid="stDownloadButton"] button:active {
        transform: scale(.98) !important;
    }

    [data-testid="stExpander"] {
        transition: border-color .2s ease, box-shadow .2s ease;
    }

    [data-testid="stExpander"]:hover {
        border-color: rgba(99,102,241,.35) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
    }

    /* Make normal Streamlit content feel slightly more polished. */
    div[data-testid="stMetric"] {
        animation: cardReveal .45s ease both;
    }

    @keyframes cardReveal {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 900px) {
        .app-header { gap: 18px; min-height: 160px; }
        .app-logo { width: 145px; height: 145px; flex-basis: 145px; }
        .app-title { font-size: 31px; }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER — LOGO STAYS OUTSIDE THE SIDEBAR
# ============================================================

components.html(
    """
    <style>
        html, body {
            margin: 0;
            padding: 0;
            background: transparent;
            overflow: hidden;
        }

        .app-header {
            width: 100%;
            height: 205px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 30px;
            box-sizing: border-box;
            padding: 8px 16px 14px;
            color: #f8fafc;
            font-family: Arial, sans-serif;
            animation: headerReveal .65s ease both;
        }

        .app-logo {
            width: 180px;
            height: 180px;
            flex: 0 0 180px;
            overflow: visible;
            transform-origin: center;
            animation: logoFloat 4s ease-in-out infinite, logoGlow 3s ease-in-out infinite;
        }

        .logo-rings {
            transform-box: fill-box;
            transform-origin: center;
            animation: ringSpin 13s linear infinite;
        }

        .logo-scan {
            transform-box: fill-box;
            transform-origin: center;
            animation: scanMove 2.7s ease-in-out infinite;
        }

        .logo-node {
            transform-box: fill-box;
            transform-origin: center;
            animation: nodePulse 2s ease-in-out infinite;
        }

        .logo-node:nth-of-type(2) { animation-delay: .35s; }
        .logo-node:nth-of-type(3) { animation-delay: .7s; }
        .logo-node:nth-of-type(4) { animation-delay: 1.05s; }

        .header-copy {
            text-align: center;
            max-width: 760px;
            animation: copyReveal .75s ease both .12s;
        }

        .app-title {
            margin: 0;
            color: #f8fafc;
            font-size: 40px;
            line-height: 1.12;
            font-weight: 800;
            letter-spacing: -.025em;
        }

        .app-subtitle {
            margin-top: 10px;
            color: #94a3b8;
            font-size: 15px;
            font-weight: 500;
        }

        .system-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 15px;
            padding: 7px 12px;
            border: 1px solid rgba(52,211,153,.25);
            border-radius: 999px;
            background: rgba(16,185,129,.07);
            color: #a7f3d0;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: .02em;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #34d399;
            animation: statusPulse 1.8s infinite;
        }

        @keyframes headerReveal {
            from { opacity: 0; transform: translateY(-14px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes copyReveal {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes logoFloat {
            0%, 100% { transform: translateY(0) scale(1); }
            50% { transform: translateY(-7px) scale(1.025); }
        }
        @keyframes logoGlow {
            0%, 100% { filter: drop-shadow(0 0 10px rgba(56,189,248,.35)) drop-shadow(0 0 18px rgba(99,102,241,.18)); }
            50% { filter: drop-shadow(0 0 18px rgba(56,189,248,.55)) drop-shadow(0 0 30px rgba(168,85,247,.25)); }
        }
        @keyframes ringSpin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        @keyframes scanMove {
            0%, 100% { transform: translateY(-13px); opacity: 0; }
            18% { opacity: 1; }
            50% { transform: translateY(28px); opacity: 1; }
            82% { opacity: 1; }
        }
        @keyframes nodePulse {
            0%, 100% { opacity: .35; transform: scale(.82); }
            50% { opacity: 1; transform: scale(1.22); }
        }
        @keyframes statusPulse {
            0% { box-shadow: 0 0 0 0 rgba(52,211,153,.65); }
            70% { box-shadow: 0 0 0 8px rgba(52,211,153,0); }
            100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
        }

        @media (max-width: 900px) {
            .app-header { gap: 16px; height: 175px; }
            .app-logo { width: 145px; height: 145px; flex-basis: 145px; }
            .app-title { font-size: 31px; }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: .01ms !important;
                animation-iteration-count: 1 !important;
            }
        }
    </style>
    <div class="app-header">
        <svg class="app-logo" viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="mainLogoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#38bdf8"/>
                    <stop offset="50%" stop-color="#6366f1"/>
                    <stop offset="100%" stop-color="#a855f7"/>
                </linearGradient>
                <linearGradient id="mainScanGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="transparent"/>
                    <stop offset="50%" stop-color="#22d3ee"/>
                    <stop offset="100%" stop-color="transparent"/>
                </linearGradient>
                <filter id="mainLogoGlow" x="-100%" y="-100%" width="300%" height="300%">
                    <feGaussianBlur stdDeviation="3" result="blur"/>
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
            </defs>

            <g class="logo-rings">
                <circle cx="80" cy="80" r="70" fill="none" stroke="url(#mainLogoGradient)" stroke-width="2.5" stroke-dasharray="28 12 5 12" opacity=".85"/>
                <circle cx="80" cy="80" r="64" fill="none" stroke="#38bdf8" stroke-width="1.2" stroke-dasharray="4 8" opacity=".38"/>
            </g>

            <g stroke="url(#mainLogoGradient)" stroke-width="2" fill="none" opacity=".8">
                <path d="M32 67 H18 V55"/>
                <path d="M118 61 H140 V42"/>
                <path d="M38 96 H22 V117"/>
                <path d="M118 92 H138 V119"/>
            </g>

            <circle cx="18" cy="55" r="3" fill="#22d3ee" class="logo-node"/>
            <circle cx="140" cy="42" r="3" fill="#a855f7" class="logo-node"/>
            <circle cx="22" cy="117" r="3" fill="#38bdf8" class="logo-node"/>
            <circle cx="138" cy="119" r="3" fill="#818cf8" class="logo-node"/>

            <g filter="url(#mainLogoGlow)">
                <circle cx="70" cy="67" r="38" fill="#0f172a" stroke="url(#mainLogoGradient)" stroke-width="6"/>
                <circle cx="70" cy="67" r="31" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5" opacity=".8"/>
                <text x="70" y="76" text-anchor="middle" font-family="Arial, sans-serif" font-size="29" font-weight="800" fill="#f8fafc">AI</text>
                <rect class="logo-scan" x="35" y="45" width="70" height="3" rx="2" fill="url(#mainScanGradient)"/>
                <path d="M45 52 L45 47 L51 47 M89 47 L95 47 L95 52 M45 82 L45 87 L51 87 M89 87 L95 87 L95 82" stroke="#22d3ee" stroke-width="2" fill="none"/>
                <rect x="98" y="96" width="12" height="38" rx="6" transform="rotate(-42 104 96)" fill="url(#mainLogoGradient)"/>
                <rect x="101" y="100" width="5" height="28" rx="2.5" transform="rotate(-42 104 100)" fill="#7dd3fc" opacity=".65"/>
            </g>
        </svg>

        <div class="header-copy">
            <div class="app-title">AI Response Validation System</div>
            <div class="app-subtitle">Agent Workspace · Independent validation agents · Weighted final verdict</div>
            <div class="system-status"><span class="status-dot"></span> Validation System Online</div>
        </div>
    </div>
    """,
    height=205,
    scrolling=False,
)

# ============================================================
# LEFT SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">Validation Console</div>
            <div class="sidebar-brand-subtitle">Multi-agent evaluation workspace</div>
        </div>
        <div class="sidebar-section-label">Navigation</div>
        """,
        unsafe_allow_html=True,
    )

    nav_items = [
        ("🏠  Dashboard", "Dashboard"),
        ("🔍  Single Evaluation", "Single Evaluation"),
        ("📊  Batch Evaluation", "Batch Evaluation"),
        ("🕘  History", "History"),
    ]

    for label, value in nav_items:
        is_active = st.session_state.page == value
        button_label = f"●  {label}" if is_active else label

        if st.button(
            button_label,
            key=f"nav_{value}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = value
            st.rerun()

    st.markdown(
        """
        <div class="sidebar-divider"></div>
        <div class="sidebar-status">
            <span class="sidebar-status-dot"></span>
            <span>System online · Agents ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

page = st.session_state.page



def generate_batch_pdf(results_df):
    """Generate a detailed batch PDF with per-record agent reasoning."""
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BatchTitle", parent=styles["Title"],
        alignment=TA_CENTER, fontSize=18, spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "BatchHeading", parent=styles["Heading2"],
        fontSize=14, spaceBefore=12, spaceAfter=8,
    )
    subheading_style = ParagraphStyle(
        "BatchSubHeading", parent=styles["Heading3"],
        fontSize=11, spaceBefore=8, spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BatchBody", parent=styles["BodyText"],
        fontSize=9, leading=12, spaceAfter=5,
    )

    story = []

    def safe_text(value):
        text_value = "" if value is None else str(value)
        return (text_value.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>"))

    def add_label(label, value):
        story.append(Paragraph(
            f"<b>{safe_text(label)}</b> {safe_text(value)}",
            body_style,
        ))

    def result(row, key):
        return json_value(row.get(key, {}))

    total = len(results_df)
    success = results_df[results_df["Status"] == "Success"]
    failed = results_df[results_df["Status"].astype(str).str.startswith("Failed")]
    skipped = results_df[results_df["Status"].astype(str).str.startswith("Skipped")]
    invalid = results_df[results_df["Status"].astype(str).str.startswith("Invalid")]

    story.append(Paragraph("AI Response Validation System", title_style))
    story.append(Paragraph("Detailed Batch Evaluation Report", heading_style))
    story.append(Paragraph(
        "Detailed individual evaluation results generated from the batch validation pipeline.",
        body_style,
    ))

    story.append(Paragraph("Batch Statistics", heading_style))
    stats = [
        ["Metric", "Value"],
        ["Total Records", str(total)],
        ["Successful", str(len(success))],
        ["Failed", str(len(failed))],
        ["Skipped", str(len(skipped))],
        ["Invalid Input", str(len(invalid))],
    ]
    table = Table(stats, colWidths=[260, 120])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(table)

    if not success.empty:
        story.append(Paragraph("Average Agent Scores", heading_style))
        avg = [
            ["Agent", "Average Score"],
            ["Relevance", f"{success['Relevance'].mean():.2f}/5"],
            ["Accuracy", f"{success['Accuracy'].mean():.2f}/5"],
            ["Completeness", f"{success['Completeness'].mean():.2f}/5"],
            ["Hallucination", f"{success['Hallucination'].mean():.2f}/5"],
        ]
        table = Table(avg, colWidths=[260, 120])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (1,1), (-1,-1), "CENTER"),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(table)

    story.append(Paragraph("Detailed Individual Record Analysis", heading_style))

    for _, row in results_df.iterrows():
        response_id = row.get("Response ID", "Unknown")
        status = str(row.get("Status", "Unknown"))
        story.append(Paragraph(f"Record {safe_text(response_id)}", heading_style))
        add_label("Status:", status)
        add_label("Question:", row.get("Question", ""))
        add_label("AI Response:", row.get("AI Response", "") or "No AI response available.")
        add_label("Reference Answer:", row.get("Reference Answer", "") or "Not provided")

        if status != "Success":
            story.append(Paragraph("Evaluation could not be completed.", subheading_style))
            add_label("Error:", row.get("Error", "No error details available."))
            story.append(Spacer(1, 10))
            continue

        relevance = result(row, "Relevance Result")
        accuracy = result(row, "Accuracy Result")
        completeness = result(row, "Completeness Result")
        hallucination = result(row, "Hallucination Result")
        verdict = result(row, "Verdict Result")

        score = float(row.get("Overall Score", 0) or 0)
        verdict_name = str(verdict.get("verdict", row.get("Verdict", "Unknown")))
        risk = str(row.get("Risk Level", "Unknown") or "Unknown")

        story.append(Paragraph("Final Evaluation Verdict", subheading_style))
        final = Table([
            ["System Verdict", "Weighted Score", "Hallucination Risk"],
            [verdict_name, f"{score:.1f} / 5.0", risk],
        ], colWidths=[125, 125, 130])
        final.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(final)

        story.append(Paragraph("Verdict Summary", subheading_style))
        summary = (verdict.get("summary") or verdict.get("explanation") or
                   verdict.get("reasoning") or verdict.get("verdict_explanation") or
                   verdict.get("analysis"))
        if summary:
            story.append(Paragraph(safe_text(summary), body_style))
        else:
            parts = []
            for label, agent in [
                ("Accuracy", accuracy),
                ("Completeness", completeness),
                ("Relevance", relevance),
                ("Hallucination", hallucination),
            ]:
                if agent.get("reasoning"):
                    parts.append(f"<b>{label}:</b> {safe_text(agent['reasoning'])}")
            story.append(Paragraph(" ".join(parts) or "No verdict summary returned.", body_style))

        story.append(Paragraph("Agent Workspace", subheading_style))
        agent_table = Table([
            ["Agent", "Score"],
            ["Relevance", f"{float(relevance.get('score',0) or 0):.1f}/5"],
            ["Accuracy", f"{float(accuracy.get('score',0) or 0):.1f}/5"],
            ["Completeness", f"{float(completeness.get('score',0) or 0):.1f}/5"],
            ["Hallucination Detection", f"{float(hallucination.get('score',0) or 0):.1f}/5"],
        ], colWidths=[260, 120])
        agent_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (1,1), (-1,-1), "CENTER"),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(agent_table)

        for title, agent in [
            ("Relevance Agent", relevance),
            ("Accuracy Agent", accuracy),
            ("Completeness Agent", completeness),
            ("Hallucination Detection Agent", hallucination),
        ]:
            story.append(Paragraph(title, subheading_style))
            add_label("Reasoning:", agent.get("reasoning", "No reasoning returned."))

            evidence = agent.get("supporting_evidence", []) or []
            if evidence:
                story.append(Paragraph("<b>Supporting Evidence</b>", body_style))
                for item in evidence:
                    story.append(Paragraph("• " + safe_text(item), body_style))

            if title == "Completeness Agent":
                missing = agent.get("missing_components", []) or []
                if missing:
                    story.append(Paragraph("<b>Missing Components</b>", body_style))
                    for item in missing:
                        story.append(Paragraph("• " + safe_text(item), body_style))
                else:
                    story.append(Paragraph("<b>Missing Components:</b> None identified.", body_style))

            if title == "Hallucination Detection Agent":
                claims = agent.get("flagged_claims", []) or []
                if claims:
                    story.append(Paragraph(f"<b>{len(claims)} flagged claim(s) detected.</b>", body_style))
                    for i, claim in enumerate(claims, 1):
                        if isinstance(claim, dict):
                            claim_text = claim.get("claim", "Not available.")
                            classification = claim.get("classification", "Flagged")
                            reason = claim.get("reason", "No additional explanation returned.")
                        else:
                            claim_text = claim
                            classification = "Flagged"
                            reason = "No additional explanation returned."
                        story.append(Paragraph(
                            f"<b>Flagged Claim #{i} — {safe_text(classification)}</b>",
                            body_style,
                        ))
                        add_label("Claim:", claim_text)
                        add_label("Reason:", reason)
                else:
                    story.append(Paragraph("<b>Flagged Claims:</b> None detected.", body_style))

        story.append(Paragraph("Verdict Agent Analysis", subheading_style))
        verdict_reasoning = (verdict.get("reasoning") or verdict.get("explanation") or
                             verdict.get("summary") or verdict.get("verdict_explanation") or
                             verdict.get("analysis"))
        story.append(Paragraph(
            safe_text(verdict_reasoning or "No separate Verdict Agent reasoning was returned."),
            body_style,
        ))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def render_batch_results(batch_results_data):
    """Render polished batch summary, compact results table, and detailed reasoning."""
    if not batch_results_data:
        return

    results_df = pd.DataFrame(batch_results_data)
    if results_df.empty:
        return

    total_rows = len(results_df)
    successful_results = results_df[results_df["Status"] == "Success"]
    failed_results = results_df[results_df["Status"].astype(str).str.startswith("Failed")]
    skipped_results = results_df[results_df["Status"].astype(str).str.startswith("Skipped")]
    invalid_results = results_df[results_df["Status"].astype(str).str.startswith("Invalid")]

    # =========================================================
    # BATCH RESULTS HEADER
    # =========================================================
    st.markdown(
        '<div class="section-title">📋 Batch Results</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "A compact overview is shown below. Expand an individual record for the complete multi-agent reasoning."
    )

    # =========================================================
    # BATCH SUMMARY CARDS
    # =========================================================
    st.markdown('<div class="batch-summary-title">📊 Batch Summary</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    pass_count = int((successful_results["Verdict"] == "Pass").sum()) if not successful_results.empty else 0
    pass_rate = (pass_count / len(successful_results) * 100) if len(successful_results) else 0

    s1.metric("Total Records", total_rows)
    s2.metric("Successful", len(successful_results))
    s3.metric("Failed / Issues", len(failed_results) + len(skipped_results) + len(invalid_results))
    s4.metric("Pass Rate", f"{pass_rate:.1f}%")

    # =========================================================
    # COMPACT RESULTS TABLE
    # =========================================================
    st.markdown('<div class="batch-summary-title">📋 Evaluation Overview</div>', unsafe_allow_html=True)

    table_rows = []
    for _, record in results_df.iterrows():
        status = str(record.get("Status", "Unknown"))
        verdict = json_value(record.get("Verdict Result", {}))
        verdict_name = str(verdict.get("verdict", record.get("Verdict", "Unknown")))
        score = float(record.get("Overall Score", 0) or 0)
        question = str(record.get("Question", ""))
        if len(question) > 70:
            question = question[:67] + "..."
        table_rows.append({
            "ID": record.get("Response ID", ""),
            "Question": question,
            "Score": f"{score:.1f}/5" if status == "Success" else "—",
            "Verdict": verdict_name if status == "Success" else status.replace("Invalid Input", "Invalid"),
            "Risk": str(record.get("Risk Level", "Unknown") or "Unknown"),
            "Status": status,
        })

    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID", width="small"),
            "Question": st.column_config.TextColumn("Question", width="large"),
            "Score": st.column_config.TextColumn("Score", width="small"),
            "Verdict": st.column_config.TextColumn("Verdict", width="medium"),
            "Risk": st.column_config.TextColumn("Risk", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
        },
    )

    # =========================================================
    # DOWNLOADS
    # =========================================================
    download_col1, download_col2 = st.columns(2)

    display_columns = [
        "Response ID", "Question", "Relevance", "Accuracy",
        "Completeness", "Hallucination", "Overall Score",
        "Verdict", "Risk Level", "Status",
    ]

    csv_data = results_df[display_columns].to_csv(index=False).encode("utf-8-sig")

    with download_col1:
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name="batch_evaluation_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with download_col2:
        pdf_data = generate_batch_pdf(results_df)
        st.download_button(
            label="📄 Download Detailed PDF",
            data=pdf_data,
            file_name="batch_evaluation_detailed_report.pdf",
            mime="application/pdf",
            key="batch_pdf_download",
            type="primary",
            use_container_width=True,
        )

    # =========================================================
    # AVERAGE AGENT SCORES
    # =========================================================
    if not successful_results.empty:
        st.markdown('<div class="batch-summary-title">📈 Average Agent Scores</div>', unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Relevance", f"{successful_results['Relevance'].mean():.2f}/5")
        a2.metric("Accuracy", f"{successful_results['Accuracy'].mean():.2f}/5")
        a3.metric("Completeness", f"{successful_results['Completeness'].mean():.2f}/5")
        a4.metric("Hallucination", f"{successful_results['Hallucination'].mean():.2f}/5")

        st.markdown('<div class="batch-summary-title">⚖️ Verdict Distribution</div>', unsafe_allow_html=True)
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Pass", int((successful_results["Verdict"] == "Pass").sum()))
        v2.metric("Needs Improvement", int((successful_results["Verdict"] == "Needs Improvement").sum()))
        v3.metric("Fail", int((successful_results["Verdict"] == "Fail").sum()))
        v4.metric("Hallucination Detected", int((successful_results["Hallucination"] < 5).sum()))

    # =========================================================
    # FAILED / SKIPPED / INVALID RECORDS
    # =========================================================
    if not failed_results.empty or not skipped_results.empty or not invalid_results.empty:
        st.markdown("### ⚠️ Failed / Skipped / Invalid Records")

        problem_mask = results_df["Status"].astype(str).str.startswith(("Failed", "Skipped", "Invalid"))
        problem_df = results_df.loc[problem_mask, ["Response ID", "Question", "Status", "Error"]].copy()

        for _, problem in problem_df.iterrows():
            response_id = problem["Response ID"]
            status = str(problem["Status"])
            error = str(problem["Error"] or "No error details available.")

            with st.expander(f"❌ Record {response_id} • {status}"):
                st.markdown("**Question**")
                st.write(problem["Question"])
                st.markdown("**Status**")
                st.write(status)
                st.markdown("**Error Details**")
                st.error(error)

    # =========================================================
    # DETAILED INDIVIDUAL ANALYSIS
    # =========================================================
    st.markdown("### 🔎 Detailed Individual Record Analysis")
    st.caption(
        "Each successful record is shown with the same detailed reasoning structure used in Single Evaluation."
    )

    for _, record in results_df.iterrows():
        status = str(record.get("Status", "Unknown"))
        response_id = record.get("Response ID", "Unknown")
        question = record.get("Question", "")
        ai_response = record.get("AI Response", "") or "No AI response available."
        reference_answer = record.get("Reference Answer", "") or "Not provided"

        if status != "Success":
            continue

        verdict = json_value(record.get("Verdict Result", {}))
        relevance = json_value(record.get("Relevance Result", {}))
        accuracy = json_value(record.get("Accuracy Result", {}))
        completeness = json_value(record.get("Completeness Result", {}))
        hallucination = json_value(record.get("Hallucination Result", {}))

        verdict_name = str(verdict.get("verdict", record.get("Verdict", "Unknown")))
        score = float(record.get("Overall Score", 0) or 0)
        risk = str(record.get("Risk Level", "Unknown") or "Unknown")

        with st.expander(f"📄 Record {response_id} • {verdict_name} • {score:.1f}/5 • {risk} Risk"):
            st.markdown("### 📌 Evaluation Input")
            st.markdown("**Question**")
            st.info(question)
            st.markdown("**AI Response**")
            st.info(ai_response)
            st.markdown("**Reference Answer**")
            st.info(reference_answer)

            st.divider()
            st.markdown("### ⚖️ Final Evaluation Verdict")
            st.caption("The final verdict is generated from the combined results of all validation agents.")

            m1, m2, m3 = st.columns(3)
            m1.metric("System Verdict", verdict_name)
            m2.metric("Weighted Score", f"{score:.1f} / 5.0")
            m3.metric("Hallucination Risk", risk)

            st.markdown("### 📝 Verdict Summary")
            verdict_summary = (
                verdict.get("summary") or verdict.get("explanation") or
                verdict.get("reasoning") or verdict.get("verdict_explanation") or
                verdict.get("analysis")
            )
            if verdict_summary:
                st.info(str(verdict_summary))
            else:
                parts = []
                for label, agent in [("Accuracy", accuracy), ("Completeness", completeness),
                                     ("Relevance", relevance), ("Hallucination", hallucination)]:
                    if agent.get("reasoning"):
                        parts.append(f"**{label}:** {agent['reasoning']}")
                st.info(" ".join(parts) or "No verdict summary returned.")

            st.divider()
            st.markdown("### 🧩 Agent Workspace")
            st.caption("Four independent validation agents analyze the response before the Verdict Agent produces the final decision.")

            cols = st.columns(4)
            for col, title, agent in zip(
                cols,
                ["🎯 Relevance", "✓ Accuracy", "◈ Completeness", "⚠️ Hallucination"],
                [relevance, accuracy, completeness, hallucination],
            ):
                with col:
                    agent_score = float(agent.get("score", 0) or 0)
                    st.markdown(f"#### {title}")
                    st.caption("Independent validation agent")
                    st.metric("Score", f"{agent_score:.1f} / 5.0")
                    st.progress(max(0.0, min(1.0, agent_score / 5.0)))
                    st.success("Good" if agent_score >= 4 else "Moderate" if agent_score >= 2.5 else "Low")

            st.markdown("### 🔎 Agent Details")

            with st.expander("🎯 Relevance Agent", expanded=True):
                st.markdown("**Reasoning**")
                st.write(relevance.get("reasoning", "No reasoning returned."))

            with st.expander("✓ Accuracy Agent", expanded=True):
                st.markdown("**Reasoning**")
                st.write(accuracy.get("reasoning", "No reasoning returned."))
                evidence = accuracy.get("supporting_evidence", []) or []
                if evidence:
                    st.markdown("**Supporting Evidence**")
                    for item in evidence:
                        st.info(str(item))
                else:
                    st.caption("No supporting evidence returned.")

            with st.expander("◈ Completeness Agent", expanded=True):
                st.markdown("**Reasoning**")
                st.write(completeness.get("reasoning", "No reasoning returned."))
                missing = completeness.get("missing_components", []) or []
                if missing:
                    st.markdown("**Missing Components**")
                    for item in missing:
                        st.warning(str(item))
                else:
                    st.success("No missing components identified.")

            with st.expander("⚠️ Hallucination Detection Agent", expanded=True):
                st.markdown("**Reasoning**")
                st.write(hallucination.get("reasoning", "No reasoning returned."))
                claims = hallucination.get("flagged_claims", []) or []
                if claims:
                    st.warning(f"{len(claims)} flagged claim(s) detected.")
                    for i, claim in enumerate(claims, 1):
                        if isinstance(claim, dict):
                            claim_text = claim.get("claim", "Not available.")
                            classification = claim.get("classification", "Flagged")
                            reason = claim.get("reason", "No additional explanation returned.")
                        else:
                            claim_text = str(claim)
                            classification = "Flagged"
                            reason = "No additional explanation returned."
                        st.markdown(f"**Flagged Claim #{i} — {classification}**")
                        st.write(f"**Claim:** {claim_text}")
                        st.write(f"**Reason:** {reason}")
                else:
                    st.success("No ungrounded or contradictory claims detected.")

            st.divider()
            st.markdown("### ⚖️ Verdict Agent Analysis")
            verdict_reasoning = (
                verdict.get("reasoning") or verdict.get("explanation") or
                verdict.get("summary") or verdict.get("verdict_explanation") or
                verdict.get("analysis")
            )
            st.info(str(verdict_reasoning or "No separate Verdict Agent reasoning was returned."))

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    singles = get_single_history()
    batches = get_batch_history()

    # ========================================================
    # BUILD A COMMON DATASET FROM STORED EVALUATIONS
    # ========================================================

    dashboard_records = []

    def build_dashboard_record(item, source_type, batch_id=None):

        relevance = json_value(
            item.get("relevance_result")
        )
        accuracy = json_value(
            item.get("accuracy_result")
        )
        completeness = json_value(
            item.get("completeness_result")
        )
        hallucination = json_value(
            item.get("hallucination_result")
        )
        verdict_data = json_value(
            item.get("verdict_result")
        )

        verdict_name = str(
            verdict_data.get(
                "verdict",
                verdict_data.get(
                    "status",
                    "Unknown",
                ),
            )
        ).strip()

        return {
            "submission_id": item.get("submission_id"),
            "question": item.get("question", ""),
            "ai_response": item.get("ai_response", ""),
            "reference_answer": item.get("reference_answer", ""),
            "source_document": item.get("source_document", ""),
            "timestamp": item.get("timestamp", ""),
            "status": item.get("status", ""),
            "source_type": source_type,
            "batch_id": batch_id,
            "batch_label": (
                f"Batch {batch_id}"
                if batch_id is not None
                else "Single Evaluation"
            ),
            "relevance": float(
                relevance.get("score", 0) or 0
            ),
            "accuracy": float(
                accuracy.get("score", 0) or 0
            ),
            "completeness": float(
                completeness.get("score", 0) or 0
            ),
            "hallucination": float(
                hallucination.get("score", 0) or 0
            ),
            "overall_score": float(
                item.get("overall_score", 0) or 0
            ),
            "verdict": verdict_name,
            "risk_level": str(
                item.get("risk_level", "Unknown")
                or "Unknown"
            ),
            "flagged_claims": hallucination.get(
                "flagged_claims", []
            ) or [],
            "missing_components": completeness.get(
                "missing_components", []
            ) or [],
            "relevance_reasoning": relevance.get(
                "reasoning", ""
            ),
            "accuracy_reasoning": accuracy.get(
                "reasoning", ""
            ),
            "completeness_reasoning": completeness.get(
                "reasoning", ""
            ),
            "hallucination_reasoning": hallucination.get(
                "reasoning", ""
            ),
        }

    for item in singles:
        dashboard_records.append(
            build_dashboard_record(
                item,
                "Single",
                None,
            )
        )

    for batch in batches:

        batch_id = batch.get("batch_id")

        if batch_id is None:
            continue

        for item in get_batch_records(batch_id):

            if str(
                item.get("status", "")
            ).strip().lower() != "success":
                continue

            dashboard_records.append(
                build_dashboard_record(
                    item,
                    "Batch",
                    batch_id,
                )
            )

    # ========================================================
    # PAGE TITLE
    # ========================================================

    st.markdown(
        f"""
        <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin:4px 0 22px 0;">
            <div>
                <div style="color:#FFFFFF;font-size:28px;font-weight:800;letter-spacing:-.4px;margin-bottom:7px;">📈 Evaluation Scoring Dashboard</div>
                <div style="color:#AAB2C8;font-size:14px;">Combined analysis of stored single and batch evaluation results.</div>
            </div>
            <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;border:1px solid rgba(16,191,163,.28);background:rgba(16,191,163,.07);border-radius:999px;white-space:nowrap;margin-top:2px;">
                <span style="width:8px;height:8px;border-radius:50%;background:#10BFA3;box-shadow:0 0 12px #10BFA3;display:inline-block;"></span>
                <span style="color:#A7F3D0;font-size:11px;font-weight:700;">LIVE DATA</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # FILTERS
    # ========================================================

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    verdict_options = [
        "All Verdicts",
        "Pass",
        "Needs Improvement",
        "Fail",
    ]

    batch_options = [
        "All Evaluations",
        "Single Evaluation",
    ]

    batch_ids = sorted(
        {
            int(x["batch_id"])
            for x in dashboard_records
            if x["batch_id"] is not None
        }
    )

    batch_options.extend(
        [f"Batch {batch_id}" for batch_id in batch_ids]
    )

    score_options = [
        "All Scores",
        "0 – 1.99",
        "2.0 – 3.49",
        "3.5 – 5.0",
    ]

    with filter_col1:
        st.markdown('<div style="color:#94A3B8;font-size:11px;font-weight:700;letter-spacing:.8px;margin:0 0 5px 2px;">VERDICT</div>', unsafe_allow_html=True)
        selected_verdict = st.selectbox(
            "Filter by Verdict",
            verdict_options,
            key="dashboard_verdict_filter",
            label_visibility="collapsed",
        )

    with filter_col2:
        st.markdown('<div style="color:#94A3B8;font-size:11px;font-weight:700;letter-spacing:.8px;margin:0 0 5px 2px;">EVALUATION BATCH</div>', unsafe_allow_html=True)
        selected_batch = st.selectbox(
            "Filter by Evaluation Batch",
            batch_options,
            key="dashboard_batch_filter",
            label_visibility="collapsed",
        )

    with filter_col3:
        st.markdown('<div style="color:#94A3B8;font-size:11px;font-weight:700;letter-spacing:.8px;margin:0 0 5px 2px;">OVERALL SCORE</div>', unsafe_allow_html=True)
        selected_score = st.selectbox(
            "Filter by Overall Score",
            score_options,
            key="dashboard_score_filter",
            label_visibility="collapsed",
        )

    # ========================================================
    # APPLY FILTERS
    # ========================================================

    filtered_records = dashboard_records[:]

    if selected_verdict != "All Verdicts":
        filtered_records = [
            x for x in filtered_records
            if x["verdict"].lower()
            == selected_verdict.lower()
        ]

    if selected_batch == "Single Evaluation":
        filtered_records = [
            x for x in filtered_records
            if x["source_type"] == "Single"
        ]

    elif selected_batch.startswith("Batch "):
        selected_batch_id = int(
            selected_batch.split(" ")[1]
        )
        filtered_records = [
            x for x in filtered_records
            if x["batch_id"] == selected_batch_id
        ]

    if selected_score == "0 – 1.99":
        filtered_records = [
            x for x in filtered_records
            if 0 <= x["overall_score"] < 2
        ]

    elif selected_score == "2.0 – 3.49":
        filtered_records = [
            x for x in filtered_records
            if 2 <= x["overall_score"] < 3.5
        ]

    elif selected_score == "3.5 – 5.0":
        filtered_records = [
            x for x in filtered_records
            if 3.5 <= x["overall_score"] <= 5
        ]

    # ========================================================
    # SUMMARY COUNTS
    # ========================================================

    total_evaluations = len(filtered_records)

    pass_count = sum(
        1
        for x in filtered_records
        if x["verdict"].lower() == "pass"
    )

    needs_count = sum(
        1
        for x in filtered_records
        if x["verdict"].lower()
        == "needs improvement"
    )

    fail_count = sum(
        1
        for x in filtered_records
        if x["verdict"].lower() == "fail"
    )

    def pct(value, total):
        if total == 0:
            return 0.0
        return value / total * 100

    # ========================================================
    # SUMMARY CARDS
    # ========================================================

    card_data = [
        ("Total Evaluations", total_evaluations, "100%", "#FFFFFF", "All stored evaluations"),
        ("Pass", pass_count, f"{pct(pass_count, total_evaluations):.1f}%", "#818CF8", "Passing responses"),
        ("Needs Improvement", needs_count, f"{pct(needs_count, total_evaluations):.1f}%", "#2DD4BF", "Responses needing review"),
        ("Fail", fail_count, f"{pct(fail_count, total_evaluations):.1f}%", "#FB6A56", "Responses requiring attention"),
    ]

    cols = st.columns(4)

    for index, (col, (label, value, percentage, value_color, subtext)) in enumerate(zip(cols, card_data)):
        with col:
            components.html(
                f"""
                <style>
                    .kpi-card{{position:relative;overflow:hidden;background:linear-gradient(145deg,#151B32 0%,#0D1222 72%);border:1px solid #252D4A;border-radius:16px;padding:18px 16px 15px;box-sizing:border-box;font-family:Arial,sans-serif;min-height:132px;animation:kpiIn .55s cubic-bezier(.22,1,.36,1) both;animation-delay:{index*.08:.2f}s;}}
                    .kpi-card:before{{content:'';position:absolute;inset:-80px auto auto -50px;width:150px;height:150px;background:{value_color};opacity:.08;filter:blur(28px);border-radius:50%;}}
                    .kpi-card:hover{{transform:translateY(-3px);border-color:{value_color}66;box-shadow:0 12px 30px rgba(0,0,0,.28),0 0 22px {value_color}12;}}
                    .kpi-label{{position:relative;color:#AAB2C8;font-size:12px;font-weight:700;letter-spacing:.2px;}}
                    .kpi-value{{position:relative;color:{value_color};font-size:31px;font-weight:800;line-height:1.1;margin-top:7px;}}
                    .kpi-percent{{position:relative;color:#E2E8F0;font-size:12px;font-weight:700;margin-top:4px;}}
                    .kpi-sub{{position:relative;color:#71809D;font-size:10px;margin-top:7px;}}
                    .kpi-line{{position:absolute;left:0;bottom:0;height:2px;width:0;background:linear-gradient(90deg,{value_color},transparent);animation:kpiLine 1.2s ease forwards;animation-delay:{.35+index*.08:.2f}s;}}
                    @keyframes kpiIn{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}
                    @keyframes kpiLine{{from{{width:0}}to{{width:100%}}}}
                </style>
                <div class="kpi-card">
                    <div class="kpi-label">{html.escape(label)}</div>
                    <div class="kpi-value" id="kpiValue">0</div>
                    <div class="kpi-percent">{percentage}</div>
                    <div class="kpi-sub">{html.escape(subtext)}</div>
                    <div class="kpi-line"></div>
                </div>
                <script>
                    const target={int(value)},el=document.getElementById('kpiValue'),start=performance.now(),duration=850;
                    function count(now){{const p=Math.min((now-start)/duration,1),e=1-Math.pow(1-p,3);el.textContent=Math.round(target*e);if(p<1)requestAnimationFrame(count);}}
                    requestAnimationFrame(count);
                </script>
                """, height=140, scrolling=False
            )

    # ========================================================
    # AVERAGE DIMENSION SCORES
    # ========================================================

    st.markdown(
        """
        <h3 style="
            color:#FFFFFF;
            font-size:20px;
            font-weight:700;
            margin:30px 0 14px 0;
        ">
            📊 Average Dimension Scores
        </h3>
        """,
        unsafe_allow_html=True,
    )

    def average_score(key):
        if not filtered_records:
            return 0.0
        return sum(
            x[key] for x in filtered_records
        ) / len(filtered_records)

    dimension_data = [
        (
            "Relevance",
            average_score("relevance"),
            "#6366F1",
        ),
        (
            "Accuracy",
            average_score("accuracy"),
            "#10BFA3",
        ),
        (
            "Hallucination Detection",
            average_score("hallucination"),
            "#F59E0B",
        ),
        (
            "Completeness",
            average_score("completeness"),
            "#8B5CF6",
        ),
    ]

    score_cols = st.columns(4)

    for index, (col, (label, score, color)) in enumerate(zip(score_cols, dimension_data)):
        with col:
            components.html(
                f"""
                <style>
                    .score-card {{ background:#0D1222; border:1px solid #252D4A; border-radius:14px; padding:18px; box-sizing:border-box; font-family:Arial,sans-serif; animation:scoreIn .55s ease both; animation-delay:{index*0.10:.2f}s; }}
                    .score-label {{ color:#AAB2C8; font-size:13px; font-weight:500; }}
                    .score-value {{ color:#FFFFFF; font-size:27px; font-weight:700; margin:5px 0 8px; }}
                    .score-track {{ height:7px; background:#242B43; border-radius:10px; overflow:hidden; }}
                    .score-fill {{ height:100%; width:0; background:{color}; border-radius:10px; animation:fillScore 1.25s cubic-bezier(.22,1,.36,1) forwards; animation-delay:{0.2+index*0.1:.2f}s; }}
                    @keyframes fillScore {{ from {{ width:0%; }} to {{ width:{max(0,min(score/5*100,100)):.1f}%; }} }}
                    @keyframes scoreIn {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:translateY(0); }} }}
                </style>
                <div class="score-card">
                    <div class="score-label">{label}</div>
                    <div class="score-value" id="scoreValue">0.00 / 5</div>
                    <div class="score-track"><div class="score-fill"></div></div>
                </div>
                <script>
                const target={score:.2f}, el=document.getElementById('scoreValue'), start=performance.now(), duration=1000;
                function tick(now){{const p=Math.min((now-start)/duration,1),e=1-Math.pow(1-p,3);el.textContent=(target*e).toFixed(2)+' / 5';if(p<1)requestAnimationFrame(tick);}}
                requestAnimationFrame(tick);
                </script>
                """, height=112, scrolling=False
            )

    # ========================================================
    # DONUT CHART FUNCTION
    # Uses SVG inside a Streamlit component so the chart is
    # rendered by the browser instead of being treated as text.
    # ========================================================

    def render_3d_pie_chart(title, values):
        """Render a polished animated pseudo-3D pie chart with hover details."""
        import math

        values = [(str(label), int(value)) for label, value in values if int(value) > 0]
        total = sum(value for _, value in values)

        if total == 0:
            components.html(f"""
                <div style="width:100%;height:350px;box-sizing:border-box;background:linear-gradient(145deg,#0E1426,#0A0F1E);border:1px solid #252D4A;border-radius:18px;padding:18px;font-family:Arial,sans-serif;color:#FFF;">
                    <div style="font-size:18px;font-weight:800;">{html.escape(title)}</div>
                    <div style="height:270px;display:flex;align-items:center;justify-content:center;color:#94A3B8;font-size:14px;">No evaluation data available</div>
                </div>
            """, height=370, scrolling=False)
            return

        palette = [
            ("#6366F1", "#34358A", "#818CF8"),
            ("#F45B45", "#963426", "#FB7185"),
            ("#10BFA3", "#056B5D", "#2DD4BF"),
            ("#F59E0B", "#8A5305", "#FBBF24"),
            ("#8B5CF6", "#4B238F", "#A78BFA"),
        ]
        cx, cy = 250, 116
        rx, ry = 108, 70
        depth = 20
        extrusion_steps = 12

        def polar(angle):
            a = math.radians(angle)
            return cx + rx * math.cos(a), cy + ry * math.sin(a)

        def sector_path(start_angle, end_angle, y_shift=0):
            span = max(0.01, end_angle - start_angle)
            large_arc = 1 if span > 180 else 0
            sx, sy = polar(start_angle)
            ex, ey = polar(end_angle)
            return (f"M {cx:.2f} {cy+y_shift:.2f} L {sx:.2f} {sy+y_shift:.2f} "
                    f"A {rx:.2f} {ry:.2f} 0 {large_arc} 1 {ex:.2f} {ey+y_shift:.2f} Z")

        slices, legend_items = [], []
        current_angle = -90.0

        for index, (label, value) in enumerate(values):
            percentage = value / total
            span = percentage * 360.0
            gap_angle = min(1.6, span * 0.055)
            start_angle = current_angle + gap_angle
            end_angle = current_angle + span - gap_angle
            if end_angle <= start_angle:
                start_angle, end_angle = current_angle, current_angle + span

            top_color, side_color, glow_color = palette[index % len(palette)]
            gradient_id = f"pieGrad{index}"
            delay = index * 0.13
            mid = math.radians((start_angle + end_angle) / 2)
            dx, dy = math.cos(mid) * 7, math.sin(mid) * 5

            for layer in range(extrusion_steps, 0, -1):
                y_shift = depth * layer / extrusion_steps
                opacity = 0.34 + (extrusion_steps - layer) * 0.035
                path = sector_path(start_angle, end_angle, y_shift)
                slices.append(f"""<path d="{path}" fill="{side_color}" opacity="{opacity:.3f}" class="pie-side" style="animation-delay:{delay:.2f}s;"/>""")

            path = sector_path(start_angle, end_angle)
            safe_label = html.escape(label, quote=True)
            percent_text = f"{percentage*100:.1f}%"
            slices.append(f"""<g class="pie-piece" data-label="{safe_label}" data-value="{value}" data-percent="{percent_text}" data-dx="{dx:.2f}" data-dy="{dy:.2f}">
                <path d="{path}" fill="url(#{gradient_id})" class="pie-slice" style="animation-delay:{delay:.2f}s;"/>
                <path d="{path}" fill="none" stroke="#FFFFFF" stroke-opacity=".17" stroke-width="1.2" class="pie-edge" style="animation-delay:{delay:.2f}s;"/>
                <path d="{path}" fill="#FFFFFF" opacity=".001" class="pie-hit"/>
            </g>""")

            legend_items.append(f"""<div class="legend-item" data-index="{index}" style="--legend-color:{top_color};--legend-glow:{glow_color};">
                <span class="legend-dot"></span><span class="legend-label">{html.escape(label)}</span>
                <span class="legend-value">{value}</span><span class="legend-percent">{percent_text}</span>
            </div>""")
            current_angle += span

        gradients = ''.join(
            f"""<linearGradient id="pieGrad{i}" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="{palette[i % len(palette)][2]}"/>
                <stop offset="48%" stop-color="{palette[i % len(palette)][0]}"/>
                <stop offset="100%" stop-color="{palette[i % len(palette)][1]}"/>
            </linearGradient>"""
            for i in range(len(values))
        )

        components.html(f"""
            <div class="pie-card">
                <style>
                    * {{ box-sizing:border-box; }}
                    .pie-card {{ width:100%;height:350px;position:relative;overflow:hidden;background:radial-gradient(circle at 50% 25%,rgba(99,102,241,.10),transparent 35%),linear-gradient(145deg,#10172B 0%,#0B1020 58%,#090E1A 100%);border:1px solid rgba(71,85,125,.55);border-radius:18px;padding:15px 16px 11px;font-family:Arial,sans-serif;color:#F8FAFC;box-shadow:inset 0 1px 0 rgba(255,255,255,.035),0 12px 30px rgba(0,0,0,.16); }}
                    .pie-card::before {{ content:"";position:absolute;width:180px;height:180px;border-radius:50%;left:50%;top:42px;transform:translateX(-50%);background:radial-gradient(circle,rgba(99,102,241,.10),transparent 68%);filter:blur(8px);pointer-events:none; }}
                    .pie-title {{ position:relative;z-index:3;font-size:18px;font-weight:800;letter-spacing:-.01em;margin:0 0 1px 2px; }}
                    .pie-subtitle {{ position:relative;z-index:3;color:#64748B;font-size:10px;margin-left:2px;margin-top:2px; }}
                    .pie-svg {{ display:block;width:100%;height:214px;margin-top:-2px;overflow:visible; }}
                    .pie-stage {{ transform-box:fill-box;transform-origin:center;animation:stageFloat 5s ease-in-out infinite; }}
                    .pie-slice {{ transform-box:fill-box;transform-origin:250px 116px;animation:pieReveal .9s cubic-bezier(.22,1,.36,1) both;transition:filter .2s ease; }}
                    .pie-side {{ transform-box:fill-box;transform-origin:250px 116px;animation:depthReveal .75s cubic-bezier(.22,1,.36,1) both; }}
                    .pie-edge {{ animation:edgeReveal 1.05s ease both; }}
                    .pie-piece {{ transition:transform .22s cubic-bezier(.2,.8,.2,1),filter .22s ease;transform-box:fill-box;transform-origin:center; }}
                    .pie-piece:hover {{ filter:drop-shadow(0 7px 8px rgba(0,0,0,.35)); }}
                    .pie-piece:hover .pie-slice {{ filter:brightness(1.12) saturate(1.08); }}
                    .pie-hit {{ cursor:pointer; }}
                    .pie-total {{ pointer-events:none;font-weight:800;font-size:21px;fill:#F8FAFC; }}
                    .pie-total-label {{ pointer-events:none;font-size:9px;fill:#94A3B8;letter-spacing:.08em; }}
                    .legend {{ position:relative;z-index:4;display:flex;flex-wrap:wrap;justify-content:center;gap:6px 7px;margin-top:-3px; }}
                    .legend-item {{ display:flex;align-items:center;gap:5px;padding:5px 8px;border:1px solid rgba(71,85,125,.32);border-radius:999px;background:rgba(15,23,42,.48);color:#CBD5E1;font-size:10px;white-space:nowrap;transition:all .18s ease;cursor:pointer; }}
                    .legend-item:hover,.legend-item.active {{ border-color:var(--legend-color);background:rgba(30,41,70,.72);transform:translateY(-1px);box-shadow:0 5px 16px rgba(0,0,0,.20); }}
                    .legend-dot {{ width:7px;height:7px;border-radius:50%;background:var(--legend-color);box-shadow:0 0 7px var(--legend-glow);flex:0 0 7px; }}
                    .legend-label {{ color:#E2E8F0; }} .legend-value {{ color:#94A3B8;font-weight:700; }} .legend-percent {{ color:#64748B; }}
                    .pie-tooltip {{ position:fixed;display:none;z-index:99999;pointer-events:none;min-width:158px;padding:11px 13px;border-radius:11px;background:rgba(7,11,23,.97);border:1px solid rgba(129,140,248,.65);box-shadow:0 14px 35px rgba(0,0,0,.48),0 0 18px rgba(99,102,241,.10);color:#F8FAFC;font:12px/1.5 Arial,sans-serif;backdrop-filter:blur(8px); }}
                    .tip-head {{ font-size:13px;font-weight:800;margin-bottom:4px;color:#FFFFFF; }} .tip-row {{ display:flex;justify-content:space-between;gap:18px;color:#CBD5E1; }} .tip-value {{ color:#FFFFFF;font-weight:800; }}
                    @keyframes pieReveal {{ 0% {{ opacity:0;transform:translateY(10px) scale(.72) rotateX(22deg); }} 62% {{ opacity:1;transform:translateY(-2px) scale(1.045) rotateX(0deg); }} 100% {{ opacity:1;transform:translateY(0) scale(1) rotateX(0deg); }} }}
                    @keyframes depthReveal {{ 0% {{ opacity:0;transform:translateY(-10px); }} 100% {{ opacity:1;transform:translateY(0); }} }}
                    @keyframes edgeReveal {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
                    @keyframes stageFloat {{ 0%,100% {{ transform:translateY(0) rotateX(0deg); }} 50% {{ transform:translateY(-2px) rotateX(.8deg); }} }}
                    @media (prefers-reduced-motion:reduce) {{ *,*::before,*::after {{ animation-duration:.01ms !important;transition:none !important; }} }}
                </style>
                <div class="pie-title">{html.escape(title)}</div>
                <div class="pie-subtitle">Hover a slice for evaluation details</div>
                <svg class="pie-svg" viewBox="0 0 500 214" xmlns="http://www.w3.org/2000/svg">
                    <defs>{gradients}<radialGradient id="pieCenterGlow"><stop offset="0%" stop-color="#1E293B" stop-opacity=".92"/><stop offset="100%" stop-color="#0B1020" stop-opacity=".98"/></radialGradient></defs>
                    <g class="pie-stage">
                        <ellipse cx="250" cy="143" rx="122" ry="25" fill="#000000" opacity=".34"/>
                        {''.join(slices)}
                        <ellipse cx="250" cy="116" rx="35" ry="22" fill="url(#pieCenterGlow)" opacity=".96"/>
                        <text x="250" y="113" text-anchor="middle" class="pie-total">{total}</text>
                        <text x="250" y="127" text-anchor="middle" class="pie-total-label">EVALUATIONS</text>
                    </g>
                </svg>
                <div class="legend">{''.join(legend_items)}</div>
                <div id="pieTooltip" class="pie-tooltip"></div>
                <script>
                (() => {{
                    const card=document.currentScript.closest('.pie-card');
                    const tip=card.querySelector('#pieTooltip');
                    const pieces=[...card.querySelectorAll('.pie-piece')];
                    const legends=[...card.querySelectorAll('.legend-item')];
                    function moveTip(e) {{ const r=card.getBoundingClientRect(); const x=e.clientX-r.left,y=e.clientY-r.top; tip.style.left=Math.min(x+14,card.clientWidth-185)+'px';tip.style.top=Math.max(8,y-18)+'px'; }}
                    function show(piece,e) {{ tip.innerHTML='<div class="tip-head">'+piece.dataset.label+'</div><div class="tip-row"><span>Evaluations</span><span class="tip-value">'+piece.dataset.value+'</span></div><div class="tip-row"><span>Share</span><span class="tip-value">'+piece.dataset.percent+'</span></div>';tip.style.display='block';moveTip(e); }}
                    function activate(piece) {{ const i=pieces.indexOf(piece);piece.style.transform=`translate(${{piece.dataset.dx}}px,${{piece.dataset.dy}}px) scale(1.018)`;if(legends[i]) legends[i].classList.add('active'); }}
                    function deactivate(piece) {{ const i=pieces.indexOf(piece);piece.style.transform='';if(legends[i]) legends[i].classList.remove('active'); }}
                    pieces.forEach(piece=>{{const hit=piece.querySelector('.pie-hit');hit.addEventListener('mouseenter',e=>{{activate(piece);show(piece,e);}});hit.addEventListener('mousemove',moveTip);hit.addEventListener('mouseleave',()=>{{deactivate(piece);tip.style.display='none';}});}});
                    legends.forEach((legend,i)=>{{legend.addEventListener('mouseenter',()=>{{if(pieces[i])activate(pieces[i]);}});legend.addEventListener('mouseleave',()=>{{if(pieces[i])deactivate(pieces[i]);}});}});
                }})();
                </script>
            </div>
        """, height=370, scrolling=False)

    # ========================================================
    # DISTRIBUTION 3D PIE CHARTS
    # ========================================================

    st.markdown(
        """
        <h3 style="
            color:#FFFFFF;
            font-size:20px;
            font-weight:700;
            margin:30px 0 14px 0;
        ">
            📈 Evaluation Distribution
        </h3>
        """,
        unsafe_allow_html=True,
    )

    hallucination_detected = sum(
        1
        for x in filtered_records
        if (
            x["hallucination"] < 5
            or len(x["flagged_claims"]) > 0
        )
    )

    no_hallucination = (
        total_evaluations
        - hallucination_detected
    )

    score_0_1 = sum(
        1
        for x in filtered_records
        if 0 <= x["overall_score"] < 2
    )

    score_2_3 = sum(
        1
        for x in filtered_records
        if 2 <= x["overall_score"] < 3.5
    )

    score_3_5 = sum(
        1
        for x in filtered_records
        if 3.5 <= x["overall_score"] <= 5
    )

    complete_count = sum(
        1
        for x in filtered_records
        if x["completeness"] >= 4
        and len(x["missing_components"]) == 0
    )

    incomplete_count = (
        total_evaluations
        - complete_count
    )

    col1, col2 = st.columns(2)

    with col1:
        render_3d_pie_chart(
            "Verdict Distribution",
            [
                ("Pass", pass_count),
                ("Fail", fail_count),
                (
                    "Needs Improvement",
                    needs_count,
                ),
            ],
        )

    with col2:
        render_3d_pie_chart(
            "Hallucination Status",
            [
                (
                    "Hallucination Detected",
                    hallucination_detected,
                ),
                (
                    "No Hallucination",
                    no_hallucination,
                ),
            ],
        )

    st.markdown(
        "<div style='height:18px'></div>",
        unsafe_allow_html=True,
    )

    col3, col4 = st.columns(2)

    with col3:
        render_3d_pie_chart(
            "Score Distribution",
            [
                ("0 – 1.99", score_0_1),
                ("2 – 3.49", score_2_3),
                ("3.5 – 5.0", score_3_5),
            ],
        )

    with col4:
        render_3d_pie_chart(
            "Completeness Status",
            [
                ("Complete", complete_count),
                ("Incomplete", incomplete_count),
            ],
        )

    # ========================================================
    # KEY INSIGHTS
    # ========================================================

    if filtered_records:
        insight_scores = {
            "Relevance": average_score("relevance"),
            "Accuracy": average_score("accuracy"),
            "Hallucination Detection": average_score("hallucination"),
            "Completeness": average_score("completeness"),
        }
        strongest_dimension = max(insight_scores, key=insight_scores.get)
        weakest_dimension = min(insight_scores, key=insight_scores.get)
        # Stored risk values are "High", "Medium", and "Low".
        # Normalize both formats ("High" and "High Risk") so the
        # dashboard counts remain correct for old and new records.
        def normalize_risk_level(value):
            risk = str(value or "").strip().lower()
            risk = risk.replace(" risk", "")
            return risk

        high_risk_count = sum(
            1 for x in filtered_records
            if normalize_risk_level(x.get("risk_level")) == "high"
        )
        medium_risk_count = sum(
            1 for x in filtered_records
            if normalize_risk_level(x.get("risk_level")) == "medium"
        )
        low_risk_count = sum(
            1 for x in filtered_records
            if normalize_risk_level(x.get("risk_level")) == "low"
        )
        unknown_risk_count = max(
            0,
            len(filtered_records)
            - high_risk_count
            - medium_risk_count
            - low_risk_count,
        )
        pass_rate = pct(pass_count, total_evaluations) if total_evaluations else 0

        st.markdown(
            """
            <div style="margin:28px 0 14px 0;color:#FFFFFF;font-size:20px;font-weight:800;">💡 Key Insights</div>
            """,
            unsafe_allow_html=True,
        )

        insight_html = f"""
        <style>
            .insight-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}}
            .insight{{background:linear-gradient(145deg,rgba(21,27,50,.92),rgba(13,18,34,.92));border:1px solid #252D4A;border-radius:14px;padding:14px 15px;min-height:92px;box-sizing:border-box;transition:.2s ease;}}
            .insight:hover{{transform:translateY(-2px);border-color:rgba(129,140,248,.45);box-shadow:0 10px 24px rgba(0,0,0,.22);}}
            .insight-k{{color:#71809D;font-size:10px;font-weight:800;letter-spacing:.7px;text-transform:uppercase;margin-bottom:7px;}}
            .insight-v{{color:#F8FAFC;font-size:17px;font-weight:800;}}
            .insight-s{{color:#94A3B8;font-size:11px;margin-top:5px;}}
            @media(max-width:900px){{.insight-grid{{grid-template-columns:repeat(2,1fr);}}}}
        </style>
        <div class="insight-grid">
            <div class="insight"><div class="insight-k">Strongest Dimension</div><div class="insight-v">{html.escape(strongest_dimension)}</div><div class="insight-s">{insight_scores[strongest_dimension]:.2f} / 5 average</div></div>
            <div class="insight"><div class="insight-k">Needs Attention</div><div class="insight-v">{html.escape(weakest_dimension)}</div><div class="insight-s">{insight_scores[weakest_dimension]:.2f} / 5 average</div></div>
            <div class="insight"><div class="insight-k">Pass Rate</div><div class="insight-v">{pass_rate:.1f}%</div><div class="insight-s">{pass_count} of {total_evaluations} evaluations</div></div>
            <div class="insight"><div class="insight-k">Risk Watch</div><div class="insight-v">🔴 {high_risk_count} High · 🟠 {medium_risk_count} Medium · 🟢 {low_risk_count} Low</div><div class="insight-s">Across {len(filtered_records)} evaluations in the current filtered view{f" · {unknown_risk_count} unknown" if unknown_risk_count else ""}</div></div>
        </div>
        """
        components.html(insight_html, height=125, scrolling=False)

    # ========================================================
    # HALLUCINATION + COMPLETENESS STATISTICS
    # ========================================================

    st.markdown(
        """
        <h3 style="
            color:#FFFFFF;
            font-size:20px;
            font-weight:700;
            margin:30px 0 14px 0;
        ">
            ⚠️ Evaluation Issue Statistics
        </h3>
        """,
        unsafe_allow_html=True,
    )

    flagged_claim_count = sum(
        len(x["flagged_claims"])
        for x in filtered_records
    )

    missing_component_count = sum(
        len(x["missing_components"])
        for x in filtered_records
    )

    hallucination_frequency = (
        flagged_claim_count / total_evaluations
        if total_evaluations
        else 0
    )

    missing_frequency = (
        missing_component_count / total_evaluations
        if total_evaluations
        else 0
    )

    low_accuracy_count = sum(
        1
        for x in filtered_records
        if x["accuracy"] < 3.5
    )

    low_relevance_count = sum(
        1
        for x in filtered_records
        if x["relevance"] < 3.5
    )

    issue_cards = [
        (
            "Flagged Hallucinated Claims",
            flagged_claim_count,
            f"{hallucination_frequency:.2f} per evaluation",
            "#F45B45",
        ),
        (
            "Missing Components",
            missing_component_count,
            f"{missing_frequency:.2f} per evaluation",
            "#F59E0B",
        ),
        (
            "Low Accuracy",
            low_accuracy_count,
            "Score below 3.5",
            "#10BFA3",
        ),
        (
            "Low Relevance",
            low_relevance_count,
            "Score below 3.5",
            "#6366F1",
        ),
    ]

    issue_cols = st.columns(4)

    for col, (
        title,
        value,
        subtitle,
        color,
    ) in zip(issue_cols, issue_cards):

        with col:
            st.markdown(
                f"""
                <div style="
                    background:#0D1222;
                    border:1px solid #252D4A;
                    border-radius:14px;
                    padding:17px;
                    min-height:112px;
                ">
                    <div style="
                        color:#AAB2C8;
                        font-size:12px;
                    ">{title}</div>
                    <div style="
                        color:{color};
                        font-size:27px;
                        font-weight:700;
                        margin-top:5px;
                    ">{value}</div>
                    <div style="
                        color:#C9D0E3;
                        font-size:11px;
                        margin-top:3px;
                    ">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
    # BATCH SUMMARY + TRENDS
    # ========================================================

    st.markdown(
        """
        <h3 style="
            color:#FFFFFF;
            font-size:20px;
            font-weight:700;
            margin:30px 0 14px 0;
        ">
            📦 Batch Analysis & Trends
        </h3>
        """,
        unsafe_allow_html=True,
    )

    batch_rows = []

    for batch in batches:

        batch_id = batch.get("batch_id")

        if batch_id is None:
            continue

        records = [
            x
            for x in dashboard_records
            if x["batch_id"] == batch_id
        ]

        if not records:
            continue

        batch_rows.append({
            "Batch": f"Batch {batch_id}",
            "Batch ID": batch_id,
            "Date": str(
                batch.get("timestamp", "")
            )[:19],
            "Records": len(records),
            "Pass": sum(
                x["verdict"].lower() == "pass"
                for x in records
            ),
            "Needs Improvement": sum(
                x["verdict"].lower()
                == "needs improvement"
                for x in records
            ),
            "Fail": sum(
                x["verdict"].lower() == "fail"
                for x in records
            ),
            "Avg Relevance": sum(
                x["relevance"] for x in records
            ) / len(records),
            "Avg Accuracy": sum(
                x["accuracy"] for x in records
            ) / len(records),
            "Avg Hallucination": sum(
                x["hallucination"] for x in records
            ) / len(records),
            "Avg Completeness": sum(
                x["completeness"] for x in records
            ) / len(records),
            "Avg Overall": sum(
                x["overall_score"] for x in records
            ) / len(records),
        })

    if batch_rows:

        batch_df = pd.DataFrame(
            batch_rows
        ).sort_values(
            "Batch ID"
        )

        trend_df = batch_df.set_index(
            "Batch"
        )[[
            "Avg Relevance",
            "Avg Accuracy",
            "Avg Hallucination",
            "Avg Completeness",
        ]].round(2)

        st.markdown(
            "**Average Dimension Scores Across Batches**"
        )

        chart_rows = []
        for _, row in batch_df.iterrows():
            chart_rows.append({
                "label": str(row["Batch"]),
                "relevance": float(row["Avg Relevance"]),
                "accuracy": float(row["Avg Accuracy"]),
                "hallucination": float(row["Avg Hallucination"]),
                "completeness": float(row["Avg Completeness"]),
            })
        chart_json = json.dumps(chart_rows)
        components.html(
            f"""
            <style>@keyframes drawLine {{ to {{ stroke-dashoffset:0; }} }} @keyframes pointIn {{ from {{ opacity:0; }} to {{ opacity:1; }} }} .trend-tooltip {{ position:fixed; display:none; z-index:9999; pointer-events:none; min-width:170px; padding:10px 12px; background:rgba(10,15,30,.97); border:1px solid rgba(99,102,241,.55); border-radius:9px; box-shadow:0 10px 28px rgba(0,0,0,.45); color:#F8FAFC; font:12px/1.5 Arial,sans-serif; }}</style>
            <div style="background:#0D1222;border:1px solid #252D4A;border-radius:14px;padding:16px;font-family:Arial,sans-serif;">
              <svg id="trendSvg" viewBox="0 0 900 300" width="100%" height="300" xmlns="http://www.w3.org/2000/svg"></svg>
              <div style="display:flex;gap:18px;justify-content:center;flex-wrap:wrap;color:#CBD5E1;font-size:12px;"><span>● Relevance</span><span>● Accuracy</span><span>● Hallucination</span><span>● Completeness</span></div>
            </div>
            <script>
            const rows={chart_json},svg=document.getElementById('trendSvg'),W=900,H=300,L=55,R=20,T=18,B=42,iw=W-L-R,ih=H-T-B;
            const xx=i=>L+(rows.length<2?iw/2:i/(rows.length-1)*iw), yy=v=>T+(5-v)/5*ih;
            let s=''; for(let g=0;g<=5;g++){{let y=yy(g);s+=`<line x1="${{L}}" y1="${{y}}" x2="${{W-R}}" y2="${{y}}" stroke="#252D4A"/><text x="${{L-10}}" y="${{y+4}}" text-anchor="end" fill="#94A3B8" font-size="11">${{g}}</text>`;}}
            const series=[['relevance','#6366F1'],['accuracy','#10BFA3'],['hallucination','#F59E0B'],['completeness','#8B5CF6']];
            series.forEach(([k,c],si)=>{{let pts=rows.map((r,i)=>`${{xx(i)}},${{yy(r[k])}}`).join(' ');s+=`<polyline points="${{pts}}" fill="none" stroke="${{c}}" stroke-width="3" stroke-linecap="round" pathLength="1000" stroke-dasharray="1000" stroke-dashoffset="1000" style="animation:drawLine 1.4s ease forwards;animation-delay:${{si*.12}}s"/>`;rows.forEach((r,i)=>s+=`<circle cx="${{xx(i)}}" cy="${{yy(r[k])}}" r="4" fill="${{c}}" style="animation:pointIn .5s ease both;animation-delay:${{.7+si*.12+i*.05}}s"/>`);}});
            rows.forEach((r,i)=>s+=`<rect x="${{xx(i)-14}}" y="${{T}}" width="28" height="${{ih}}" fill="#FFFFFF" opacity="0" class="trend-hit" data-label="${{r.label}}" data-relevance="${{r.relevance.toFixed(2)}}" data-accuracy="${{r.accuracy.toFixed(2)}}" data-hallucination="${{r.hallucination.toFixed(2)}}" data-completeness="${{r.completeness.toFixed(2)}}"/>`);
            rows.forEach((r,i)=>s+=`<text x="${{xx(i)}}" y="${{H-14}}" text-anchor="middle" fill="#94A3B8" font-size="11">${{r.label}}</text>`); svg.innerHTML=s;
            const tip=document.createElement('div'); tip.className='trend-tooltip'; tip.id='trendTooltip'; document.body.appendChild(tip);
            document.querySelectorAll('.trend-hit').forEach(el=>{{
              el.addEventListener('mouseenter',e=>{{
                tip.innerHTML='<b>'+el.dataset.label+'</b><br>Relevance: <b>'+el.dataset.relevance+'</b> / 5<br>Accuracy: <b>'+el.dataset.accuracy+'</b> / 5<br>Hallucination: <b>'+el.dataset.hallucination+'</b> / 5<br>Completeness: <b>'+el.dataset.completeness+'</b> / 5';
                tip.style.display='block'; tip.style.left=Math.min(e.clientX+14,window.innerWidth-245)+'px'; tip.style.top=Math.max(8,e.clientY-20)+'px';
              }});
              el.addEventListener('mousemove',e=>{{ tip.style.left=Math.min(e.clientX+14,window.innerWidth-245)+'px'; tip.style.top=Math.max(8,e.clientY-20)+'px'; }});
              el.addEventListener('mouseleave',()=>{{tip.style.display='none';}});
            }});
            </script>
            """, height=350, scrolling=False
        )

        st.markdown(
            "**Batch Verdict Distribution**"
        )

        verdict_trend_df = batch_df.set_index(
            "Batch"
        )[[
            "Pass",
            "Needs Improvement",
            "Fail",
        ]]

        st.dataframe(
            verdict_trend_df,
            use_container_width=True,
        )

        st.markdown(
            "**Batch Summary**"
        )

        display_batch_df = batch_df[[
            "Batch",
            "Date",
            "Records",
            "Pass",
            "Needs Improvement",
            "Fail",
            "Avg Relevance",
            "Avg Accuracy",
            "Avg Hallucination",
            "Avg Completeness",
            "Avg Overall",
        ]].copy()

        score_columns = [
            "Avg Relevance",
            "Avg Accuracy",
            "Avg Hallucination",
            "Avg Completeness",
            "Avg Overall",
        ]

        display_batch_df[
            score_columns
        ] = display_batch_df[
            score_columns
        ].round(2)

        st.dataframe(
            display_batch_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No completed batch evaluations are available yet."
        )

    # ========================================================
    # MOST FREQUENT EVALUATION ISSUES
    # ========================================================

    issue_frequency = pd.DataFrame([
        {
            "Issue": "Hallucinated / unsupported claims",
            "Frequency": flagged_claim_count,
        },
        {
            "Issue": "Missing or incomplete information",
            "Frequency": missing_component_count,
        },
        {
            "Issue": "Low accuracy",
            "Frequency": low_accuracy_count,
        },
        {
            "Issue": "Low relevance",
            "Frequency": low_relevance_count,
        },
    ]).sort_values(
        "Frequency",
        ascending=False,
    )

    st.markdown(
        "**Most Frequently Occurring Evaluation Issues**"
    )

    st.dataframe(
        issue_frequency,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # DRILL-DOWN TO INDIVIDUAL RESULTS
    # ========================================================

    st.markdown(
        """
        <h3 style="
            color:#FFFFFF;
            font-size:20px;
            font-weight:700;
            margin:30px 0 14px 0;
        ">
            🔎 Evaluation Drill-Down
        </h3>
        """,
        unsafe_allow_html=True,
    )

    if not filtered_records:

        st.info(
            "No evaluation records match the selected filters."
        )

    else:

        drill_rows = []

        for x in filtered_records:
            drill_rows.append({
                "ID": x["submission_id"],
                "Source": x["batch_label"],
                "Question": x["question"][:100],
                "Overall": round(
                    x["overall_score"],
                    2,
                ),
                "Verdict": x["verdict"],
                "Risk": x["risk_level"],
                "Hallucinated Claims": len(
                    x["flagged_claims"]
                ),
                "Missing Components": len(
                    x["missing_components"]
                ),
            })

        st.dataframe(
            pd.DataFrame(drill_rows),
            use_container_width=True,
            hide_index=True,
        )

        drill_options = {
            f"{x['submission_id']} • "
            f"{x['verdict']} • "
            f"{x['overall_score']:.2f}/5 • "
            f"{x['question'][:60]}": x
            for x in filtered_records
        }

        selected_drill_label = st.selectbox(
            "Open Individual Evaluation",
            list(drill_options.keys()),
            key="dashboard_drilldown",
        )

        selected_record = drill_options[
            selected_drill_label
        ]

        with st.expander(
            "📄 View Full Evaluation Details",
            expanded=True,
        ):

            st.write(
                f"**Question:** "
                f"{selected_record['question']}"
            )

            st.write(
                f"**AI Response:** "
                f"{selected_record['ai_response']}"
            )

            st.write(
                f"**Reference Answer:** "
                f"{selected_record['reference_answer'] or 'Not provided'}"
            )

            detail_cols = st.columns(5)

            detail_values = [
                (
                    "Relevance",
                    selected_record["relevance"],
                ),
                (
                    "Accuracy",
                    selected_record["accuracy"],
                ),
                (
                    "Completeness",
                    selected_record["completeness"],
                ),
                (
                    "Hallucination",
                    selected_record["hallucination"],
                ),
                (
                    "Overall",
                    selected_record["overall_score"],
                ),
            ]

            for col, (
                label,
                value,
            ) in zip(
                detail_cols,
                detail_values,
            ):
                with col:
                    st.metric(
                        label,
                        f"{value:.2f}/5",
                    )

            st.write(
                f"**Verdict:** "
                f"{selected_record['verdict']}"
            )

            st.write(
                f"**Risk Level:** "
                f"{selected_record['risk_level']}"
            )

            st.markdown(
                "**Hallucination Findings**"
            )

            if selected_record[
                "flagged_claims"
            ]:
                for claim in selected_record[
                    "flagged_claims"
                ]:
                    st.warning(str(claim))
            else:
                st.success(
                    "No flagged claims identified."
                )

            st.markdown(
                "**Completeness Findings**"
            )

            if selected_record[
                "missing_components"
            ]:
                for missing in selected_record[
                    "missing_components"
                ]:
                    st.warning(str(missing))
            else:
                st.success(
                    "No missing components identified."
                )

            with st.expander(
                "🎯 Relevance Reasoning"
            ):
                st.write(
                    selected_record[
                        "relevance_reasoning"
                    ]
                    or "No reasoning returned."
                )

            with st.expander(
                "✓ Accuracy Reasoning"
            ):
                st.write(
                    selected_record[
                        "accuracy_reasoning"
                    ]
                    or "No reasoning returned."
                )

            with st.expander(
                "◈ Completeness Reasoning"
            ):
                st.write(
                    selected_record[
                        "completeness_reasoning"
                    ]
                    or "No reasoning returned."
                )

            with st.expander(
                "⚠ Hallucination Reasoning"
            ):
                st.write(
                    selected_record[
                        "hallucination_reasoning"
                    ]
                    or "No reasoning returned."
                )

    # ========================================================
    # RECENT SINGLE EVALUATIONS
    # ========================================================

    st.markdown(
        """
        <h3 style="
            color:#FFFFFF;
            font-size:20px;
            font-weight:700;
            margin:30px 0 14px 0;
        ">
            🕘 Recent Single Evaluations
        </h3>
        """,
        unsafe_allow_html=True,
    )

    recent = singles[:5]

    if not recent:

        st.info(
            "No completed single evaluations yet."
        )

    else:

        for item in recent:

            verdict = json_value(
                item.get("verdict_result")
            )

            verdict_name = verdict.get(
                "verdict",
                verdict.get(
                    "status",
                    "Completed",
                ),
            )

            score = float(
                item.get(
                    "overall_score",
                    0,
                )
                or 0
            )

            with st.expander(
                f"🔍 "
                f"{item['question'][:90]}  •  "
                f"{verdict_name}  •  "
                f"{score:.1f}/5"
            ):

                st.write(
                    f"**Timestamp:** "
                    f"{item['timestamp']}"
                )

                st.write(
                    f"**AI Response:** "
                    f"{item['ai_response']}"
                )

                st.write(
                    f"**Reference Answer:** "
                    f"{item.get('reference_answer') or 'Not provided'}"
                )


# ============================================================
# SINGLE EVALUATION
# ============================================================

elif page == "Single Evaluation":

    st.markdown(
        '<div class="section-title">'
        '🔍 Single Evaluation'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Evaluate one AI response using the "
        "complete multi-agent validation pipeline."
    )

    # ========================================================
    # SINGLE EVALUATION UI
    # ========================================================

    st.markdown(
        """
        <div class="ui-section-title">⚙️ Evaluation Provider</div>
        """,
        unsafe_allow_html=True,
    )

    llm_provider = st.selectbox(
        "LLM Provider",
        ["Gemini", "Groq", "Ollama"],
        index=["Gemini", "Groq", "Ollama"].index(
            st.session_state.single_llm_provider_data
        ),
        key="single_llm_provider",
    )

    st.session_state.single_llm_provider_data = llm_provider

    if llm_provider == "Gemini":
        st.caption("☁️ Using Gemini API")
    elif llm_provider == "Groq":
        st.caption("⚡ Using Groq API")
    else:
        st.caption(
            "🖥️ Using local Ollama model: "
            "qwen3:4b-instruct-2507-q4_K_M"
        )

    set_llm_provider(llm_provider.lower())

    # ========================================================
    # VALIDATION PIPELINE
    # ========================================================

    st.markdown(
        """
        <div class="ui-section-title pipeline-title">🧠 Validation Pipeline</div>
        <div class="ui-section-subtitle">
            Each stage independently analyzes the AI response before producing the final weighted verdict.
        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline = [
        ("📝", "AI Response"),
        ("🎯", "Relevance"),
        ("✓", "Accuracy"),
        ("◈", "Completeness"),
        ("⚠", "Hallucination"),
        ("⚖", "Verdict"),
    ]

    pipeline_html = '<div class="validation-pipeline">'

    for i, (icon, title) in enumerate(pipeline):
        pipeline_html += f"""
            <div class="pipeline-item">
                <div class="pipeline-card">
                    <div class="pipeline-icon">{icon}</div>
                    <div class="pipeline-label">{html.escape(title)}</div>
                </div>
            </div>
        """
        if i < len(pipeline) - 1:
            pipeline_html += '<div class="pipeline-arrow">→</div>'

    pipeline_html += "</div>"

    st.markdown(pipeline_html, unsafe_allow_html=True)

    # ========================================================
    # EVALUATION INPUT
    # ========================================================

    st.markdown(
        """
        <div class="ui-section-title input-title">📝 Evaluation Input</div>
        <div class="ui-section-subtitle">
            Enter the question and the AI-generated response you want the validation agents to evaluate.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            '<div class="field-heading">❓ Question</div>',
            unsafe_allow_html=True,
        )
        question = st.text_area(
            "Question",
            placeholder="Enter the question given to the AI...",
            height=145,
            key="single_question_widget",
            label_visibility="collapsed",
        )

    with col2:
        st.markdown(
            '<div class="field-heading">🤖 AI Generated Response</div>',
            unsafe_allow_html=True,
        )
        ai_response = st.text_area(
            "AI Generated Response",
            placeholder="Enter the AI-generated response to evaluate...",
            height=145,
            key="single_ai_response_widget",
            label_visibility="collapsed",
        )

    st.session_state.single_question_data = question
    st.session_state.single_ai_response_data = ai_response

    # ========================================================
    # REFERENCE INFORMATION
    # ========================================================

    st.markdown(
        """
        <div class="ui-section-title reference-title">📚 Reference Information</div>
        <div class="ui-section-subtitle">
            Optional ground-truth information used by the validation agents to assess factual accuracy and completeness.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown(
            '<div class="field-heading">📖 Reference Answer <span class="optional-label">Optional</span></div>',
            unsafe_allow_html=True,
        )
        reference_answer = st.text_area(
            "Reference Answer",
            placeholder="Enter the expected or ground-truth answer...",
            height=120,
            key="single_reference_widget",
            label_visibility="collapsed",
        )
        st.session_state.single_reference_answer_data = reference_answer

    with col4:
        st.markdown(
            '<div class="field-heading">📄 Reference Document <span class="optional-label">Optional</span></div>',
            unsafe_allow_html=True,
        )
        reference_document = st.file_uploader(
            "Reference Document",
            type=["txt", "pdf", "docx"],
            key="single_document",
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="upload-help">TXT · PDF · DOCX &nbsp; • &nbsp; Maximum 200 MB</div>',
            unsafe_allow_html=True,
        )

    if reference_document is not None:
        st.session_state.single_reference_document_name = reference_document.name
        st.session_state.single_reference_document_bytes = reference_document.getvalue()
    elif st.session_state.single_reference_document_name:
        st.caption(
            "Previously used document: "
            + st.session_state.single_reference_document_name
        )

    # ========================================================
    # READY STATUS
    # ========================================================

    st.markdown(
        """
        <div class="ready-status">
            <span class="ready-dot"></span>
            <span>Ready to evaluate</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # RUN SINGLE EVALUATION
    # ========================================================

    if st.button(
        "🔍 Run Single Evaluation",
        type="primary",
        use_container_width=True,
    ):

        if not question.strip():

            st.error(
                "Please enter a question."
            )

        elif not ai_response.strip():

            st.error(
                "Please enter an AI response."
            )

        else:

            try:

                with st.spinner(
                    "Executing multi-agent evaluation suite..."
                ):

                    results = retrieve(
                        question,
                        top_k=5,
                    )

                    reference_context = "\n\n".join(
                        result["chunk"].page_content
                        for result in results
                    )

                    evaluation_results = evaluate_response(
                        question=question,
                        ai_response=ai_response,
                        reference_context=reference_context,
                        reference_answer=reference_answer,
                    )

                accuracy = (
                    evaluation_results.get(
                        "accuracy",
                        {},
                    )
                    or {}
                )

                completeness = (
                    evaluation_results.get(
                        "completeness",
                        {},
                    )
                    or {}
                )

                relevance = (
                    evaluation_results.get(
                        "relevance",
                        {},
                    )
                    or {}
                )

                hallucination = (
                    evaluation_results.get(
                        "hallucination",
                        {},
                    )
                    or {}
                )

                verdict = (
                    evaluation_results.get(
                        "verdict",
                        {},
                    )
                    or {}
                )

                risk_info = get_risk_level(
                    hallucination
                )

                _, _, overall_score = (
                    format_verdict_summary(
                        verdict
                    )
                )

                saved_id = save_single_history(
                    question=question,
                    ai_response=ai_response,
                    reference_answer=reference_answer,
                    source_document=(
                        reference_document.name
                        if reference_document
                        else st.session_state.single_reference_document_name
                    ),
                    relevance=relevance,
                    accuracy=accuracy,
                    completeness=completeness,
                    hallucination=hallucination,
                    verdict=verdict,
                    overall_score=overall_score,
                    risk_level=risk_info["level"],
                )

                # =================================================
                # PERSIST COMPLETE EVALUATION
                # =================================================

                st.session_state.single_evaluation_data = {
                    "evaluation_results":
                        evaluation_results,
                    "saved_id":
                        saved_id,
                }

                st.session_state.single_question_data = (
                    question
                )

                st.session_state.single_ai_response_data = (
                    ai_response
                )

                st.session_state.single_reference_answer_data = (
                    reference_answer
                )

                st.success(
                    "Evaluation completed successfully "
                    "and saved to History."
                )

            except Exception as e:

                st.error(
                    "Evaluation failed. "
                    "The result was NOT saved to History."
                )

                st.code(
                    str(e)
                )

    # ========================================================
    # DISPLAY SAVED EVALUATION
    # ========================================================

    if (
        st.session_state.single_evaluation_data
        is not None
    ):

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '📋 Current Evaluation Result'
            '</div>',
            unsafe_allow_html=True,
        )

        render_final_result(
            st.session_state.single_evaluation_data[
                "evaluation_results"
            ],
            saved_id=(
                st.session_state.single_evaluation_data[
                    "saved_id"
                ]
            ),
            question=st.session_state.single_question_data,
            ai_response=st.session_state.single_ai_response_data,
            reference_answer=(
                st.session_state.single_reference_answer_data
            ),
        )


# ============================================================
# BATCH EVALUATION
# ============================================================

elif page == "Batch Evaluation":

    st.markdown(
        '<div class="section-title">📊 Batch Evaluation</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Upload a CSV dataset and evaluate multiple AI responses through the complete validation pipeline."
    )

    # ========================================================
    # DATASET UPLOAD
    # ========================================================

    st.markdown(
        '<div class="batch-section-title">📄 Dataset</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="batch-section-subtitle">'
        'Required columns: <b>question</b> and <b>ai_response</b>. '
        'Optional columns: <b>reference_answer</b> and <b>reference_document</b>.'
        '</div>',
        unsafe_allow_html=True,
    )

    batch_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_csv",
    )

    batch_df = None

    if batch_file is not None:
        st.session_state.batch_csv_name = batch_file.name
        st.session_state.batch_csv_bytes = batch_file.getvalue()
        batch_df = pd.read_csv(batch_file)
    elif st.session_state.batch_csv_bytes is not None:
        batch_df = pd.read_csv(io.BytesIO(st.session_state.batch_csv_bytes))

    # ========================================================
    # EVALUATION PROVIDER
    # ========================================================

    st.markdown(
        '<div class="batch-section-title">⚙️ Evaluation Provider</div>',
        unsafe_allow_html=True,
    )

    batch_llm_provider = st.selectbox(
        "LLM Provider",
        ["Gemini", "Groq", "Ollama"],
        index=["Gemini", "Groq", "Ollama"].index(
            st.session_state.batch_llm_provider_data
        ),
        key="batch_llm_provider",
        help="Choose the LLM provider for batch evaluation.",
        label_visibility="collapsed",
    )

    st.session_state.batch_llm_provider_data = batch_llm_provider

    if batch_llm_provider == "Gemini":
        st.caption("☁️ Using Gemini API")
    elif batch_llm_provider == "Groq":
        st.caption("⚡ Using Groq API")
    else:
        st.caption("🖥️ Using local Ollama model: qwen3:4b-instruct-2507-q4_K_M")

    # ========================================================
    # BATCH WORKFLOW
    # ========================================================

    st.markdown(
        '<div class="batch-section-title">🧠 Batch Workflow</div>',
        unsafe_allow_html=True,
    )

    workflow = [
        ("📄", "Upload CSV"),
        ("🔍", "Validate Records"),
        ("🤖", "Evaluate"),
        ("📊", "Generate Results"),
    ]

    workflow_html = '<div class="batch-workflow">'
    for i, (icon, label) in enumerate(workflow):
        workflow_html += f'''
            <div class="batch-workflow-item">
                <div class="batch-workflow-card">
                    <span class="batch-workflow-icon">{icon}</span>
                    <span class="batch-workflow-label">{html.escape(label)}</span>
                </div>
            </div>
        '''
        if i < len(workflow) - 1:
            workflow_html += '<span class="batch-workflow-arrow">→</span>'
    workflow_html += '</div>'
    st.markdown(workflow_html, unsafe_allow_html=True)

    # ========================================================
    # LOAD / VALIDATE CSV
    # ========================================================

    if batch_df is not None:
        try:
            required_columns = ["question", "ai_response"]
            missing_columns = [col for col in required_columns if col not in batch_df.columns]

            if missing_columns:
                st.error("Missing required columns: " + ", ".join(missing_columns))
                batch_df = None
            else:
                st.success(f"CSV loaded successfully — {len(batch_df)} records found.")

                ready_label = "record" if len(batch_df) == 1 else "records"
                st.markdown(
                    f'<div class="batch-ready-status">'
                    f'<span class="ready-dot"></span>'
                    f'<span>{len(batch_df)} {ready_label} ready for evaluation</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("👀 Preview Dataset"):
                    st.dataframe(
                        batch_df.head(10),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.caption(
                        f"Showing up to 10 of {len(batch_df)} records. "
                        "Required: question, ai_response."
                    )

        except Exception as e:
            st.error(f"Unable to read CSV file: {e}")
            batch_df = None

    # ========================================================
    # RUN BATCH
    # ========================================================

    if st.button(
        "▶️ Run Batch Evaluation",
        type="primary",
        use_container_width=True,
    ):

        set_llm_provider(
            batch_llm_provider.lower()
        )

        # ----------------------------------------------------
        # VALIDATE CSV
        # ----------------------------------------------------

        if batch_df is None:

            st.error(
                "Please upload a valid CSV file first."
            )

        else:

            batch_results = []

            total_rows = len(
                batch_df
            )

            quota_exhausted = False

            progress_bar = st.progress(
                0
            )

            status_text = st.empty()

            # =================================================
            # PROCESS RECORDS
            # =================================================

            for index, row in enumerate(
                batch_df.to_dict("records"),
                start=1
            ):

                response_id = index




                # ------------------------------------------------
                # SKIP AFTER QUOTA
                # ------------------------------------------------

                if quota_exhausted:

                    batch_results.append({

                        "Response ID":
                            response_id,

                        "Question":
                            str(
                                row.get(
                                    "question",
                                    "",
                                )
                            ),

                        "AI Response":
                            str(
                                row.get(
                                    "ai_response",
                                    "",
                                )
                            ),

                        "Reference Answer":
                            "",

                        "Reference Document":
                            None,

                        "Relevance": 0,
                        "Accuracy": 0,
                        "Completeness": 0,
                        "Hallucination": 0,
                        "Overall Score": 0,

                        "Verdict":
                            "Skipped",

                        "Risk Level":
                            "Unknown",

                        "Status":
                            "Skipped - API Quota Exhausted",

                        "Error":
                            (
                                "Not evaluated because "
                                "the API quota/rate limit "
                                "was already exhausted "
                                "by an earlier record."
                            ),

                        "Relevance Result": {},
                        "Accuracy Result": {},
                        "Completeness Result": {},
                        "Hallucination Result": {},
                        "Verdict Result": {},
                    })

                    progress_bar.progress(
                        min(response_id / total_rows, 1.0)
                    )

                    continue

                # =================================================
                # PROCESS CURRENT RECORD
                # =================================================

                try:

                    row_question = str(
                        row.get(
                            "question",
                            "",
                        )
                    ).strip()

                    row_response = str(
                        row.get(
                            "ai_response",
                            "",
                        )
                    ).strip()

                    # ------------------------------------------------
                    # QUESTION VALIDATION
                    # ------------------------------------------------

                    if (
                        not row_question
                        or row_question.lower()
                        == "nan"
                    ):

                        batch_results.append({

                            "Response ID":
                                response_id,

                            "Question":
                                "",

                            "AI Response":
                                row_response,

                            "Reference Answer":
                                "",

                            "Reference Document":
                                None,

                            "Relevance": 0,
                            "Accuracy": 0,
                            "Completeness": 0,
                            "Hallucination": 0,
                            "Overall Score": 0,

                            "Verdict":
                                "Error",

                            "Risk Level":
                                "Unknown",

                            "Status":
                                "Invalid Input",

                            "Error":
                                "Question is missing.",

                            "Relevance Result": {},
                            "Accuracy Result": {},
                            "Completeness Result": {},
                            "Hallucination Result": {},
                            "Verdict Result": {},
                        })

                        progress_bar.progress(
                            min(response_id / total_rows, 1.0)
                        )

                        continue

                    # ------------------------------------------------
                    # RESPONSE VALIDATION
                    # ------------------------------------------------

                    if (
                        not row_response
                        or row_response.lower()
                        == "nan"
                    ):

                        batch_results.append({

                            "Response ID":
                                response_id,

                            "Question":
                                row_question,

                            "AI Response":
                                "",

                            "Reference Answer":
                                "",

                            "Reference Document":
                                None,

                            "Relevance": 0,
                            "Accuracy": 0,
                            "Completeness": 0,
                            "Hallucination": 0,
                            "Overall Score": 0,

                            "Verdict":
                                "Error",

                            "Risk Level":
                                "Unknown",

                            "Status":
                                "Invalid Input",

                            "Error":
                                "AI response is missing.",

                            "Relevance Result": {},
                            "Accuracy Result": {},
                            "Completeness Result": {},
                            "Hallucination Result": {},
                            "Verdict Result": {},
                        })

                        progress_bar.progress(
                            min(response_id / total_rows, 1.0)
                        )

                        continue

                    # ------------------------------------------------
                    # REFERENCE ANSWER
                    # ------------------------------------------------

                    row_reference_answer = ""

                    if (
                        "reference_answer"
                        in batch_df.columns
                    ):

                        value = row[
                            "reference_answer"
                        ]

                        if pd.notna(value):

                            row_reference_answer = (
                                str(value).strip()
                            )

                    status_text.write(
                        f"🔄 Processing record "
                        f"{response_id} of "
                        f"{total_rows}..."
                    )

                    # ------------------------------------------------
                    # KNOWLEDGE BASE
                    # ------------------------------------------------

                    retrieved_results = retrieve(
                        row_question,
                        top_k=5,
                    )

                    reference_context = (
                        "\n\n".join(
                            result[
                                "chunk"
                            ].page_content
                            for result in retrieved_results
                        )
                    )

                    # ------------------------------------------------
                    # EVALUATION
                    # ------------------------------------------------

                    evaluation = evaluate_response(
                        question=row_question,
                        ai_response=row_response,
                        reference_context=reference_context,
                        reference_answer=row_reference_answer,
                    )

                    # ------------------------------------------------
                    # AGENT RESULTS
                    # ------------------------------------------------

                    accuracy = (
                        evaluation.get(
                            "accuracy",
                            {},
                        )
                        or {}
                    )

                    completeness = (
                        evaluation.get(
                            "completeness",
                            {},
                        )
                        or {}
                    )

                    relevance = (
                        evaluation.get(
                            "relevance",
                            {},
                        )
                        or {}
                    )

                    hallucination = (
                        evaluation.get(
                            "hallucination",
                            {},
                        )
                        or {}
                    )

                    verdict = (
                        evaluation.get(
                            "verdict",
                            {},
                        )
                        or {}
                    )

                    # ------------------------------------------------
                    # SCORE
                    # ------------------------------------------------

                    risk_info = get_risk_level(
                        hallucination
                    )

                    _, _, overall_score = (
                        format_verdict_summary(
                            verdict
                        )
                    )

                    # ------------------------------------------------
                    # SUCCESS
                    # ------------------------------------------------

                    batch_results.append({

                        "Response ID":
                            response_id,

                        "Question":
                            row_question,

                        "AI Response":
                            row_response,

                        "Reference Answer":
                            row_reference_answer,

                        "Reference Document":
                            None,

                        "Relevance":
                            float(
                                relevance.get(
                                    "score",
                                    0,
                                )
                                or 0
                            ),

                        "Accuracy":
                            float(
                                accuracy.get(
                                    "score",
                                    0,
                                )
                                or 0
                            ),

                        "Completeness":
                            float(
                                completeness.get(
                                    "score",
                                    0,
                                )
                                or 0
                            ),

                        "Hallucination":
                            float(
                                hallucination.get(
                                    "score",
                                    0,
                                )
                                or 0
                            ),

                        "Overall Score":
                            float(
                                overall_score
                            ),

                        "Verdict":
                            verdict.get(
                                "verdict",
                                "Unknown",
                            ),

                        "Risk Level":
                            risk_info["level"],

                        "Status":
                            "Success",

                        "Error":
                            "",

                        "Relevance Result":
                            relevance,

                        "Accuracy Result":
                            accuracy,

                        "Completeness Result":
                            completeness,

                        "Hallucination Result":
                            hallucination,

                        "Verdict Result":
                            verdict,
                    })

                # =================================================
                # ERROR HANDLING
                # =================================================

                except Exception as e:

                    error_message = str(e)

                    error_lower = (
                        error_message.lower()
                    )

                    is_quota_error = (
                        "429"
                        in error_message
                        or
                        "resource_exhausted"
                        in error_lower
                        or
                        "quota"
                        in error_lower
                        or
                        "rate limit"
                        in error_lower
                        or
                        "rate_limit"
                        in error_lower
                    )

                    is_service_unavailable = (
                        "503"
                        in error_message
                        or
                        "service unavailable"
                        in error_lower
                        or
                        "temporarily unavailable"
                        in error_lower
                        or
                        "overloaded"
                        in error_lower
                    )

                    is_api_error = (
                        is_quota_error
                        or
                        is_service_unavailable
                        or
                        "api"
                        in error_lower
                        or
                        "gemini"
                        in error_lower
                    )

                    if is_quota_error:

                        quota_exhausted = True

                    if is_quota_error:

                        error_status = (
                            "Failed - API Quota / "
                            "Rate Limit"
                        )

                    elif is_service_unavailable:

                        error_status = (
                            "Failed - API Service "
                            "Unavailable"
                        )

                    elif is_api_error:

                        error_status = (
                            "Failed - API Error"
                        )

                    else:

                        error_status = (
                            "Failed - Evaluation Error"
                        )

                    batch_results.append({

                        "Response ID":
                            response_id,

                        "Question":
                            str(
                                row.get(
                                    "question",
                                    "",
                                )
                            ),

                        "AI Response":
                            str(
                                row.get(
                                    "ai_response",
                                    "",
                                )
                            ),

                        "Reference Answer":
                            "",

                        "Reference Document":
                            None,

                        "Relevance": 0,
                        "Accuracy": 0,
                        "Completeness": 0,
                        "Hallucination": 0,
                        "Overall Score": 0,

                        "Verdict":
                            "Error",

                        "Risk Level":
                            "Unknown",

                        "Status":
                            error_status,

                        "Error":
                            error_message,

                        "Relevance Result": {},
                        "Accuracy Result": {},
                        "Completeness Result": {},
                        "Hallucination Result": {},
                        "Verdict Result": {},
                    })

                progress_bar.progress(
                    min(response_id / total_rows, 1.0)
                )
                

            # =================================================
            # RESULTS DATAFRAME
            # =================================================

            results_df = pd.DataFrame(
                batch_results
            )

            # Persist results
            st.session_state.batch_results_data = (
                results_df.to_dict(
                    "records"
                )
            )

            # =================================================
            # CLASSIFY RESULTS
            # =================================================

            successful_results = (
                results_df[
                    results_df["Status"]
                    == "Success"
                ]
            )

            failed_results = (
                results_df[
                    results_df["Status"]
                    .astype(str)
                    .str.startswith(
                        "Failed"
                    )
                ]
            )

            skipped_results = (
                results_df[
                    results_df["Status"]
                    .astype(str)
                    .str.startswith(
                        "Skipped"
                    )
                ]
            )

            invalid_results = (
                results_df[
                    results_df["Status"]
                    .astype(str)
                    .str.startswith(
                        "Invalid"
                    )
                ]
            )

            # =================================================
            # SAVE BATCH TO HISTORY
            # =================================================

            batch_saved_id = None

            try:

                batch_saved_id = save_batch_history(
                    batch_results
                )

                st.session_state.batch_saved_id = (
                    batch_saved_id
                )

                if (
                    len(failed_results) == 0
                    and len(skipped_results) == 0
                    and len(invalid_results) == 0
                ):

                    status_text.success(
                        f"✅ Batch completed successfully. "
                        f"Batch ID: {batch_saved_id}"
                    )

                else:

                    status_text.warning(
                        f"⚠️ Batch saved to History with "
                        f"{len(failed_results)} failed, "
                        f"{len(skipped_results)} skipped, and "
                        f"{len(invalid_results)} invalid "
                        f"record(s). "
                        f"Batch ID: {batch_saved_id}"
                    )

            except Exception as e:

                st.error(
                    "Batch evaluation completed, "
                    "but History could not be saved: "
                    f"{e}"
                )


    # ========================================================
    # RESTORE / RENDER BATCH RESULTS AFTER NAVIGATION
    # ========================================================

    if st.session_state.batch_results_data is not None:

        render_batch_results(
            st.session_state.batch_results_data
        )


elif page == "History":

    st.markdown(
        '<div class="section-title">'
        '🕘 Evaluation History'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Completed batches are stored in History, including "
        "successful, failed, skipped, and invalid records."
    )

    history_type = st.radio(
        "History Type",
        [
            "Single Evaluations",
            "Batch Evaluations",
        ],
        index=[
            "Single Evaluations",
            "Batch Evaluations",
        ].index(
            st.session_state.history_type_data
        ),
        horizontal=True,
    )

    st.session_state.history_type_data = (
        history_type
    )

    search_query = st.text_input(
        "🔎 Search History",
        placeholder="Search by question, AI response, reference answer, verdict, risk, timestamp, or ID...",
        key="history_search_query",
    ).strip().lower()

    # ========================================================
    # SINGLE HISTORY
    # ========================================================

    if history_type == "Single Evaluations":

        history = get_single_history()
        if search_query:
            history = [item for item in history if search_query in " ".join([str(item.get("submission_id","")),str(item.get("timestamp","")),str(item.get("question","")),str(item.get("ai_response","")),str(item.get("reference_answer","")),str(item.get("source_document","")),str(item.get("risk_level","")),str(item.get("verdict_result",""))]).lower()]

        if not history:

            st.info(
                "No completed single evaluations "
                "in History yet."
            )

        else:

            # History summary
            total_history = len(history)
            pass_count = 0
            needs_count = 0
            fail_count = 0
            high_count = 0
            medium_count = 0
            low_count = 0

            for _item in history:
                _verdict = json_value(_item.get("verdict_result"))
                _name = str(_verdict.get("verdict", _verdict.get("status", ""))).strip().lower()
                if "pass" in _name:
                    pass_count += 1
                elif "need" in _name:
                    needs_count += 1
                elif "fail" in _name:
                    fail_count += 1

                _risk = str(_item.get("risk_level", "")).strip().lower().replace(" risk", "")
                if _risk == "high":
                    high_count += 1
                elif _risk == "medium":
                    medium_count += 1
                elif _risk == "low":
                    low_count += 1

            k1, k2, k3, k4 = st.columns(4)

            with k1:
                st.markdown(
                    f'<div class="history-kpi"><div class="history-kpi-value">{total_history}</div><div class="history-kpi-label">Completed Evaluations</div></div>',
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    f'<div class="history-kpi"><div class="history-kpi-value">🟢 {pass_count}</div><div class="history-kpi-label">Pass</div></div>',
                    unsafe_allow_html=True,
                )
            with k3:
                st.markdown(
                    f'<div class="history-kpi"><div class="history-kpi-value">🟠 {needs_count}</div><div class="history-kpi-label">Needs Improvement</div></div>',
                    unsafe_allow_html=True,
                )
            with k4:
                st.markdown(
                    f'<div class="history-kpi"><div class="history-kpi-value">🔴 {fail_count}</div><div class="history-kpi-label">Fail</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="history-section-label">Evaluation Records</div>', unsafe_allow_html=True)

            for item in history:

                verdict = json_value(
                    item.get(
                        "verdict_result"
                    )
                )

                relevance = json_value(
                    item.get(
                        "relevance_result"
                    )
                )

                accuracy = json_value(
                    item.get(
                        "accuracy_result"
                    )
                )

                completeness = json_value(
                    item.get(
                        "completeness_result"
                    )
                )

                hallucination = json_value(
                    item.get(
                        "hallucination_result"
                    )
                )

                verdict_name = verdict.get(
                    "verdict",
                    verdict.get(
                        "status",
                        "Completed",
                    ),
                )

                score = float(
                    item.get(
                        "overall_score"
                    )
                    or 0
                )

                risk = (
                    item.get(
                        "risk_level"
                    )
                    or "Unknown"
                )

                verdict_lower = str(verdict_name).strip().lower()
                if "pass" in verdict_lower:
                    verdict_icon = "🟢"
                elif "need" in verdict_lower:
                    verdict_icon = "🟠"
                elif "fail" in verdict_lower:
                    verdict_icon = "🔴"
                else:
                    verdict_icon = "⚪"

                risk_clean = str(risk).replace(" Risk", "").strip()
                risk_lower = risk_clean.lower()
                risk_icon = {
                    "high": "🔴",
                    "medium": "🟠",
                    "low": "🟢",
                }.get(risk_lower, "⚪")

                with st.expander(
                    f"🔍 {item['question'][:90]}  •  "
                    f"{verdict_icon} {verdict_name}  •  "
                    f"{score:.1f}/5  •  "
                    f"{risk_icon} {risk_clean or 'Unknown'} Risk"
                ):

                    st.write(
                        f"**Evaluation ID:** "
                        f"{item['submission_id']}"
                    )

                    st.write(
                        f"**Timestamp:** "
                        f"{item['timestamp']}"
                    )

                    st.markdown(
                        "### Question"
                    )

                    st.write(
                        item["question"]
                    )

                    st.markdown(
                        "### AI Response"
                    )

                    st.write(
                        item["ai_response"]
                    )

                    st.markdown(
                        "### Reference Answer"
                    )

                    st.write(
                        item.get(
                            "reference_answer"
                        )
                        or
                        "No reference answer provided."
                    )

                    st.markdown(
                        "### Reference Document"
                    )

                    st.write(
                        item.get(
                            "source_document"
                        )
                        or
                        "No reference document provided."
                    )

                    st.markdown(
                        "### Final Result"
                    )

                    m1, m2, m3 = st.columns(3)

                    m1.metric(
                        "Verdict",
                        verdict_name,
                    )

                    m2.metric(
                        "Overall Score",
                        f"{score:.1f}/5",
                    )

                    m3.metric(
                        "Risk",
                        risk,
                    )

                    render_agent_results(
                        relevance,
                        accuracy,
                        completeness,
                        hallucination,
                        get_risk_level(
                            hallucination
                        ),
                    )

                    with st.expander(
                        "⚖️ Verdict Agent Details"
                    ):

                        st.json(
                            verdict
                        )

    # ========================================================
    # BATCH HISTORY
    # ========================================================

    else:

        batches = get_batch_history()
        if search_query:
            matching_batches = []
            for batch in batches:
                batch_text = " ".join([
                    str(batch.get("batch_id", "")),
                    str(batch.get("timestamp", "")),
                    str(batch.get("status", "")),
                    str(batch.get("total_records", "")),
                    str(batch.get("pass_count", "")),
                    str(batch.get("needs_improvement_count", "")),
                    str(batch.get("fail_count", "")),
                    str(batch.get("hallucination_count", "")),
                    str(batch.get("avg_relevance", "")),
                    str(batch.get("avg_accuracy", "")),
                    str(batch.get("avg_completeness", "")),
                    str(batch.get("avg_hallucination", "")),
                ]).lower()

                record_text = ""
                try:
                    records_for_search = get_batch_records(batch["batch_id"])
                    record_text = " ".join(
                        " ".join([
                            str(record.get("submission_id", "")),
                            str(record.get("question", "")),
                            str(record.get("ai_response", "")),
                            str(record.get("reference_answer", "")),
                            str(record.get("source_document", "")),
                            str(record.get("status", "")),
                        ])
                        for record in records_for_search
                    ).lower()
                except Exception:
                    record_text = ""

                if search_query in batch_text or search_query in record_text:
                    matching_batches.append(batch)

            batches = matching_batches

        if not batches:

            st.info(
                "No completed batch evaluations "
                "in History yet."
            )

        else:

            total_batches = len(batches)
            total_records = sum(int(batch.get("total_records", 0) or 0) for batch in batches)
            total_pass = sum(int(batch.get("pass_count", 0) or 0) for batch in batches)
            total_fail = sum(int(batch.get("fail_count", 0) or 0) for batch in batches)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(
                    f'<div class="history-kpi"><div class="history-kpi-value">{total_batches}</div><div class="history-kpi-label">Batches</div></div>',
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    f'<div class="history-kpi"><div class="history-kpi-value">{total_records}</div><div class="history-kpi-label">Total Records</div></div>',
                    unsafe_allow_html=True,
                )
            with k3:
                st.markdown(
                    f'<div class="history-kpi"><div class="history-kpi-value">🟢 {total_pass}</div><div class="history-kpi-label">Pass</div></div>',
                    unsafe_allow_html=True,
                )
            with k4:
                st.markdown(
                    f'<div class="history-kpi"><div class="history-kpi-value">🔴 {total_fail}</div><div class="history-kpi-label">Fail</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="history-section-label">Batch Records</div>', unsafe_allow_html=True)

            for batch in batches:

                batch_id = batch[
                    "batch_id"
                ]

                batch_status = batch.get(
                    "status",
                    "Unknown",
                )

                with st.expander(
                    f"📊 Batch #{batch_id}  |  "
                    f"{batch['total_records']} records  |  "
                    f"{batch['timestamp']}"
                ):

                    st.write(
                        f"**Batch ID:** "
                        f"{batch_id}"
                    )

                    st.write(
                        f"**Timestamp:** "
                        f"{batch['timestamp']}"
                    )

                    st.write(
                        f"**Total Records:** "
                        f"{batch['total_records']}"
                    )

                    st.write(
                        f"**Batch Status:** "
                        f"{batch_status}"
                    )

                    m1, m2, m3, m4 = st.columns(4)

                    m1.metric(
                        "Avg Relevance",
                        f"{batch['avg_relevance']:.2f}/5",
                    )

                    m2.metric(
                        "Avg Accuracy",
                        f"{batch['avg_accuracy']:.2f}/5",
                    )

                    m3.metric(
                        "Avg Completeness",
                        f"{batch['avg_completeness']:.2f}/5",
                    )

                    m4.metric(
                        "Avg Hallucination",
                        f"{batch['avg_hallucination']:.2f}/5",
                    )

                    st.markdown(
                        "### Verdict Distribution"
                    )

                    d1, d2, d3, d4 = st.columns(4)

                    d1.metric(
                        "Pass",
                        batch["pass_count"],
                    )

                    d2.metric(
                        "Needs Improvement",
                        batch[
                            "needs_improvement_count"
                        ],
                    )

                    d3.metric(
                        "Fail",
                        batch["fail_count"],
                    )

                    d4.metric(
                        "Hallucination Detected",
                        batch[
                            "hallucination_count"
                        ],
                    )

                    records = get_batch_records(
                        batch_id
                    )

                    if records:

                        st.markdown(
                            "### Batch Records"
                        )

                        table_rows = []

                        for record in records:

                            verdict = json_value(
                                record.get(
                                    "verdict_result"
                                )
                            )

                            table_rows.append({

                                "ID":
                                    record[
                                        "submission_id"
                                    ],

                                "Question":
                                    record[
                                        "question"
                                    ],

                                "Overall Score":
                                    record[
                                        "overall_score"
                                    ],

                                "Verdict":
                                    verdict.get(
                                        "verdict",
                                        "Unknown",
                                    ),

                                "Risk":
                                    record.get(
                                        "risk_level",
                                        "Unknown",
                                    ),

                                "Status":
                                    record.get(
                                        "status",
                                        "Unknown",
                                    ),
                            })

                        st.dataframe(
                            pd.DataFrame(
                                table_rows
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                        # =================================================
                        # DETAILED RECORDS
                        # =================================================

                        st.markdown(
                            "### Detailed Records"
                        )

                        for record in records:

                            record_status = str(
                                record.get(
                                    "status",
                                    "Unknown",
                                )
                            )

                            verdict = json_value(
                                record.get(
                                    "verdict_result"
                                )
                            )

                            relevance = json_value(
                                record.get(
                                    "relevance_result"
                                )
                            )

                            accuracy = json_value(
                                record.get(
                                    "accuracy_result"
                                )
                            )

                            completeness = json_value(
                                record.get(
                                    "completeness_result"
                                )
                            )

                            hallucination = json_value(
                                record.get(
                                    "hallucination_result"
                                )
                            )

                            verdict_name = (
                                verdict.get(
                                    "verdict",
                                    "Unknown",
                                )
                            )

                            score = float(
                                record.get(
                                    "overall_score",
                                    0,
                                )
                                or 0
                            )

                            with st.expander(
                                f"Record "
                                f"{record['submission_id']} "
                                f"• "
                                f"{verdict_name} "
                                f"• "
                                f"{score:.1f}/5"
                            ):

                                st.write(
                                    f"**Question:** "
                                    f"{record['question']}"
                                )

                                st.write(
                                    f"**AI Response:** "
                                    f"{record['ai_response']}"
                                )

                                st.write(
                                    f"**Reference Answer:** "
                                    f"{record.get('reference_answer') or 'Not provided'}"
                                )

                                st.write(
                                    f"**Status:** "
                                    f"{record_status}"
                                )

                                st.write(
                                    f"**Risk:** "
                                    f"{record.get('risk_level', 'Unknown')}"
                                )

                                # -----------------------------------------
                                # NON-SUCCESS RECORD
                                # -----------------------------------------

                                if record_status != "Success":

                                    error_message = (
                                        record.get(
                                            "error_message",
                                            "",
                                        )
                                        or ""
                                    )

                                    if error_message:

                                        st.error(
                                            f"**Error:** "
                                            f"{error_message}"
                                        )

                                    if record_status.startswith(
                                        "Skipped"
                                    ):

                                        st.info(
                                            "This record was skipped "
                                            "and was not evaluated."
                                        )

                                    elif record_status.startswith(
                                        "Invalid"
                                    ):

                                        st.warning(
                                            "This record had invalid "
                                            "input and was not evaluated."
                                        )

                                    else:

                                        st.warning(
                                            "This record was not "
                                            "successfully evaluated."
                                        )

                                # -----------------------------------------
                                # SUCCESS RECORD
                                # -----------------------------------------

                                else:

                                    render_agent_results(
                                        relevance,
                                        accuracy,
                                        completeness,
                                        hallucination,
                                        get_risk_level(
                                            hallucination
                                        ),
                                    )