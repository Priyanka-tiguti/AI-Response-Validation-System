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
    page_icon="🛡️",
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

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ AI Response Validation System"
)

st.caption(
    "Agent Workspace · Independent validation agents · "
    "Weighted final verdict"
)


# ============================================================
# NAVIGATION
# ============================================================

nav_col1, nav_col2, nav_col3, nav_col4 = (
    st.columns(4)
)

nav_items = [
    (
        "🏠 Dashboard",
        "Dashboard",
    ),
    (
        "🔍 Single Evaluation",
        "Single Evaluation",
    ),
    (
        "📊 Batch Evaluation",
        "Batch Evaluation",
    ),
    (
        "🕘 History",
        "History",
    ),
]

for col, (
    label,
    value,
) in zip(
    [
        nav_col1,
        nav_col2,
        nav_col3,
        nav_col4,
    ],
    nav_items,
):

    with col:

        is_active = (
            st.session_state.page
            == value
        )

        button_label = (
            f"●  {label}"
            if is_active
            else label
        )

        if st.button(
            button_label,
            key=f"nav_{value}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):

            st.session_state.page = value

            st.rerun()


page = st.session_state.page



def generate_batch_pdf(results_df):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "BatchTitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "BatchHeadingStyle",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8,
    )

    subheading_style = ParagraphStyle(
        "BatchSubHeadingStyle",
        parent=styles["Heading3"],
        fontSize=10,
        spaceBefore=8,
        spaceAfter=5,
    )

    body_style = ParagraphStyle(
        "BatchBodyStyle",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
        spaceAfter=5,
    )

    small_style = ParagraphStyle(
        "BatchSmallStyle",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=10,
        spaceAfter=3,
    )

    def pdf_value(value, default="Not available."):
        if value is None:
            return default
        text_value = str(value).strip()
        return text_value if text_value else default

    def pdf_paragraph(label, value, style=body_style):
        return Paragraph(
            f"<b>{html.escape(label)}</b><br/>"
            f"{html.escape(pdf_value(value))}",
            style,
        )

    def result_dict(row, column):
        return json_value(row.get(column, {}))

    story = []

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
            "Batch Evaluation Report",
            heading_style,
        )
    )

    story.append(Spacer(1, 10))

    # ========================================================
    # STATISTICS
    # ========================================================

    total_records = len(results_df)

    successful = results_df[
        results_df["Status"] == "Success"
    ]

    failed = results_df[
        results_df["Status"].astype(str).str.startswith("Failed")
    ]

    skipped = results_df[
        results_df["Status"].astype(str).str.startswith("Skipped")
    ]

    invalid = results_df[
        results_df["Status"].astype(str).str.startswith("Invalid")
    ]

    for label, value in [
        ("Total Records", total_records),
        ("Successful", len(successful)),
        ("Failed", len(failed)),
        ("Skipped", len(skipped)),
        ("Invalid", len(invalid)),
    ]:
        story.append(
            Paragraph(
                f"<b>{label}:</b> {value}",
                body_style,
            )
        )

    story.append(Spacer(1, 10))

    # ========================================================
    # AVERAGE SCORES
    # ========================================================

    if not successful.empty:
        story.append(
            Paragraph(
                "Average Agent Scores",
                heading_style,
            )
        )

        avg_data = [
            ["Agent", "Average Score"],
            [
                "Relevance",
                f"{successful['Relevance'].mean():.2f}/5",
            ],
            [
                "Accuracy",
                f"{successful['Accuracy'].mean():.2f}/5",
            ],
            [
                "Completeness",
                f"{successful['Completeness'].mean():.2f}/5",
            ],
            [
                "Hallucination",
                f"{successful['Hallucination'].mean():.2f}/5",
            ],
        ]

        avg_table = Table(
            avg_data,
            colWidths=[250, 150],
        )

        avg_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(avg_table)
        story.append(Spacer(1, 15))

    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    story.append(
        Paragraph(
            "Evaluation Results",
            heading_style,
        )
    )

    table_data = [
        [
            "ID",
            "Relevance",
            "Accuracy",
            "Completeness",
            "Hallucination",
            "Score",
            "Verdict",
            "Risk",
            "Status",
        ]
    ]

    for _, row in results_df.iterrows():
        table_data.append(
            [
                str(row.get("Response ID", "")),
                str(row.get("Relevance", "")),
                str(row.get("Accuracy", "")),
                str(row.get("Completeness", "")),
                str(row.get("Hallucination", "")),
                str(row.get("Overall Score", "")),
                str(row.get("Verdict", "")),
                str(row.get("Risk Level", "")),
                str(row.get("Status", "")),
            ]
        )

    results_table = Table(
        table_data,
        repeatRows=1,
        colWidths=[
            35,
            45,
            45,
            55,
            55,
            40,
            55,
            50,
            70,
        ],
    )

    results_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(results_table)
    story.append(Spacer(1, 18))

    # ========================================================
    # DETAILED PER-RECORD ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "Detailed Record Analysis",
            heading_style,
        )
    )

    for _, row in results_df.iterrows():

        response_id = row.get("Response ID", "")
        status = str(row.get("Status", "Unknown"))

        story.append(
            Paragraph(
                f"Record {html.escape(str(response_id))}",
                subheading_style,
            )
        )

        # Input data
        story.append(
            Paragraph(
                "Evaluation Input",
                subheading_style,
            )
        )

        story.append(
            pdf_paragraph(
                "Question:",
                row.get("Question", ""),
            )
        )

        story.append(
            pdf_paragraph(
                "AI Response:",
                row.get("AI Response", ""),
            )
        )

        reference_answer = row.get(
            "Reference Answer",
            "",
        )

        story.append(
            pdf_paragraph(
                "Reference Answer:",
                reference_answer
                if pdf_value(reference_answer) != "Not available."
                else "No reference answer provided.",
            )
        )

        # Invalid / failed records do not have agent analysis.
        if status != "Success":

            story.append(
                Paragraph(
                    f"<b>Status:</b> {html.escape(status)}",
                    body_style,
                )
            )

            story.append(
                pdf_paragraph(
                    "Reason:",
                    row.get("Error", ""),
                )
            )

            story.append(Spacer(1, 8))
            continue

        relevance = result_dict(
            row,
            "Relevance Result",
        )
        accuracy = result_dict(
            row,
            "Accuracy Result",
        )
        completeness = result_dict(
            row,
            "Completeness Result",
        )
        hallucination = result_dict(
            row,
            "Hallucination Result",
        )
        verdict = result_dict(
            row,
            "Verdict Result",
        )

        risk_info = get_risk_level(
            hallucination
        )

        verdict_text, verdict_name, overall_score = (
            format_verdict_summary(verdict)
        )

        # Final result
        story.append(
            Paragraph(
                "Final Evaluation",
                subheading_style,
            )
        )

        final_data = [
            ["Metric", "Result"],
            [
                "System Verdict",
                pdf_value(verdict_name),
            ],
            [
                "Weighted Score",
                f"{overall_score:.1f} / 5.0",
            ],
            [
                "Hallucination Risk",
                pdf_value(risk_info["text"]),
            ],
            [
                "Status",
                "Success",
            ],
        ]

        final_table = Table(
            final_data,
            colWidths=[170, 260],
        )

        final_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )

        story.append(final_table)

        # Agent scores
        story.append(
            Paragraph(
                "Agent Scores and Analysis",
                subheading_style,
            )
        )

        agent_rows = [
            ["Agent", "Score", "Analysis"],
            [
                "Relevance",
                f"{float(relevance.get('score', 0) or 0):.1f}/5",
                pdf_value(
                    relevance.get(
                        "reasoning",
                        "No reasoning returned.",
                    )
                ),
            ],
            [
                "Accuracy",
                f"{float(accuracy.get('score', 0) or 0):.1f}/5",
                pdf_value(
                    accuracy.get(
                        "reasoning",
                        "No reasoning returned.",
                    )
                ),
            ],
            [
                "Completeness",
                f"{float(completeness.get('score', 0) or 0):.1f}/5",
                pdf_value(
                    completeness.get(
                        "reasoning",
                        "No reasoning returned.",
                    )
                ),
            ],
            [
                "Hallucination",
                f"{float(hallucination.get('score', 0) or 0):.1f}/5",
                pdf_value(
                    hallucination.get(
                        "reasoning",
                        "No reasoning returned.",
                    )
                ),
            ],
        ]

        agent_table_data = []

        for row_data in agent_rows:
            agent_table_data.append(
                [
                    Paragraph(
                        html.escape(str(row_data[0])),
                        small_style,
                    ),
                    Paragraph(
                        html.escape(str(row_data[1])),
                        small_style,
                    ),
                    Paragraph(
                        html.escape(str(row_data[2])),
                        small_style,
                    ),
                ]
            )

        agent_table = Table(
            agent_table_data,
            colWidths=[85, 55, 290],
            repeatRows=1,
        )

        agent_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("ALIGN", (1, 1), (1, -1), "CENTER"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )

        story.append(agent_table)

        # Supporting evidence
        evidence = accuracy.get(
            "supporting_evidence",
            [],
        ) or []

        if evidence:
            story.append(
                Paragraph(
                    "Supporting Evidence",
                    subheading_style,
                )
            )

            for item in evidence:
                story.append(
                    Paragraph(
                        "• " + html.escape(str(item)),
                        small_style,
                    )
                )

        # Missing components
        missing = completeness.get(
            "missing_components",
            [],
        ) or []

        if missing:
            story.append(
                Paragraph(
                    "Missing Components",
                    subheading_style,
                )
            )

            for item in missing:
                story.append(
                    Paragraph(
                        "• " + html.escape(str(item)),
                        small_style,
                    )
                )

        # Flagged hallucination claims
        flagged = hallucination.get(
            "flagged_claims",
            [],
        ) or []

        if flagged:
            story.append(
                Paragraph(
                    "Flagged Hallucination Claims",
                    subheading_style,
                )
            )

            for i, claim in enumerate(
                flagged,
                1,
            ):
                if isinstance(claim, dict):
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
                    claim_text = str(claim)
                    classification = "Flagged"
                    reason = (
                        "No additional explanation returned."
                    )

                story.append(
                    Paragraph(
                        f"<b>Claim #{i} — "
                        f"{html.escape(classification)}</b>",
                        small_style,
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

        # Verdict agent reasoning
        if verdict_text:
            story.append(
                Paragraph(
                    "Verdict Agent Analysis",
                    subheading_style,
                )
            )

            story.append(
                Paragraph(
                    html.escape(str(verdict_text)),
                    body_style,
                )
            )

        story.append(Spacer(1, 12))

    # ========================================================
    # CREATE PDF
    # ========================================================

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# RENDER PERSISTED BATCH RESULTS
# ============================================================

def render_batch_results(batch_results_data):

    if not batch_results_data:
        return

    results_df = pd.DataFrame(batch_results_data)

    if results_df.empty:
        return

    total_rows = len(results_df)

    successful_results = results_df[
        results_df["Status"] == "Success"
    ]

    failed_results = results_df[
        results_df["Status"].astype(str).str.startswith("Failed")
    ]

    skipped_results = results_df[
        results_df["Status"].astype(str).str.startswith("Skipped")
    ]

    invalid_results = results_df[
        results_df["Status"].astype(str).str.startswith("Invalid")
    ]

    # =================================================
    # BATCH RESULTS
    # =================================================

    st.markdown(
        '<div class="section-title">'
        '📋 Batch Results'
        '</div>',
        unsafe_allow_html=True,
    )

    display_columns = [
        "Response ID",
        "Question",
        "Relevance",
        "Accuracy",
        "Completeness",
        "Hallucination",
        "Overall Score",
        "Verdict",
        "Risk Level",
        "Status",
    ]

    st.dataframe(
        results_df[
            display_columns
        ],
        use_container_width=True,
        hide_index=True,
    )

    # =================================================
    # DOWNLOAD
    # =================================================

    csv_data = (
        results_df[
            display_columns
        ]
        .to_csv(
            index=False
        )
        .encode("utf-8-sig")
    )

    st.download_button(
        label="📥 Download Batch Results",
        data=csv_data,
        file_name=(
            "batch_evaluation_results.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )
    
    pdf_data = generate_batch_pdf(results_df)

    st.download_button(
        label="📄 Download PDF Batch Report",
        data=pdf_data,
        file_name="batch_evaluation_report.pdf",
        mime="application/pdf",
        key="batch_pdf_download",
        type="primary",
        use_container_width=True,
    )

    # =================================================
    # STATISTICS
    # =================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Batch Statistics'
        '</div>',
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4, s5 = st.columns(5)

    s1.metric(
        "Total Records",
        total_rows,
    )

    s2.metric(
        "Successful",
        len(successful_results),
    )

    s3.metric(
        "Failed",
        len(failed_results),
    )

    s4.metric(
        "Skipped",
        len(skipped_results),
    )

    s5.metric(
        "Invalid Input",
        len(invalid_results),
    )

    # =================================================
    # SUCCESS STATISTICS
    # =================================================

    if not successful_results.empty:

        avg_relevance = (
            successful_results[
                "Relevance"
            ].mean()
        )

        avg_accuracy = (
            successful_results[
                "Accuracy"
            ].mean()
        )

        avg_completeness = (
            successful_results[
                "Completeness"
            ].mean()
        )

        avg_hallucination = (
            successful_results[
                "Hallucination"
            ].mean()
        )

        st.markdown(
            "### Average Agent Scores"
        )

        a1, a2, a3, a4 = st.columns(4)

        a1.metric(
            "Avg Relevance",
            f"{avg_relevance:.2f}/5",
        )

        a2.metric(
            "Avg Accuracy",
            f"{avg_accuracy:.2f}/5",
        )

        a3.metric(
            "Avg Completeness",
            f"{avg_completeness:.2f}/5",
        )

        a4.metric(
            "Avg Hallucination",
            f"{avg_hallucination:.2f}/5",
        )

        st.markdown(
            "### Verdict Distribution"
        )

        v1, v2, v3, v4 = st.columns(4)

        v1.metric(
            "Pass",
            int(
                (
                    successful_results[
                        "Verdict"
                    ]
                    == "Pass"
                ).sum()
            ),
        )

        v2.metric(
            "Needs Improvement",
            int(
                (
                    successful_results[
                        "Verdict"
                    ]
                    == "Needs Improvement"
                ).sum()
            ),
        )

        v3.metric(
            "Fail",
            int(
                (
                    successful_results[
                        "Verdict"
                    ]
                    == "Fail"
                ).sum()
            ),
        )

        v4.metric(
            "Hallucination Detected",
            int(
                (
                    successful_results[
                        "Hallucination"
                    ]
                    < 5
                ).sum()
            ),
        )

    # =================================================
    # DETAILED PER-RECORD ANALYSIS
    # =================================================

    st.markdown(
        "### 🔎 Detailed Record Analysis"
    )

    st.caption(
        "Each successful record shows the question, AI response, "
        "reference answer, four independent agent analyses, "
        "and the final Verdict Agent reasoning."
    )

    for _, record in results_df.iterrows():

        response_id = record.get(
            "Response ID",
            "",
        )

        status = str(
            record.get(
                "Status",
                "Unknown",
            )
        )

        with st.expander(
            f"Record {response_id} — {status}",
            expanded=False,
        ):

            st.markdown("**Question**")
            st.write(
                record.get(
                    "Question",
                    "",
                )
                or "Not available."
            )

            st.markdown("**AI Response**")
            ai_response_value = record.get(
                "AI Response",
                "",
            )

            st.write(
                ai_response_value
                if ai_response_value
                else "Not provided."
            )

            st.markdown("**Reference Answer**")
            reference_value = record.get(
                "Reference Answer",
                "",
            )

            st.write(
                reference_value
                if reference_value
                else "No reference answer provided."
            )

            if status != "Success":

                st.warning(
                    f"**Status:** {status}"
                )

                st.write(
                    f"**Reason:** "
                    f"{record.get('Error', '') or 'No error details provided.'}"
                )

                continue

            relevance = json_value(
                record.get(
                    "Relevance Result",
                    {},
                )
            )

            accuracy = json_value(
                record.get(
                    "Accuracy Result",
                    {},
                )
            )

            completeness = json_value(
                record.get(
                    "Completeness Result",
                    {},
                )
            )

            hallucination = json_value(
                record.get(
                    "Hallucination Result",
                    {},
                )
            )

            verdict = json_value(
                record.get(
                    "Verdict Result",
                    {},
                )
            )

            detail_cols = st.columns(4)

            agent_data = [
                (
                    detail_cols[0],
                    "🎯 Relevance",
                    relevance,
                ),
                (
                    detail_cols[1],
                    "✓ Accuracy",
                    accuracy,
                ),
                (
                    detail_cols[2],
                    "◈ Completeness",
                    completeness,
                ),
                (
                    detail_cols[3],
                    "⚠️ Hallucination",
                    hallucination,
                ),
            ]

            for col, name, data in agent_data:

                with col:

                    score = float(
                        data.get(
                            "score",
                            0,
                        )
                        or 0
                    )

                    st.metric(
                        name,
                        f"{score:.1f}/5",
                    )

            st.markdown("**Agent Analysis**")

            st.markdown(
                "**🎯 Relevance Agent:** "
                + str(
                    relevance.get(
                        "reasoning",
                        "No reasoning returned.",
                    )
                )
            )

            st.markdown(
                "**✓ Accuracy Agent:** "
                + str(
                    accuracy.get(
                        "reasoning",
                        "No reasoning returned.",
                    )
                )
            )

            evidence = accuracy.get(
                "supporting_evidence",
                [],
            ) or []

            if evidence:
                st.markdown("**Supporting Evidence**")
                for item in evidence:
                    st.info(str(item))

            st.markdown(
                "**◈ Completeness Agent:** "
                + str(
                    completeness.get(
                        "reasoning",
                        "No reasoning returned.",
                    )
                )
            )

            missing = completeness.get(
                "missing_components",
                [],
            ) or []

            if missing:
                st.markdown("**Missing Components**")
                for item in missing:
                    st.warning(str(item))

            st.markdown(
                "**⚠️ Hallucination Detection Agent:** "
                + str(
                    hallucination.get(
                        "reasoning",
                        "No reasoning returned.",
                    )
                )
            )

            flagged = hallucination.get(
                "flagged_claims",
                [],
            ) or []

            if flagged:
                st.markdown(
                    f"**Flagged Claims:** {len(flagged)}"
                )

                for i, claim in enumerate(
                    flagged,
                    1,
                ):

                    if isinstance(
                        claim,
                        dict,
                    ):
                        claim_text = claim.get(
                            "claim",
                            "Not available.",
                        )
                        classification = claim.get(
                            "classification",
                            "Flagged",
                        )
                        reason = claim.get(
                            "reason",
                            "No additional explanation returned.",
                        )
                    else:
                        claim_text = claim
                        classification = "Flagged"
                        reason = (
                            "No additional explanation returned."
                        )

                    st.warning(
                        f"**Claim #{i} — {classification}**\n\n"
                        f"**Claim:** {claim_text}\n\n"
                        f"**Reason:** {reason}"
                    )
            else:
                st.success(
                    "No ungrounded or contradictory claims detected."
                )

            verdict_text, verdict_name, overall_score = (
                format_verdict_summary(verdict)
            )

            st.markdown("**⚖️ Verdict Agent**")

            v1, v2, v3 = st.columns(3)

            with v1:
                st.metric(
                    "System Verdict",
                    verdict_name,
                )

            with v2:
                st.metric(
                    "Weighted Score",
                    f"{overall_score:.1f}/5",
                )

            with v3:
                risk_info = get_risk_level(
                    hallucination
                )
                st.metric(
                    "Hallucination Risk",
                    f"{risk_info['icon']} "
                    f"{risk_info['text']}",
                )

            st.info(
                verdict_text
                if verdict_text
                else "No verdict analysis returned."
            )

    # =================================================
    # PROBLEM RECORDS
    # =================================================

    if (
        not failed_results.empty
        or not skipped_results.empty
        or not invalid_results.empty
    ):

        st.markdown(
            "### ⚠️ Failed / Skipped / Invalid Records"
        )

        problem_mask = (
            results_df[
                "Status"
            ]
            .astype(str)
            .str.startswith(
                (
                    "Failed",
                    "Skipped",
                    "Invalid",
                )
            )
        )

        problem_df = (
            results_df.loc[
                problem_mask,
                [
                    "Response ID",
                    "Question",
                    "Status",
                    "Error",
                ],
            ].copy()
        )

        st.dataframe(
            problem_df,
            use_container_width=True,
            hide_index=True,
        )

        # =================================================
        # ERROR DETAILS
        # =================================================

        st.markdown(
            "### 🔎 Error Details"
        )

        for _, problem in (
            problem_df.iterrows()
        ):

            response_id = (
                problem[
                    "Response ID"
                ]
            )

            status = str(
                problem["Status"]
            )

            error = str(
                problem["Error"]
            )

            # ------------------------------------------------
            # QUOTA
            # ------------------------------------------------

            if (
                "API Quota"
                in status
                or
                "429"
                in error
                or
                "quota"
                in error.lower()
                or
                "resource_exhausted"
                in error.lower()
                or
                "rate limit"
                in error.lower()
            ):

                st.error(
                    f"""
    🔴 **Record {response_id} — API Quota / Rate Limit Error**

    **Status:** {status}

    **Error:**

    {error}

    **What this means:**

    The API quota or rate limit was reached.
    The remaining records were not evaluated to avoid
    additional API requests.
    """
                )

            # ------------------------------------------------
            # 503
            # ------------------------------------------------

            elif (
                "503"
                in error
                or
                "service unavailable"
                in error.lower()
                or
                "temporarily unavailable"
                in error.lower()
                or
                "overloaded"
                in error.lower()
            ):

                st.warning(
                    f"""
    🟠 **Record {response_id} — API Service Unavailable**

    **Status:** {status}

    **Error:**

    {error}

    **What this means:**

    The model was temporarily unavailable or overloaded.
    You can retry the batch later.
    """
                )

            # ------------------------------------------------
            # INVALID INPUT
            # ------------------------------------------------

            elif status.startswith(
                "Invalid"
            ):

                st.warning(
                    f"""
    🟡 **Record {response_id} — Invalid Input**

    **Status:** {status}

    **Reason:**

    {error}
    """
                )

            # ------------------------------------------------
            # SKIPPED
            # ------------------------------------------------

            elif status.startswith(
                "Skipped"
            ):

                st.info(
                    f"""
    ⏭️ **Record {response_id} — Skipped**

    **Status:** {status}

    **Reason:**

    {error}
    """
                )

            # ------------------------------------------------
            # OTHER API ERROR
            # ------------------------------------------------

            elif "API" in status:

                st.error(
                    f"""
    🔴 **Record {response_id} — External API Error**

    **Status:** {status}

    **Error:**

    {error}
    """
                )

            # ------------------------------------------------
            # NORMAL ERROR
            # ------------------------------------------------

            else:

                st.error(
                    f"""
    ❌ **Record {response_id} — Evaluation Error**

    **Status:** {status}

    **Error:**

    {error}
    """
                )


    # ============================================================
    # HISTORY
    # ============================================================

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
        """
        <h2 style="
            color:#FFFFFF;
            font-size:28px;
            font-weight:700;
            margin:0 0 8px 0;
        ">
            📈 Evaluation Scoring Dashboard
        </h2>
        <div style="
            color:#AAB2C8;
            font-size:14px;
            margin-bottom:24px;
        ">
            Combined analysis of stored single and batch evaluation results.
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
        selected_verdict = st.selectbox(
            "Filter by Verdict",
            verdict_options,
            key="dashboard_verdict_filter",
        )

    with filter_col2:
        selected_batch = st.selectbox(
            "Filter by Evaluation Batch",
            batch_options,
            key="dashboard_batch_filter",
        )

    with filter_col3:
        selected_score = st.selectbox(
            "Filter by Overall Score",
            score_options,
            key="dashboard_score_filter",
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
        (
            "Total Evaluations",
            total_evaluations,
            "100%",
            "#FFFFFF",
        ),
        (
            "Pass",
            pass_count,
            f"{pct(pass_count, total_evaluations):.1f}%",
            "#6366F1",
        ),
        (
            "Needs Improvement",
            needs_count,
            f"{pct(needs_count, total_evaluations):.1f}%",
            "#10BFA3",
        ),
        (
            "Fail",
            fail_count,
            f"{pct(fail_count, total_evaluations):.1f}%",
            "#F45B45",
        ),
    ]

    cols = st.columns(4)

    for col, (
        label,
        value,
        percentage,
        value_color,
    ) in zip(cols, card_data):

        with col:
            st.markdown(
                f"""
                <div style="
                    background:linear-gradient(135deg,#151B32,#0D1222);
                    border:1px solid #252D4A;
                    border-radius:14px;
                    padding:17px 12px;
                    text-align:center;
                    min-height:116px;
                    box-shadow:0 4px 15px rgba(0,0,0,0.20);
                ">
                    <div style="
                        color:#AAB2C8;
                        font-size:13px;
                        font-weight:500;
                    ">{label}</div>
                    <div style="
                        color:{value_color};
                        font-size:29px;
                        font-weight:700;
                        margin-top:5px;
                    ">{value}</div>
                    <div style="
                        color:#C9D0E3;
                        font-size:12px;
                        margin-top:2px;
                    ">{percentage}</div>
                </div>
                """,
                unsafe_allow_html=True,
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

    for col, (
        label,
        score,
        color,
    ) in zip(score_cols, dimension_data):

        with col:
            st.markdown(
                f"""
                <div style="
                    background:#0D1222;
                    border:1px solid #252D4A;
                    border-radius:14px;
                    padding:18px;
                ">
                    <div style="
                        color:#AAB2C8;
                        font-size:13px;
                        font-weight:500;
                    ">{label}</div>
                    <div style="
                        color:#FFFFFF;
                        font-size:27px;
                        font-weight:700;
                        margin:5px 0 8px 0;
                    ">{score:.2f} / 5</div>
                    <div style="
                        height:7px;
                        background:#242B43;
                        border-radius:10px;
                        overflow:hidden;
                    ">
                        <div style="
                            width:{max(0, min(score / 5 * 100, 100)):.1f}%;
                            height:100%;
                            background:{color};
                            border-radius:10px;
                        "></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
    # DONUT CHART FUNCTION
    # Uses SVG inside a Streamlit component so the chart is
    # rendered by the browser instead of being treated as text.
    # ========================================================

    def render_donut_chart(title, values):

        values = [
            (str(label), int(value))
            for label, value in values
            if int(value) > 0
        ]

        total = sum(
            value for _, value in values
        )

        if total == 0:
            components.html(
                f"""
                <div style=\"
                    width:100%;
                    height:390px;
                    box-sizing:border-box;
                    background:#0D1222;
                    border:1px solid #252D4A;
                    border-radius:14px;
                    padding:20px;
                    font-family:Arial,sans-serif;
                    color:#FFFFFF;
                \" >
                    <div style=\"font-size:18px;font-weight:700;\">{title}</div>
                    <div style=\"height:300px;display:flex;align-items:center;justify-content:center;color:#AAB2C8;\">
                        No evaluation data available
                    </div>
                </div>
                """,
                height=410,
                scrolling=False,
            )
            return

        palette = [
            "#6366F1",
            "#F45B45",
            "#10BFA3",
            "#F59E0B",
            "#8B5CF6",
        ]

        cx = 250
        cy = 165
        radius = 92
        stroke_width = 38
        circumference = 2 * 3.141592653589793 * radius

        svg_segments = []
        legend_items = []
        offset = 0.0

        for index, (label, value) in enumerate(values):

            percentage = (value / total) * 100.0
            color = palette[index % len(palette)]
            dash = percentage / 100.0 * circumference

            svg_segments.append(
                f"""
                <circle
                    cx=\"{cx}\"
                    cy=\"{cy}\"
                    r=\"{radius}\"
                    fill=\"none\"
                    stroke=\"{color}\"
                    stroke-width=\"{stroke_width}\"
                    stroke-dasharray=\"{dash:.3f} {circumference:.3f}\"
                    stroke-dashoffset=\"{-offset:.3f}\"
                    transform=\"rotate(-90 {cx} {cy})\"
                />
                """
            )

            legend_items.append(
                f"""
                <div style=\"
                    display:flex;
                    align-items:center;
                    gap:7px;
                    margin:5px 9px;
                    color:#FFFFFF;
                    font-size:13px;
                    white-space:nowrap;
                \">
                    <span style=\"
                        width:11px;
                        height:11px;
                        min-width:11px;
                        border-radius:2px;
                        background:{color};
                        display:inline-block;
                    \"></span>
                    <span>{label}</span>
                    <span style=\"color:#C9D0E3;\">
                        {percentage:.1f}%
                    </span>
                </div>
                """
            )

            offset += dash

        svg = "".join(svg_segments)
        legend = "".join(legend_items)

        components.html(
            f"""
            <div style=\"
                width:100%;
                height:390px;
                box-sizing:border-box;
                background:#0D1222;
                border:1px solid #252D4A;
                border-radius:14px;
                padding:14px 18px 10px 18px;
                font-family:Arial,sans-serif;
                color:#FFFFFF;
            \">
                <div style=\"
                    color:#FFFFFF;
                    font-size:18px;
                    font-weight:700;
                    margin-bottom:2px;
                \">
                    {title}
                </div>

                <svg
                    width=\"100%\"
                    height=\"285\"
                    viewBox=\"0 0 500 285\"
                    xmlns=\"http://www.w3.org/2000/svg\"
                >
                    <g>
                        <circle
                            cx=\"{cx}\"
                            cy=\"{cy}\"
                            r=\"{radius}\"
                            fill=\"none\"
                            stroke=\"#252D4A\"
                            stroke-width=\"{stroke_width}\"
                        />
                        {svg}
                    </g>

                    <circle
                        cx=\"{cx}\"
                        cy=\"{cy}\"
                        r=\"70\"
                        fill=\"#0D1222\"
                    />

                    <text
                        x=\"{cx}\"
                        y=\"160\"
                        text-anchor=\"middle\"
                        fill=\"#FFFFFF\"
                        font-size=\"27\"
                        font-weight=\"700\"
                    >{total}</text>

                    <text
                        x=\"{cx}\"
                        y=\"182\"
                        text-anchor=\"middle\"
                        fill=\"#AAB2C8\"
                        font-size=\"11\"
                    >Evaluations</text>
                </svg>

                <div style=\"
                    display:flex;
                    flex-wrap:wrap;
                    justify-content:center;
                    align-items:center;
                    margin-top:-2px;
                \">
                    {legend}
                </div>
            </div>
            """,
            height=410,
            scrolling=False,
        )

    # ========================================================
    # DISTRIBUTION DONUTS
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
        render_donut_chart(
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
        render_donut_chart(
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
        render_donut_chart(
            "Score Distribution",
            [
                ("0 – 1.99", score_0_1),
                ("2 – 3.49", score_2_3),
                ("3.5 – 5.0", score_3_5),
            ],
        )

    with col4:
        render_donut_chart(
            "Completeness Status",
            [
                ("Complete", complete_count),
                ("Incomplete", incomplete_count),
            ],
        )

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

        st.line_chart(
            trend_df,
            use_container_width=True,
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
    # LLM PROVIDER
    # ========================================================

    llm_provider = st.selectbox(
        "LLM Provider",
        [
            "Gemini",
            "Groq",
            "Ollama",
        ],
        index=[
            "Gemini",
            "Groq",
            "Ollama",
        ].index(
            st.session_state.single_llm_provider_data
        ),
        key="single_llm_provider",
    )

    st.session_state.single_llm_provider_data = (
        llm_provider
    )

    if llm_provider == "Gemini":

        st.caption(
            "☁️ Using Gemini API"
        )
    elif llm_provider == "Groq":

        st.caption(
            "⚡ Using Groq API"
        )
    else:

        st.caption(
            "🖥️ Using local Ollama model: "
            "qwen3:4b-instruct-2507-q4_K_M"
        )

    set_llm_provider(
        llm_provider.lower()
    )

    # ========================================================
    # PIPELINE
    # ========================================================

    pipeline = [
        "📝 AI Response",
        "🎯 Relevance",
        "✓ Accuracy",
        "◈ Completeness",
        "⚠ Hallucination",
        "⚖ Verdict",
    ]

    cols = st.columns(
        len(pipeline)
    )

    for i, (
        col,
        step,
    ) in enumerate(
        zip(
            cols,
            pipeline,
        )
    ):

        with col:

            st.markdown(
                f"**{step}**"
            )

            if i < len(pipeline) - 1:

                st.caption("→")

    # ========================================================
    # INPUT
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        question = st.text_area(
            "Question",
            placeholder=(
                "Enter the prompt given to the AI..."
            ),
            height=150,
            key="single_question_widget",
        )

    with col2:

        ai_response = st.text_area(
            "AI Generated Response",
            placeholder=(
                "Enter the generated output to evaluate..."
            ),
            height=150,
            key="single_ai_response_widget",
        )

    # Save values immediately
    st.session_state.single_question_data = question
    st.session_state.single_ai_response_data = ai_response

    # ========================================================
    # OPTIONAL REFERENCE
    # ========================================================

    col3, col4 = st.columns(2)

    with col3:

        reference_answer = st.text_area(
            "Reference Answer (Optional)",
            placeholder=(
                "Enter expected ground truth "
                "or golden answer..."
            ),
            height=120,
            key="single_reference_widget",
        )

        st.session_state.single_reference_answer_data = (
            reference_answer
        )

    with col4:

        reference_document = st.file_uploader(
            "Reference Document (Optional)",
            type=[
                "txt",
                "pdf",
                "docx",
            ],
            key="single_document",
        )

    if reference_document is not None:

        st.session_state.single_reference_document_name = (
            reference_document.name
        )

        st.session_state.single_reference_document_bytes = (
            reference_document.getvalue()
        )

    elif st.session_state.single_reference_document_name:

        st.caption(
            "Previously used document: "
            + st.session_state.single_reference_document_name
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
        '<div class="section-title">'
        '📊 Batch Evaluation'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Upload a CSV containing question and "
        "ai_response columns."
    )

    # ========================================================
    # CSV UPLOAD
    # ========================================================

    batch_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_csv",
    )

    batch_df = None

    # ========================================================
    # RESTORE CSV
    # ========================================================

    if batch_file is not None:

        st.session_state.batch_csv_name = (
            batch_file.name
        )

        st.session_state.batch_csv_bytes = (
            batch_file.getvalue()
        )

        batch_df = pd.read_csv(
            batch_file
        )

    elif (
        st.session_state.batch_csv_bytes
        is not None
    ):

        batch_df = pd.read_csv(
            io.BytesIO(
                st.session_state.batch_csv_bytes
            )
        )

    # ========================================================
    # LLM PROVIDER
    # ========================================================

    batch_llm_provider = st.selectbox(
        "LLM Provider",
        [
            "Gemini",
            "Groq",
            "Ollama",
        ],
        index=[
            "Gemini",
            "Groq",
            "Ollama",
        ].index(
            st.session_state.batch_llm_provider_data
        ),
        key="batch_llm_provider",
        help="Choose the LLM provider for batch evaluation.",
    )

    st.session_state.batch_llm_provider_data = (
        batch_llm_provider
    )

    if batch_llm_provider == "Gemini":

        st.caption(
            "☁️ Using Gemini API"
        )
    elif batch_llm_provider == "Groq":

        st.caption(
            "⚡ Using Groq API"
        )
    else:

        st.caption(
            "🖥️ Using local Ollama model: "
            "qwen3:4b-instruct-2507-q4_K_M"
        )

    # ========================================================
    # LOAD CSV
    # ========================================================

    if batch_df is not None:

        try:

            required_columns = [
                "question",
                "ai_response",
            ]

            missing_columns = [
                col
                for col in required_columns
                if col not in batch_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(
                        missing_columns
                    )
                )

                batch_df = None

            else:

                st.success(
                    f"CSV loaded successfully — "
                    f"{len(batch_df)} records found."
                )

                with st.expander(
                    "👀 Preview CSV"
                ):

                    st.dataframe(
                        batch_df.head(10),
                        use_container_width=True,
                    )

        except Exception as e:

            st.error(
                f"Unable to read CSV file: {e}"
            )

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

    # ========================================================
    # SINGLE HISTORY
    # ========================================================

    if history_type == "Single Evaluations":

        history = get_single_history()

        if not history:

            st.info(
                "No completed single evaluations "
                "in History yet."
            )

        else:

            st.write(
                f"**{len(history)} completed evaluation(s)**"
            )

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

                with st.expander(
                    f"🔍 "
                    f"{item['question'][:90]}  |  "
                    f"{verdict_name}  |  "
                    f"{score:.1f}/5  |  "
                    f"{risk} Risk"
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

        if not batches:

            st.info(
                "No completed batch evaluations "
                "in History yet."
            )

        else:

            st.write(
                f"**{len(batches)} batch(es) in History**"
            )

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