
import sqlite3
import json
from datetime import datetime


DATABASE = "evaluation.db"


def create_database():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    # ============================================================
    # CREATE TABLE
    # ============================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_submissions (
            submission_id INTEGER PRIMARY KEY AUTOINCREMENT,

            evaluation_type TEXT NOT NULL,

            question TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            reference_answer TEXT,
            source_document TEXT,

            relevance_result TEXT,
            accuracy_result TEXT,
            completeness_result TEXT,
            hallucination_result TEXT,
            verdict_result TEXT,

            overall_score REAL,
            risk_level TEXT,

            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,

            created_at TEXT
        )
    """)

    # ============================================================
    # DATABASE MIGRATION
    # ============================================================

    # Check existing columns
    cursor.execute("PRAGMA table_info(evaluation_submissions)")
    columns = [row[1] for row in cursor.fetchall()]

    # Add created_at to older databases
    if "created_at" not in columns:
        cursor.execute("""
            ALTER TABLE evaluation_submissions
            ADD COLUMN created_at TEXT
        """)

    # Copy old timestamp values into created_at
    cursor.execute("""
        UPDATE evaluation_submissions
        SET created_at = timestamp
        WHERE created_at IS NULL
    """)

    conn.commit()
    conn.close()


def save_submission(
    question,
    ai_response,
    reference_answer=None,
    source_document=None,
    evaluation_type="single",
    relevance_result=None,
    accuracy_result=None,
    completeness_result=None,
    hallucination_result=None,
    verdict_result=None,
    overall_score=None,
    risk_level=None,
    status="completed"
):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    # Current timestamp
    timestamp = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO evaluation_submissions
        (
            evaluation_type,
            question,
            ai_response,
            reference_answer,
            source_document,
            relevance_result,
            accuracy_result,
            completeness_result,
            hallucination_result,
            verdict_result,
            overall_score,
            risk_level,
            timestamp,
            status,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        evaluation_type.lower(),
        question,
        ai_response,
        reference_answer,
        source_document,

        json.dumps(relevance_result)
        if relevance_result else None,

        json.dumps(accuracy_result)
        if accuracy_result else None,

        json.dumps(completeness_result)
        if completeness_result else None,

        json.dumps(hallucination_result)
        if hallucination_result else None,

        json.dumps(verdict_result)
        if verdict_result else None,

        overall_score,
        risk_level,

        timestamp,
        status,

        timestamp
    ))

    submission_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return submission_id

