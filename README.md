# AI Response Validation System with Hallucination Detection Assistance

An AI-powered application that evaluates AI-generated responses for relevance, accuracy, hallucinations, and completeness using a multi-agent evaluation architecture.

## 1. Project Overview

Large Language Models (LLMs) can generate responses that appear correct but may contain inaccurate, incomplete, or unsupported information. The AI Response Validation System helps evaluate the quality and reliability of AI-generated responses through multiple specialized evaluation agents.

The system accepts a question, an AI-generated response, and optional reference information. It evaluates the response against configured criteria, calculates an overall score, generates a verdict, and stores the results for further analysis.

The application supports single-response evaluation, batch evaluation using CSV files, evaluation history, and a dashboard for reviewing results.

## 2. Objectives

- Evaluate the quality and reliability of AI-generated responses.
- Detect potentially unsupported or fabricated information.
- Measure response accuracy, relevance, and completeness.
- Use independent evaluation agents to analyze different quality dimensions.
- Generate an overall score and final verdict.
- Support batch processing of multiple evaluation records.
- Store evaluation results for historical analysis.
- Present evaluation statistics through a dashboard.
- Provide a foundation for comparing responses from different AI model providers.

## 3. Key Features

### Single Evaluation
Evaluate an individual AI-generated response by providing a question, response, and optional reference answer or document.

### Batch Evaluation
Upload a CSV file containing multiple test cases and evaluate the records in batches.

### Multi-Agent Evaluation
Use specialized agents to assess different aspects of response quality.

### Hallucination Detection
Identify potentially unsupported claims and assess whether responses are consistent with available evidence.

### Knowledge Base and Retrieval
Process reference datasets, create embeddings, build a FAISS index, and retrieve relevant information to support evidence-based evaluation.

### Weighted Scoring
Combine evaluation criteria using configured weights to calculate an overall score.

### Verdict Generation
Classify evaluation results as Pass, Needs Improvement, or Fail according to configured scoring thresholds and rules.

### Evaluation History
Store and review previous evaluations using the application's database.

### Dashboard and Analytics
Review overall evaluation statistics, score summaries, and verdict distributions.

### Multiple AI Providers
Support configured providers such as Google Gemini and Groq, subject to API configuration and availability.

### Report Generation
Provide downloadable evaluation reports where supported by the configured application.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Data Processing | Pandas |
| Database | SQLite |
| Embeddings | Sentence Transformers or configured embedding models |
| Vector Search | FAISS |
| AI Model Providers | Google Gemini and Groq |
| Dataset Processing | Python and Pandas |
| Knowledge Retrieval | Custom retrieval pipeline |
| Testing | Python test modules |
| Version Control | Git and GitHub |

## 5. System Architecture

The application follows a modular architecture consisting of input handling, validation, knowledge retrieval, specialized evaluation agents, verdict generation, database storage, and visualization.

### Processing Workflow

1. **Input Module:** Accepts the question, AI-generated response, and optional reference information.
2. **Input Validation:** Checks required fields and handles missing or invalid inputs.
3. **Preprocessing:** Prepares reference information for downstream processing.
4. **Knowledge Retrieval:** Retrieves relevant evidence from the configured knowledge base when applicable.
5. **Evaluation Agents:** Analyze response relevance, accuracy, hallucination risk, and completeness.
6. **Verdict Agent:** Combines evaluation results and determines the final verdict according to the configured rules.
7. **Database:** Stores evaluation records and associated results.
8. **Dashboard:** Displays evaluation statistics, historical results, and available reports.

### Evaluation Workflow Diagram

```text
       User Input
           |
           v
  Question + AI Response
  + Optional Reference
           |
           v
    Input Validation
           |
           v
   Knowledge Retrieval
       (FAISS)
           |
           v
  +----------------------+
  |  Evaluation Agents   |
  +----------------------+
    |        |        |
    v        v        v
 Relevance Accuracy Hallucination
    |        |        |
    +--------+--------+
             |
      Completeness
             |
             v
      Verdict Agent
             |
             v
     Overall Score
      and Verdict
             |
             v
     SQLite Database
             |
             v
 Dashboard / History /
    Reports
```

## 6. Evaluation Criteria and Scoring

The application uses four evaluation dimensions with the following configured weights.

| Evaluation Criterion | Weight | Purpose |
|---|---:|---|
| Hallucination Detection | 40% | Assesses unsupported or potentially fabricated claims |
| Accuracy | 30% | Measures factual correctness |
| Relevance | 15% | Measures how well the response addresses the question |
| Completeness | 15% | Assesses whether the response adequately covers the expected answer |

The weights total 100%.

### Overall Score

The system combines normalized criterion scores using the configured weighted scoring method.

For a quality-oriented scoring scale from 0 to 5:

Overall Score = (Hallucination Quality Score × 0.40) + (Accuracy Score × 0.30) + (Relevance Score × 0.15) + (Completeness Score × 0.15)

**Scoring note:** If the hallucination agent produces a risk score where higher values mean greater hallucination risk, the application must convert that risk into a quality score before using this formula. The README should match the actual implementation.

### Verdict Classification

The configured overall-score thresholds are:

- **Pass:** Overall score is greater than or equal to 4.0 out of 5.
- **Needs Improvement:** Overall score is greater than or equal to 2.5 and below 4.0.
- **Fail:** Overall score is below 2.5.

Additional hallucination-risk rules may influence the final verdict.

## 7. Project Directory Structure

```text
AI-Response-Validation-System/
│
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── accuracy_agent.py
│   ├── completeness_agent.py
│   ├── hallucination_agent.py
│   ├── relevance_agent.py
│   └── verdict_agent.py
│
├── architecture/
│
├── docs/
│   └── AI_Response_Validation_System_Final_Report.docx
│
├── kb/
│   ├── data/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── faiss_index.py
│   ├── load_squad.py
│   ├── load_truthfulqa.py
│   ├── preprocess.py
│   ├── retrieve.py
│   ├── truthfulness-related dataset files
│   └── reference index and embedding files
│
├── tests/
│   ├── test_accuracy.py
│   ├── test_gemini.py
│   └── test_verdict.py
│
├── app.py
├── batch_evaluator.py
├── batch_results.json
├── database.py
├── evaluation.db
├── orchestrator.py
├── requirements.txt
├── test_cases.csv
├── README.md
├── LICENSE
├── .gitignore
└── .env
```

This structure summarizes the files visible in the development environment. Generated files, datasets, and environment-specific files may vary. The `requirements.txt`, `README.md`, and Agile documentation should be included in the final repository if they are part of your submission.

## 8. Installation and Setup

### Prerequisites

- Python installed on your computer.
- Git installed for repository management.
- A GitHub account.
- API credentials for a configured AI provider if required.
- Sufficient disk space for the knowledge base, embeddings, and model dependencies.

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-Response-Validation-System
```

Replace the placeholder with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On Linux or macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure the requirements file includes the dependencies used by the current implementation.

### Step 4: Configure API Credentials

Configure the API keys required by your chosen provider.

For example, if your application reads these environment variables:

```text
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

Use only the variables required by your implementation. Store credentials securely in a local environment file or the deployment platform's secrets manager.

**Never upload real API keys or your `.env` file to GitHub.**

### Step 5: Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal. The default Streamlit address is generally:

`http://localhost:8501`

## 9. How to Use the Application

### Single Response Evaluation

1. Launch the application.
2. Select an available AI provider if the interface offers provider selection.
3. Enter the question.
4. Enter the AI-generated response.
5. Provide a reference answer or document if required or available.
6. Run the evaluation.
7. Review the criterion scores, findings, overall score, and verdict.

### Batch Response Evaluation

1. Prepare a CSV file using the column names expected by the application.
2. Include the question and AI-generated response for each test case.
3. Include optional reference-answer or reference-document columns if supported.
4. Upload the CSV through the batch evaluation interface.
5. Select the configured provider if prompted.
6. Run the batch evaluation.
7. Review successful evaluations and failed or invalid records.
8. Inspect the results in the dashboard or history and export available reports.

Refer to the project's sample CSV or test cases for the exact column names.

### Dashboard and History

The dashboard and history features allow users to review stored results, examine verdict distributions, compare evaluation scores, and access supported reporting functionality.

## 10. Knowledge Base and Datasets

The knowledge-base module contains components for:

- Text chunking and preprocessing.
- Generating embeddings.
- Building a FAISS index.
- Retrieving relevant reference information.
- Loading the SQuAD dataset.
- Loading the TruthfulQA dataset.

These components support reference-based evaluation and evidence retrieval where enabled by the current implementation.

The reliability of the evaluation depends on the quality, relevance, and coverage of the reference information and retrieved evidence.

## 11. Testing

The project includes test modules for selected evaluation components.

Example test files:

- `tests/test_accuracy.py`
- `tests/test_gemini.py`
- `tests/test_verdict.py`

Run the available tests from the project root using:

```bash
python -m pytest tests/
```

This command requires `pytest` to be installed. Run the tests with the required environment variables and dependencies configured.

Recommended additional checks include:

- Valid question and response.
- Incorrect or unsupported claims.
- Incomplete answers.
- Irrelevant answers.
- Missing input fields.
- Multiple valid CSV records.
- CSV records containing missing fields.
- Database storage and history retrieval.
- Each configured AI provider.
- Report generation and download.
- API errors and rate-limit handling.

## 12. Deployment

The application can be deployed to a hosting platform that supports Streamlit, such as Streamlit Community Cloud.

General deployment steps:

1. Push the project to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select the repository and deployment branch.
5. Configure `app.py` as the main application file.
6. Add the required API credentials in the platform's secrets settings.
7. Deploy the application.
8. Test the public URL and verify that the required features work.

### Deployment Considerations

- Hosted applications may not preserve local SQLite data across restarts or redeployments.
- Large embedding files and vector indexes may need to be generated during setup or stored using an appropriate persistent-storage solution.
- Model downloads and dependencies may increase deployment time and resource usage.
- External APIs may impose usage limits or rate limits.
- Test the complete application after deployment rather than relying only on a successful startup.

## 13. Security and Limitations

- Keep API keys and other secrets private.
- Automated evaluation cannot guarantee that every response is factually correct.
- Evaluation results can vary between model providers and individual runs.
- Retrieval quality depends on the available reference data and search configuration.
- Batch processing may be affected by API limits, invalid records, or network failures.
- Persistent database storage requires appropriate configuration on hosted platforms.
- Evaluation scores should be interpreted as assessment signals, not definitive proof of correctness.

## 14. Future Enhancements

Potential future improvements include:

- More comprehensive automated testing.
- Improved evidence-based hallucination detection.
- More detailed claim-level explanations.
- Additional AI model providers.
- More reliable large-batch processing.
- Enhanced dashboard analytics.
- Persistent cloud database support.
- Improved evaluation benchmarking and calibration.

## 15. License

This project includes an MIT License. Refer to the `LICENSE` file in the repository for the license text and applicable terms.

## 16. Acknowledgements

Developed as part of the Infosys Springboard Virtual Internship.

The project explores AI response validation, hallucination detection assistance, multi-agent evaluation, knowledge retrieval, batch processing, and evaluation analytics.
