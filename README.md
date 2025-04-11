# Hackathon-Hack-the-Future-A-Gen-AI-Sprint-Powered-by-Data
Hackathon : Hack the Future: A Gen AI Sprint Powered by Data

# Intelligent Recruitment Automation System (IRAS)

## Overview

The **Intelligent Recruitment Automation System (IRAS)** is an AI-powered tool designed to revolutionize recruitment by automating the entire pipeline—from job description (JD) analysis to candidate notification. Using advanced natural language processing (NLP), vector embeddings, and similarity search, IRAS processes unstructured JD and CV data, matches candidates to jobs with high accuracy, and automates shortlisting and outreach. This system reduces manual effort, minimizes errors, and accelerates talent acquisition for organizations.

### Key Features
- **JD Summarization**: Extracts key requirements from job descriptions.
- **CV Parsing**: Structures candidate data (e.g., skills, experience) from PDFs.
- **Candidate Matching**: Ranks candidates using semantic similarity.
- **Automated Shortlisting**: Filters top candidates based on a score threshold.
- **Email Notifications**: Sends interview invites to shortlisted candidates.

---

## Problem Statement

Manual recruitment processes are slow, costly, and error-prone due to the labor-intensive review of JDs and CVs. This results in delayed hiring, higher operational expenses, and frequent mismatches. IRAS solves this by automating data extraction, matching, and notifications, enabling faster and more accurate recruitment.

---

## Solution Architecture

IRAS employs a multi-agent framework where specialized agents handle distinct tasks, ensuring modularity and scalability.

### Workflow
1. **Data Ingestion**:  
   - Reads JDs from a CSV file and CVs from PDFs.
2. **NLP Processing**:  
   - Summarizes JDs and parses CVs using a local LLM (Ollama).
3. **Data Storage**:  
   - Stores structured data in SQLite.
4. **Embedding Generation**:  
   - Converts JDs and CVs into vector embeddings.
5. **Vector Indexing**:  
   - Indexes embeddings in Faiss for fast similarity searches.
6. **Matching & Shortlisting**:  
   - Matches candidates to jobs and filters based on scores.
7. **Notification**:  
   - Sends automated emails to shortlisted candidates.

### Technology Stack
- **CrewAI**: Orchestrates multi-agent workflow.
- **Ollama**: LLM for NLP and embeddings.
- **Faiss**: Vector database for similarity search.
- **SQLite**: Relational database for data storage.
- **pdfplumber**: PDF text extraction.
- **smtplib**: Email automation.
- **Python**: Core language with libraries (`csv`, `numpy`).

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- Ollama (with `llama2` model)
- Git

### Installation
1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/your-username/iras.git
   cd iras


   ```
   Replace `your-username` with your GitHub username.

2. **Install Dependencies**:  
   ```bash
   pip install crewai ollama pdfplumber faiss-cpu numpy
   ```

3. **Set Up Ollama**:  
   - Install Ollama: [ollama.ai](https://ollama.ai/)
   - Pull the `llama2` model:  
     ```bash
     ollama pull llama2
     ```
   - Start the server:  
     ```bash
     ollama serve
     ```

4. **Prepare Input Data**:  
   - Add `job_descriptions.csv` with columns "Job Title" and "Job Description".
   - Create a `CVs` folder and place candidate CV PDFs inside.

5. **Configure Email**:  
   - Edit `agents.py` with your SMTP details:  
     ```python
     sender_email = "your_email@gmail.com"
     sender_password = "your_app_specific_password"
     ```

---

## Usage

1. **Run the System**:  
   ```bash
   python main.py
   ```

2. **What Happens**:  
   - IRAS processes JDs and CVs, stores data, generates embeddings, matches candidates, shortlists them, and sends emails.

3. **Outputs**:  
   - `recruitment.db`: SQLite database with JD and CV data.
   - `cv_index.faiss` & `cv_ids.pkl`: Faiss index for CV embeddings.
   - Emails sent to shortlisted candidates.

---

## Project Structure

```bash
iras/
├── setup.py       # Database setup and file reading
├── agents.py      # Agent definitions and tasks
├── main.py        # Main script to run IRAS
├── job_descriptions.csv  # Input JD data
├── CVs/           # Folder for CV PDFs
└── README.md      # This file
```

---

## Conclusion

IRAS transforms recruitment by automating tedious tasks with AI-driven precision. It offers:
- Faster hiring through automation.
- Cost savings by reducing manual effort.
- Better candidate-job matches via semantic analysis.

Upload this repository to GitHub, update the clone URL, and submit the link for your hackathon!

---

**Repository Link**: [https://github.com/your-username/iras](https://github.com/your-username/iras)  
Replace with your actual repo URL after uploading.
```

---

### Next Steps
1. **Fill the Form**: Use the **Problem Statement** and **Solution** sections above directly in your form.
2. **Upload to GitHub**:  
   - Create a new repository named `iras` on GitHub.
   - Add your project files (`setup.py`, `agents.py`, `main.py`, `job_descriptions.csv`, `CVs/` folder).
   - Copy the README text into a `README.md` file and upload it.
   - Update the repository URL in the README and your form submission.
3. **Submit**: Provide the GitHub link as required.

You’re all set to showcase IRAS! Let me know if you need help with anything else.
