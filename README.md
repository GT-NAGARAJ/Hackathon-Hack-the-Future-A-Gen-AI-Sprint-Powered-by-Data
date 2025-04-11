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
