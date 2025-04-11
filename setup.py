import sqlite3
import csv
import os
import pdfplumber

# Setup SQLite database for jobs and candidates
def setup_database():
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    # Table for job descriptions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            summary TEXT
        )
    ''')
    
    # Table for candidate CV data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            skills TEXT,
            experience TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# Read job descriptions from a CSV file
def read_jds(csv_path):
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        return list(reader)

# Read CVs from PDF files in a folder
def read_cv(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

# Initialize database and load data (for testing)
if __name__ == "__main__":
    setup_database()
    jds = read_jds('job_descriptions.csv')
    cv_folder = 'CVs'
    cv_texts = {cv_file: read_cv(os.path.join(cv_folder, cv_file)) 
                for cv_file in os.listdir(cv_folder) if cv_file.endswith('.pdf')}