from crewai import Agent, Task, Crew
import sqlite3
import faiss
import numpy as np
import pickle
import smtplib
from email.mime.text import MIMEText
from ollama import Client

# Initialize Ollama client
llm = Client()

# --- Tools ---

def summarize_jd(jd):
    """Summarize a job description using the LLM."""
    prompt = "Summarize this job description: " + jd['Job Description']
    response = llm.generate(model='llama2', prompt=prompt)
    return response['response']  # Adjust based on actual Ollama API response

def parse_cv(cv_text):
    """Extract structured data from CV text using the LLM."""
    prompt = "Extract name, email, skills, experience from this CV:\n" + cv_text
    response = llm.generate(model='llama2', prompt=prompt)
    # Assuming response is a stringified dict; adjust parsing as needed
    return eval(response['response'])

def generate_embedding(text):
    """Generate an embedding for text using the LLM."""
    response = llm.embeddings(model='llama2', prompt=text)
    return response['embedding']  # Adjust based on actual Ollama API

def store_in_db(data, table):
    """Store data in the SQLite database."""
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    if table == 'jobs':
        cursor.execute("INSERT INTO jobs (title, summary) VALUES (?, ?)", 
                       (data['title'], data['summary']))
    elif table == 'candidates':
        cursor.execute("INSERT INTO candidates (name, email, skills, experience) VALUES (?, ?, ?, ?)", 
                       (data['name'], data['email'], data['skills'], data['experience']))
    conn.commit()
    conn.close()

def store_embeddings(cv_embeddings):
    """Store CV embeddings in a Faiss index."""
    dimension = len(next(iter(cv_embeddings.values())))
    index = faiss.IndexFlatL2(dimension)
    cv_ids = list(cv_embeddings.keys())
    cv_vectors = np.array([cv_embeddings[id] for id in cv_ids], dtype=np.float32)
    index.add(cv_vectors)
    faiss.write_index(index, 'cv_index.faiss')
    with open('cv_ids.pkl', 'wb') as f:
        pickle.dump(cv_ids, f)

def match_jds_to_cvs(top_k=10):
    """Match JDs to CVs using Faiss similarity search."""
    index = faiss.read_index('cv_index.faiss')
    with open('cv_ids.pkl', 'rb') as f:
        cv_ids = pickle.load(f)
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, summary FROM jobs")
    jobs = cursor.fetchall()
    matches = {}
    for jd_id, summary in jobs:
        jd_embedding = generate_embedding(summary)
        query = np.array([jd_embedding], dtype=np.float32)
        distances, indices = index.search(query, top_k)
        max_dist = max(distances[0]) if max(distances[0]) > 0 else 1
        matches[jd_id] = [(cv_ids[idx], 1 - (dist / max_dist)) 
                          for idx, dist in zip(indices[0], distances[0])]
    conn.close()
    return matches

def shortlist_candidates(matches, threshold=0.8):
    """Shortlist candidates based on similarity scores."""
    return {jd_id: [cand_id for cand_id, score in matches[jd_id] if score > threshold] 
            for jd_id in matches}

def send_emails(shortlisted):
    """Send interview invitation emails to shortlisted candidates."""
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "your_email@gmail.com"  # Replace with your email
    sender_password = "your_app_specific_password"  # Replace with your app-specific password
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, sender_password)
    for jd_id, cand_ids in shortlisted.items():
        cursor.execute("SELECT title FROM jobs WHERE id = ?", (jd_id,))
        job_title = cursor.fetchone()[0]
        for cand_id in cand_ids:
            cursor.execute("SELECT email, name FROM candidates WHERE id = ?", (cand_id,))
            email, name = cursor.fetchone()
            msg = MIMEText(f"Dear {name},\n\nYou’ve been shortlisted for an interview for the {job_title} position.\n\nRegards,\nRecruitment Team")
            msg['Subject'] = f"Interview Invitation: {job_title}"
            msg['From'] = sender_email
            msg['To'] = email
            server.send_message(msg)
    server.quit()
    conn.close()

# --- Define Agents ---

jd_summarizer = Agent(
    role='JD Summarizer',
    goal='Summarize job descriptions accurately.',
    backstory='Expert in summarizing job descriptions concisely.',
    tools=[summarize_jd],
    llm=llm,
    verbose=True
)

jd_storer = Agent(
    role='JD Storer',
    goal='Store summarized JDs in the database.',
    backstory='Database specialist for job data.',
    tools=[store_in_db],
    llm=llm,
    verbose=True
)

cv_parser = Agent(
    role='CV Parser',
    goal='Parse CV text into name, email, skills, and experience.',
    backstory='Specialist in extracting structured data from CVs.',
    tools=[parse_cv],
    llm=llm,
    verbose=True
)

cv_storer = Agent(
    role='CV Storer',
    goal='Store parsed CV data in the database.',
    backstory='Database expert for candidate data.',
    tools=[store_in_db],
    llm=llm,
    verbose=True
)

embedding_generator = Agent(
    role='Embedding Generator',
    goal='Generate embeddings for JDs and CVs.',
    backstory='Machine learning expert for creating text embeddings.',
    tools=[generate_embedding],
    llm=llm,
    verbose=True
)

vector_storer = Agent(
    role='Vector Storer',
    goal='Store CV embeddings in Faiss.',
    backstory='Vector database specialist.',
    tools=[store_embeddings],
    llm=llm,
    verbose=True
)

matching_agent = Agent(
    role='Matching Agent',
    goal='Match JDs to CVs using vector similarity.',
    backstory='Expert in similarity matching.',
    tools=[match_jds_to_cvs],
    llm=llm,
    verbose=True
)

shortlisting_agent = Agent(
    role='Shortlisting Agent',
    goal='Shortlist top candidates based on scores.',
    backstory='Expert in candidate filtering.',
    tools=[shortlist_candidates],
    llm=llm,
    verbose=True
)

notification_agent = Agent(
    role='Notification Agent',
    goal='Send interview invitations.',
    backstory='Communication specialist.',
    tools=[send_emails],
    llm=llm,
    verbose=True
)

# --- Define Tasks ---

task1 = Task(
    description='Summarize each job description from the provided list.',
    agent=jd_summarizer,
    expected_output='List of summarized JDs'
)

task2 = Task(
    description='Store summarized JDs in the database.',
    agent=jd_storer,
    expected_output='JDs stored in database'
)

task3 = Task(
    description='Parse each CV text into structured data.',
    agent=cv_parser,
    expected_output='List of parsed CV data'
)

task4 = Task(
    description='Store parsed CV data in the database.',
    agent=cv_storer,
    expected_output='CV data stored in database'
)

task5 = Task(
    description='Generate embeddings for JDs and CVs.',
    agent=embedding_generator,
    expected_output='Embeddings for JDs and CVs'
)

task6 = Task(
    description='Store CV embeddings in a Faiss index.',
    agent=vector_storer,
    expected_output='CV embeddings stored in Faiss'
)

task7 = Task(
    description='Match JDs to CVs using similarity search.',
    agent=matching_agent,
    expected_output='Dictionary of JD-to-CV matches'
)

task8 = Task(
    description='Shortlist candidates based on a threshold.',
    agent=shortlisting_agent,
    expected_output='Dictionary of shortlisted candidates'
)

task9 = Task(
    description='Send interview invitation emails.',
    agent=notification_agent,
    expected_output='Emails sent to candidates'
)

# --- Create Crew ---

crew = Crew(
    agents=[jd_summarizer, jd_storer, cv_parser, cv_storer, embedding_generator, 
            vector_storer, matching_agent, shortlisting_agent, notification_agent],
    tasks=[task1, task2, task3, task4, task5, task6, task7, task8, task9],
    verbose=True
)