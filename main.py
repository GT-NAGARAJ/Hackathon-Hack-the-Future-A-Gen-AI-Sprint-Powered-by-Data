from setup import setup_database, read_jds, read_cv
from agents import crew
import os

# Setup database and load data
setup_database()
jds = read_jds('A:/GT colab/Project Hackathon/ShortlistCVs/job_descriptions.csv')
cv_folder = 'A:/GT colab/Project Hackathon/ShortlistCVs/CVs'
cv_texts = {cv_file: read_cv(os.path.join(cv_folder, cv_file)) 
            for cv_file in os.listdir(cv_folder) if cv_file.endswith('.pdf')}

# Run the crew with input data
crew.kickoff(inputs={'jds': jds, 'cv_texts': cv_texts})