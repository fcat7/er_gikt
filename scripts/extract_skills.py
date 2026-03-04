import pandas as pd
import os

def extract_unique_skills(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        # Assistments 2009 usually uses ISO-8859-1 encoding
        df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
        
        if 'skill_name' not in df.columns:
            print(f"Error: 'skill_name' column not found in {file_path}")
            print(f"Available columns: {list(df.columns)}")
            return

        # Drop NaNs and get unique values
        skill_series = df['skill_name'].dropna().astype(str)
        
        unique_skills = set()
        for skills in skill_series:
            # Split by comma and strip whitespace
            split_skills = [s.strip() for s in skills.split(',') if s.strip()]
            unique_skills.update(split_skills)
        
        sorted_skills = sorted(list(unique_skills))
        
        print(f"Total Unique Skills Found: {len(sorted_skills)}")
        print("\n--- Skill List ---")
        for skill in sorted_skills:
            print(skill)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    csv_path = r"H:\dataset\assistments_2009_2010_non_skill_builder_data_new.csv"
    extract_unique_skills(csv_path)
