# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print('Dependencies Imported')

# %%
import pandas as pd

data = pd.read_csv(r"Coursera.csv")
print(data.head())


# %%
data.shape


# %%
data.info()

# %%
data.isnull().sum()

# %%
data['Difficulty Level'].value_counts()

# %%
data['Course Rating'].value_counts()

# %%
data['University'].value_counts()

# %%
data['Course Name']

# %%
data = data[['Course Name','Difficulty Level','Course Description','Skills','Course URL','Course Rating']]

# %%
data.head(5)

# %%
'''Data Pre-Processing¶
An important part of the process is to pre-process the data into usable format for the recommendation system'''
# Removing spaces between the words (Lambda funtions can be used as well)

data['Course Name'] = data['Course Name'].str.replace(' ',',')
data['Course Name'] = data['Course Name'].str.replace(',,',',')
data['Course Name'] = data['Course Name'].str.replace(':','')
data['Course Description'] = data['Course Description'].str.replace(' ',',')
data['Course Description'] = data['Course Description'].str.replace(',,',',')
data['Course Description'] = data['Course Description'].str.replace('_','')
data['Course Description'] = data['Course Description'].str.replace(':','')
data['Course Description'] = data['Course Description'].str.replace('(','')
data['Course Description'] = data['Course Description'].str.replace(')','')

#removing paranthesis from skills columns
data['Skills'] = data['Skills'].str.replace('(','')
data['Skills'] = data['Skills'].str.replace(')','')

# %%
data.head(5)

# %%
data['tags'] = data['Course Name'] + data['Difficulty Level'] + data['Course Description'] + data['Skills']

# %%
data.head(5)

# %%
data['tags'].iloc[1]

# %%
new_df = data[['Course Name','tags']]

# %%
new_df.head(5)

# %%
new_df['tags'] = data['tags'].str.replace(',',' ')

# %%
new_df['Course Name'] = data['Course Name'].str.replace(',',' ')

# %%
new_df.rename(columns = {'Course Name':'course_name'}, inplace = True)

# %%
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower()) #lower casing the tags column

# %%
new_df.head(5)

# %%
new_df.shape #3522 courses with tags and 2 columns (course_name and tags)

# %%
# Build a clean text representation for the Sentence Transformer
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Restore spaces in name and description for natural text embedding
clean_title = data['Course Name'].str.replace(',', ' ')
clean_desc = data['Course Description'].str.replace(',', ' ')
clean_skills = data['Skills'].str.replace(',', ' ').fillna('')

combined_text = (
    "Course: " + clean_title + 
    ". Difficulty: " + data['Difficulty Level'] + 
    ". Description: " + clean_desc + 
    ". Skills: " + clean_skills
)

# Convert all text to lowercase for embedding uniformity
combined_text = combined_text.str.lower()

# %%
# Load or compute the course embeddings
embeddings_cache_file = "course_embeddings.npy"
model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists(embeddings_cache_file):
    print("Loading course embeddings from cache...")
    course_embeddings = np.load(embeddings_cache_file)
else:
    print("Computing course embeddings using SentenceTransformer (all-MiniLM-L6-v2)...")
    course_embeddings = model.encode(combined_text.tolist(), show_progress_bar=True)
    print(f"Saving course embeddings to {embeddings_cache_file}...")
    np.save(embeddings_cache_file, course_embeddings)

# %%
def recommend(course):
    # Backward compatibility fallback
    if 'similarity' not in globals():
        global similarity
        similarity = cosine_similarity(course_embeddings)
    course_index = new_df[new_df['course_name'] == course].index[0]
    distances = similarity[course_index]
    course_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:7]

    for i in course_list:
        print(new_df.iloc[i[0]].course_name)

# %%
def recommend_course(course_name):
    # Backward compatibility fallback
    if 'similarity' not in globals():
        global similarity
        similarity = cosine_similarity(course_embeddings)
    course_name = course_name.lower().strip()
    matches = new_df[new_df['course_name'].str.contains(course_name, case=False, na=False)]
    if matches.empty:
        return [{"name": "Course not found. Please check the course name.", "difficulty": "N/A", "rating": "N/A", "url": "#"}]
    course_index = matches.index[0]
    distances = similarity[course_index]
    similar_courses = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    recommendations = []
    for index, score in similar_courses:
        rec = {
            "name": new_df.iloc[index]['course_name'],
            "difficulty": data.iloc[index]['Difficulty Level'],
            "rating": data.iloc[index]['Course Rating'] if 'Course Rating' in data.columns else 'N/A',
            "url": data.iloc[index]['Course URL'] if 'Course URL' in data.columns else '#'
        }
        recommendations.append(rec)
    return recommendations

# %%
def recommend_personalized(interests, career_goal, skill_level, weekly_hours):
    if not interests.strip():
        return []
        
    # 1. Encode user inputs dynamically using Sentence Transformer
    query_emb = model.encode([interests.lower().strip()])[0]
    career_emb = model.encode([career_goal.lower().strip()])[0]
    
    # 2. Compute similarity scores
    S_sim = cosine_similarity([query_emb], course_embeddings)[0]
    S_career = cosine_similarity([career_emb], course_embeddings)[0]
    
    # Clip scores to [0, 1] for normalization
    S_sim = np.clip(S_sim, 0, 1)
    S_career = np.clip(S_career, 0, 1)
    
    # 3. Compute skill alignment score
    skill_mapping = {
        'beginner': 0,
        'conversant': 1,
        'intermediate': 2,
        'advanced': 3
    }
    user_skill_val = skill_mapping.get(skill_level.lower(), 0)
    
    course_difficulty_mapping = {
        'beginner': 0,
        'conversant': 1,
        'intermediate': 2,
        'advanced': 3,
        'not calibrated': 0
    }
    
    S_skill = []
    for diff in data['Difficulty Level']:
        diff_val = course_difficulty_mapping.get(str(diff).lower(), 0)
        dist = abs(user_skill_val - diff_val)
        S_skill.append(1.0 - (dist / 3.0))
    S_skill = np.array(S_skill)
    
    # 4. Compute ratings score
    S_rating = []
    for r in data['Course Rating']:
        try:
            val = float(r) if str(r).lower() != 'not calibrated' else 4.0
        except ValueError:
            val = 4.0
        S_rating.append(val / 5.0)
    S_rating = np.array(S_rating)
    
    # 5. Combined Score
    # Weights: interests (40%), career goal (25%), skill alignment (20%), ratings (15%)
    final_scores = (0.40 * S_sim) + (0.25 * S_career) + (0.20 * S_skill) + (0.15 * S_rating)
    
    # 6. Sort and get top 6 recommendations
    top_indices = np.argsort(final_scores)[::-1][:6]
    
    recommendations = []
    for index in top_indices:
        # Pacing calculation based on weekly hours (average course load = 30 hours)
        avg_hours = 30
        try:
            hours = float(weekly_hours)
            if hours <= 0:
                hours = 5
        except ValueError:
            hours = 5
            
        weeks = int(np.ceil(avg_hours / hours))
        pacing_info = f"{weeks} week{'s' if weeks > 1 else ''} (at {int(hours)} hrs/week)"
        
        rec = {
            "name": data.iloc[index]['Course Name'].replace(',', ' '),
            "difficulty": data.iloc[index]['Difficulty Level'],
            "rating": data.iloc[index]['Course Rating'],
            "url": data.iloc[index]['Course URL'] if 'Course URL' in data.columns else '#',
            "pacing": pacing_info
        }
        recommendations.append(rec)
        
    return recommendations



# %%
