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
data = data[['Course Name','Difficulty Level','Course Description','Skills']]

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
from sklearn.feature_extraction.text import CountVectorizer

# %%
cv = CountVectorizer(max_features=5000,stop_words='english')

# %%
vectors = cv.fit_transform(new_df['tags']).toarray()

# %%
import nltk #for stemming process

# %%
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# %%
#defining the stemming function
def stem(text):
    y=[]

    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)

# %%
new_df['tags'] = new_df['tags'].apply(stem) #applying stemming on the tags column

# %%
from sklearn.metrics.pairwise import cosine_similarity

# %%
similarity = cosine_similarity(vectors)

# %%
def recommend(course):
    course_index = new_df[new_df['course_name'] == course].index[0]
    distances = similarity[course_index]
    course_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:7]

    for i in course_list:
        print(new_df.iloc[i[0]].course_name)

# %%
def recommend_course(course_name):
    # Convert input to lowercase for better matching
    course_name = course_name.lower().strip()
    
    # Find courses that match the input name (partial match allowed)
    matches = new_df[new_df['course_name'].str.contains(course_name, case=False, na=False)]
    
    if matches.empty:
        return ["Course not found. Please check the course name."]

    # Pick the first matched course index
    course_index = matches.index[0]
    
    # Get similarity scores
    distances = similarity[course_index]
    similar_courses = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]

    # Prepare results
    recommendations = [
        f"{i+1}. {new_df.iloc[index]['course_name']} - Rating: {data.iloc[index]['Difficulty Level']}"
        for i, (index, score) in enumerate(similar_courses)
    ]

    return recommendations



# %%
