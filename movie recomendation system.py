#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[12]:


# loading the data from the csv file to apandas dataframe
movies1_file = pd.read_csv("C:\\Users\\CHAITHANYA\\Downloads\\movies.csv")


# In[13]:


# printing the first 5 rows of the dataframe
movies1_file.head()


# In[14]:


# number of rows and columns in the data frame

movies1_file.shape


# In[22]:


#selecting the relevant features for recommendation

recommended_features = ['genres','keywords','tagline','cast','director']
print(recommended_features)


# In[23]:


# replacing the null valuess with null string

for feature in recommended_features:
  movies1_file[feature] = movies1_file[feature].fillna('')


# In[26]:


# combining all the 5 selected features

combined_recomended_feature = movies1_file['genres']+' '+movies1_file['keywords']+' '+movies1_file['tagline']+' '+movies1_file['cast']+' '+movies1_file['director']


# In[27]:


print(combined_recomended_feature)


# In[28]:


# converting the text data to feature vectors

vectorizer = TfidfVectorizer()


# In[30]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[31]:


print(feature_vectors)


# In[32]:


# getting the similarity scores using cosine similarity

similarity = cosine_similarity(feature_vectors)


# In[33]:


print(similarity)


# In[52]:


# getting the movie name from the user

movie_name = input(' Enter your favourite movie name : ')


# In[53]:


# creating a list with all the movie names given in the dataset

list_of_all_titles = movies1_file['title'].tolist()
print(list_of_all_titles)


# In[55]:


# finding the close match for the movie name given by the user

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[56]:


close_match = find_close_match[0]
print(close_match)


# In[57]:


# finding the index of the movie with title

index_of_the_movie = movies1_file[movies1_file.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[58]:


# getting a list of similar movies

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[59]:


len(similarity_score)


# In[60]:


# sorting the movies based on their similarity score

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)


# In[61]:


# print the name of similar movies based on the index

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies1_file[movies1_file.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# In[62]:


movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies1_file['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies1_file[movies1_file.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies1_file[movies1_file.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




