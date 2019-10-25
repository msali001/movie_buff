import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval


df1=pd.read_csv('../nlp_internship/dataset/credits.csv')
df2=pd.read_csv('../nlp_internship/dataset/movies_metadata.csv')
df4=pd.read_csv('../nlp_internship/dataset/keywords.csv')



df2 = df2[df2.id!='1997-08-20']
df2 = df2[df2.id!='2012-09-29']
df2 = df2[df2.id!='2014-01-01']

df2['id'] = df2['id'].astype(int)

df2=df2.merge(df1, on='id')

choice=int(input("Choose which sort of recommendation you need:\n1.Similar movie based on a Particular Movie \n2.Based on Movie Popularity and/or Genre \n"))

if(choice==1):
    tfidf = TfidfVectorizer(stop_words='english')
    df2['overview'] = df2['overview'].fillna('')
    ran = random.randint(25000, 30000)
    df3 = df2.head(ran)

    tfidf_matrix = tfidf.fit_transform(df3['overview'])

    cosine_sim =  linear_kernel(tfidf_matrix, tfidf_matrix, True)

    indices = pd.Series(df3.index, index=df3['title']).drop_duplicates()

    def get_recommendations(title, cosine_sim=cosine_sim):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]

        return df3['title'].iloc[movie_indices]


    df2 = df2.merge(df4, on='id')


    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df2[feature] = df2[feature].apply(literal_eval)

    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    def get_list(x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            if len(names) > 3:
                names = names[:3]
            return names

    
        return []


    df2['director'] = df2['crew'].apply(get_director)

    features = ['cast', 'keywords', 'genres']
    for feature in features:
        df2[feature] = df2[feature].apply(get_list)


    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
        
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''


    features = ['cast', 'keywords', 'director', 'genres']

    for feature in features:
        df2[feature] = df2[feature].apply(clean_data)


    def create_soup(x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
    df2['soup'] = df2.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words='english')

 
    df5 = df2['soup'].head(20000)

    count_matrix = count.fit_transform(df5)

    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    df2 = df2.reset_index()
    indices = pd.Series(df2.index, index=df2['title'])

    mv=input("Enter the Movie: ")
    print(get_recommendations(mv, cosine_sim2))

if(choice==2):

    C= df2['vote_average'].mean()
    m= df2['vote_count'].quantile(0.9)

    q_movies = df2.copy().loc[df2['vote_count'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']

        return (v/(v+m) * R) + (m/(m+v) * C)

    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

    q_movies = q_movies.sort_values('score', ascending=False)


    n=int(input("List of how much Top Movies do you want: "))
    print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(n))
