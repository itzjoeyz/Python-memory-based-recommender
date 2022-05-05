from ast import Str
from collections import UserDict
from tkinter.tix import Tree
from turtle import right
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from CollabItemRecommender import CollabItemRecommender

def gatherCollabData(doPrint : bool):
    movies_datalink = pd.merge(pd.read_csv('./data/movies.csv'),pd.read_csv('./data/links.csv'), how='inner')
    ratings = pd.read_csv('./data/ratings.csv')
    movies_ratings = pd.merge(movies_datalink,ratings, how='inner')

    df_user_rating = movies_ratings.pivot_table(index='userId', columns=['title'], values='rating')

    if(doPrint):
        print(df_user_rating.head(10))

    return df_user_rating, movies_ratings

def gatherContentData(doPrint : bool):
    movies_datalink = pd.merge(pd.read_csv('./data/movies.csv'),pd.read_csv('./data/links.csv'), how='inner' )
    tags = pd.read_csv('./data/tags.csv')

    movies = pd.merge(movies_datalink,tags, how='inner')
    # movies['tag'] = movies['tag'].fillna('')

    # movies_tags = pd.crosstab(movies.index,columns=movies['tag']).astype(int)
    # movies_tags = movies[['movieId','title','tag']].drop_duplicates()

    movies_tags = movies.groupby(['title'])['tag'].apply(','.join).reset_index()
    # movies_tags = pd.merge(movies_tags, movies_datalink, how='inner').drop_duplicates()

    mapping = pd.Series(movies_tags.index,index = movies_tags['title']).drop_duplicates()

    if(doPrint):
        print(movies_tags)

    return movies_tags, mapping

def predictCollab(data_user_ratings : pd.DataFrame,data_ratings : pd.DataFrame, targetTitle : Str,
    recommendationLimit : int):
    rated_movie = data_user_ratings[targetTitle]

    similar_movies = data_user_ratings.corrwith(rated_movie)
    similar_movies.dropna(inplace=True)
    similar_movies = pd.DataFrame(similar_movies, columns=['correlation'])
    similar_movies = similar_movies.sort_values(by='correlation', ascending=False)
    
    # Calculate the total number of ratings and the mean rating for each movie
    data_ratings['total_ratings'] = data_ratings.groupby('movieId')['rating'].transform('count')
    data_ratings['mean_rating'] = data_ratings.groupby('movieId')['rating'].transform('mean')

    # drop duplicate rows and create new dataframe that includes core information
    df_movie_statistics = data_ratings[['movieId', 'title', 'total_ratings', 'mean_rating']]
    df_movie_statistics.drop_duplicates('movieId', keep='first', inplace=True)

    # create new dataframe which only contains those with 60 or more ratings.
    df_popular_movies = df_movie_statistics['total_ratings'] >= 60
    df_popular_movies = df_movie_statistics[df_popular_movies].sort_values(['total_ratings', 
                                                        'mean_rating'], ascending=False)

    # drop the nan values on all of the films that do not have ratings
    # sort in order of correlation
    similar_movies = similar_movies.reset_index()
    popular_similar_movies = similar_movies.merge(df_popular_movies, on='title', how='left')
    popular_similar_movies = popular_similar_movies.dropna()
    popular_similar_movies = popular_similar_movies.sort_values(['correlation','mean_rating'], ascending=False)

    return popular_similar_movies[:recommendationLimit]

def predictContent(data : pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words='english')

    overview_matrix = tfidf.fit_transform(data['tag'])
    similarity_matrix = linear_kernel(overview_matrix,overview_matrix)

    return similarity_matrix

def recommendMovies(similarity_matrix : pd.DataFrame, tagData : pd.DataFrame,mapping: pd.Series ,movieTitle : Str,
recommendationLimit : int):
    movie_idx = mapping[movieTitle]

    similarity_score = list(enumerate(similarity_matrix[movie_idx]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[:recommendationLimit]
    movie_indices = [i[0] for i in similarity_score]

    return (tagData[['title','tag']].iloc[movie_indices])

def gatherCollabUserData(doPrint: bool):
    movies_datalink = pd.merge(pd.read_csv('./data/movies.csv'),pd.read_csv('./data/links.csv'), how='inner')
    ratings = pd.read_csv('./data/ratings.csv')
    movies_ratings = pd.merge(movies_datalink,ratings, how='inner')

    df_user_rating = movies_ratings.pivot_table(index='title', columns=['userId'], values='rating')


    if(doPrint):
        print(df_user_rating.head(10))
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(movies_ratings[movies_ratings['userId']==1]['title'])

    return df_user_rating, movies_ratings

def predictCollabUser(data_user_ratings : pd.DataFrame,data_ratings : pd.DataFrame, targetUserId : int,
    recommendationLimit : int):
    currentUser = data_user_ratings[targetUserId]

    user_similarity = data_user_ratings.corrwith(currentUser)

    user_similarity.dropna(inplace=True)
    user_similarity = pd.DataFrame(user_similarity, columns=['correlation'])
    user_similarity = user_similarity.sort_values(by='correlation', ascending=False)

    user_similarity = user_similarity.loc[user_similarity.index != targetUserId][:15]

    overview_data = data_ratings[['movieId','userId','rating']]
    overview_data = overview_data[overview_data['userId'].isin(user_similarity.index)]

    overview_data['would_recommend'] = overview_data[['rating']]>3
    overview_data['number_of_ratings'] = overview_data.groupby('movieId')['would_recommend'].transform('nunique')
    overview_data['mean_rating'] = overview_data.groupby('movieId')['rating'].transform('mean')
    overview_data = overview_data.groupby(['movieId']).max()

    watchedMoviesList = data_ratings[data_ratings['userId'] == targetUserId]['movieId']
    overview_data = overview_data[~overview_data.index.isin(watchedMoviesList)]

    overview_data = overview_data.sort_values(['number_of_ratings','mean_rating'], ascending=False) 
    returnVal = data_ratings[data_ratings['movieId'].isin(overview_data[:recommendationLimit].index)]
    returnVal = returnVal.groupby(['movieId']).max()
    returnVal['mean_rating'] = overview_data[:recommendationLimit]['mean_rating']
    returnVal['number_of_ratings'] = overview_data[:recommendationLimit]['number_of_ratings']
    returnVal = returnVal.sort_values(['number_of_ratings','mean_rating'], ascending=False)

    return returnVal[['title','genres','number_of_ratings','mean_rating']]

def simtest():
    frame = pd.DataFrame(columns=['userId','movieId','rating'])
    counter = 0
    for i in range(5):

        for j in range(7):
            counter += 1
            movieId = j
            if(np.random.randint(0,2) < 1):
                frame.loc[counter] = [i,j,np.random.randint(1,5)]

    print(frame)
    frame = frame.pivot_table(index='movieId', columns=['userId'], values='rating')
    print(frame)

    print(frame[1])
    frame_sim = frame.corrwith(frame[1])
    frame_sim = pd.DataFrame(frame_sim, columns=['correlation'])
    frame_sim = frame_sim.sort_values(by='correlation', ascending=False)
    print(frame_sim)

if __name__ == '__main__':
    title = 'Star Wars: Episode IV - A New Hope (1977)'
    simtest()
    # userRating, rating = gatherCollabData(False);
    # print(predictCollab(userRating, rating,title,5))
    # https://practicaldatascience.co.uk/data-science/how-to-create-a-collaborative-filtering-recommender-system
    
    # tagData, mapping = gatherContentData(False)
    # similarity_matrix = predictContent(tagData)
    # print(recommendMovies(similarity_matrix,tagData,mapping,title,5))
    
    # userRating, rating = gatherCollabUserData(False)
    # print(predictCollabUser(userRating, rating, 1, 15))

    # print()
    # print("You searched by:")
    # print(tagData[['title','tag']].iloc[mapping[title]])