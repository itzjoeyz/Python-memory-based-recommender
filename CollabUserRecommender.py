import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollabUserRecommender:
    def __init__(self, 
    movies : pd.DataFrame, ratings : pd.DataFrame):
        print('welcome to your new collaborative user-item recommender')
        self.prepare(movies,ratings)
    
    def prepare(self, movies : pd.DataFrame, ratings : pd.DataFrame):
        self.movies_ratings = pd.merge(movies,ratings, how='inner')
        self.user_ratings = self.movies_ratings.pivot_table(index='title', columns=['userId'], values='rating')
        
    def recommend(self,targetUserId : int, limit : int, userLimit = 15):
        currentUser = self.user_ratings[targetUserId]

        currentUser = pd.DataFrame(currentUser.values,columns=[1],index=currentUser.index)
        currentUser.dropna(inplace=True)

        count_ratings = self.user_ratings
        count_ratings = count_ratings[count_ratings.index.isin(currentUser.index)]

        list_of_ratings = []
        for user in count_ratings.columns:
            temp_collection = count_ratings[user]
            temp_collection.dropna(inplace=True)
            list_of_ratings.append({'userId':user,"count":temp_collection.count()})

        list_of_ratings = sorted(list_of_ratings, key=lambda x: x['count'],reverse=True)

        list_of_ratings = list_of_ratings[1:]
        list_of_ratings = list_of_ratings[:userLimit]
        user_similarity = [d['userId'] for d in list_of_ratings]

        # get all films that are rated by users within the userLimit sorted on correlation
        overview_data = self.movies_ratings[['movieId','userId','rating']]

        overview_data = overview_data[overview_data['userId'].isin(user_similarity)]

        overview_data['would_recommend'] = overview_data[['rating']]>3
        
        # number of reviews that are above 3 (would recommend = True) by the users that are similar to the chosen user. 
        overview_data['number_of_ratings'] = overview_data[overview_data['would_recommend'] == True].groupby('movieId')['would_recommend'].transform('count')
        # mean rating of all movies including the not recommended ones
        overview_data['mean_rating'] = overview_data.groupby('movieId')['rating'].transform('mean')
        overview_data = overview_data.groupby(['movieId']).max()

        watchedMoviesList = self.movies_ratings[self.movies_ratings['userId'] == targetUserId]['movieId']
        overview_data = overview_data[~overview_data.index.isin(watchedMoviesList)]

        overview_data = overview_data.sort_values(['number_of_ratings','mean_rating'], ascending=False) 
        returnVal = self.movies_ratings[self.movies_ratings['movieId'].isin(overview_data[:limit].index)]
        returnVal = returnVal.groupby(['movieId']).max()
        returnVal['mean_rating'] = overview_data[:limit]['mean_rating']
        returnVal['number_of_ratings'] = overview_data[:limit]['number_of_ratings']
        returnVal = returnVal.sort_values(['number_of_ratings','mean_rating'], ascending=False)

        return returnVal[['title','genres','number_of_ratings','mean_rating']]

        