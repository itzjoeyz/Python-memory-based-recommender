import pandas as pd

class CollabItemRecommender:
    def __init__(self, 
    movies : pd.DataFrame, ratings : pd.DataFrame):
        self.prepare(movies,ratings)

    def prepare(self, movies : pd.DataFrame, ratings : pd.DataFrame):
        self.movies_ratings = pd.merge(movies,ratings, how='inner')
        self.user_ratings = self.movies_ratings.pivot_table(index='userId', columns=['title'], values='rating')
    
    def recommend(self,targetTitle : str, limit : int):
        rated_movie = self.user_ratings[targetTitle]
        similar_movies = self.user_ratings.corrwith(rated_movie)
        
        similar_movies.dropna(inplace=True)
        similar_movies = pd.DataFrame(similar_movies, columns=['correlation'])
        similar_movies = similar_movies.sort_values(by='correlation', ascending=False)
        
        # Calculate the total number of ratings and the mean rating for each movie
        current_movie_ratings = self.movies_ratings
        current_movie_ratings['total_ratings'] = current_movie_ratings.groupby('movieId')['rating'].transform('count')
        current_movie_ratings['mean_rating'] = current_movie_ratings.groupby('movieId')['rating'].transform('mean')

        # drop duplicate rows and create new dataframe that includes core information
        movie_statistics = current_movie_ratings[['movieId', 'title', 'total_ratings', 'mean_rating']]
        movie_statistics.drop_duplicates('movieId', keep='first', inplace=True)

        # create new dataframe which only contains those with 60 or more ratings.
        df_popular_movies = movie_statistics['total_ratings'] >= 60
        df_popular_movies = movie_statistics[df_popular_movies].sort_values(['total_ratings', 
                                                            'mean_rating'], ascending=False)

        # drop the nan values on all of the films that do not have ratings
        # sort in order of correlation
        similar_movies = similar_movies.reset_index()
        popular_similar_movies = similar_movies.merge(df_popular_movies, on='title', how='left')
        popular_similar_movies = popular_similar_movies.dropna()
        popular_similar_movies = popular_similar_movies.sort_values(['correlation','total_ratings'], ascending=False)
        
        # remove the movie that is given
        self.popular_similar_movies = popular_similar_movies[popular_similar_movies['title']!= targetTitle]

        if(limit > 0):
            return self.popular_similar_movies[:limit]
        return self.popular_similar_movies