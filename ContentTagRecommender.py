import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentTagRecommender:
    def __init__(self, 
    movies : pd.DataFrame, tags : pd.DataFrame):
        self.prepare(movies,tags)
        
    def prepare(self, movies : pd.DataFrame, tags : pd.DataFrame):
        self.tags = tags

        self.movie_metadata = pd.merge(movies,tags, how='inner')
        self.movie_metadata = self.movie_metadata.groupby(['title','genres'])['tag'].apply(','.join).reset_index()
        self.movie_metadata['genres'] = self.movie_metadata['genres'].str.replace("|", ",")
        self.movie_mapping = pd.Series(self.movie_metadata.index,index = self.movie_metadata['title']).drop_duplicates()
            
        tfidf = TfidfVectorizer(stop_words='english')

        overview_matrix = tfidf.fit_transform(self.movie_metadata['tag'])
        overview_matrix_genres = tfidf.fit_transform(self.movie_metadata['genres'])
        self.similarity_matrix = linear_kernel(overview_matrix,overview_matrix)
        self.genres_similarity_matrix = linear_kernel(overview_matrix_genres,overview_matrix_genres)

    def recommend(self,targetTitle : str, limit : int):
        similarity_tag_score = self.getTagSimilarityScore(targetTitle)
        similarity_genre_score = self.getGenreSimilarityScore(targetTitle)

        # prevent the given movie from being recommended
        movie_data = self.movie_metadata[self.movie_metadata['title'] != targetTitle]
        movie_data['idx'] = movie_data.index

        sim_score_tag = pd.DataFrame(similarity_tag_score)
        sim_score_tag = sim_score_tag.rename(columns={0:'idx',1:'sim_score_tag'})

        sim_score_genre = pd.DataFrame(similarity_genre_score)
        sim_score_genre = sim_score_genre.rename(columns={0:'idx',1:'sim_score_genre'})

        movie_data = pd.merge(movie_data,sim_score_tag, how='inner', left_on=movie_data.index,right_on=['idx'])
        movie_data = pd.merge(movie_data,sim_score_genre, how='inner', left_on=movie_data.index,right_on=['idx'])
        movie_data = movie_data[['title','tag','genres','sim_score_genre','sim_score_tag']]
        movie_data['tot_score'] = movie_data['sim_score_genre'] + (movie_data['sim_score_tag'] * 2 ) / 3
        return movie_data.sort_values(['tot_score'], ascending=False)[:limit]

    def getTagSimilarityScore(self, targetTitle : str):
        movie_idx = self.movie_mapping[targetTitle]
        return sorted(list(enumerate(self.similarity_matrix[movie_idx])), key=lambda x: x[1], reverse=True)

    def getGenreSimilarityScore(self,targetTitle : str):
        movie_idx = self.movie_mapping[targetTitle]
        return sorted(list(enumerate(self.genres_similarity_matrix[movie_idx])), key=lambda x: x[1], reverse=True)
        