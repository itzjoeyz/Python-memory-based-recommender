import pandas as pd

from CollabItemRecommender import CollabItemRecommender
from ContentTagRecommender import ContentTagRecommender

class HybridRecommender():

    def __init__(self, movies : pd.DataFrame, ratings : pd.DataFrame ,tags : pd.DataFrame):
        self.prepare(movies,ratings,tags)

    def prepare(self,movies : pd.DataFrame,ratings : pd.DataFrame ,tags : pd.DataFrame):
        self.collabRecommender = CollabItemRecommender(movies,ratings)
        self.contentRecommender = ContentTagRecommender(movies,tags)

    def recommend(self, targetTitle : str, limit : int):        
        try:
            similarity_tag_score = pd.DataFrame(self.contentRecommender.getTagSimilarityScore(targetTitle))
            similarity_tag_score.rename(columns={0:'idx',1:'sim_tag_score'},inplace=True)
            similarity_tag_score.set_index(['idx'])

            similarity_genre_score = pd.DataFrame(self.contentRecommender.getGenreSimilarityScore(targetTitle))
            similarity_genre_score.rename(columns={0:'idx',1:'sim_genre_score'},inplace=True)
            similarity_genre_score.set_index(['idx'])

            movie_meta_data = self.contentRecommender.movie_metadata[self.contentRecommender.movie_metadata['title']!=targetTitle]
            movie_meta_data['idx'] = movie_meta_data.index
            data = pd.merge(movie_meta_data,similarity_tag_score,how='inner')
            data = pd.merge(data,similarity_genre_score)
        except:
            print('no tags available')
            similarity_tag_score = pd.DataFrame()
        
        try:
            collab_scores = pd.DataFrame(self.collabRecommender.recommend(targetTitle,-1))
            collab_scores = collab_scores[collab_scores['title']!= targetTitle]
        except:
            print('no ratings available')
            collab_scores = pd.DataFrame()

        if(similarity_tag_score.empty or collab_scores.empty):
            return sorted(similarity_tag_score[:limit], key=lambda x: x[1], reverse=True) if collab_scores.empty else collab_scores.sort_values(['correlation','mean_rating'], ascending=False)[:limit]

        total_score = pd.merge(collab_scores,data, how='outer')
        total_score['total_score'] = (total_score['sim_tag_score'] + total_score['correlation']  * 2 + total_score['sim_genre_score']) / 3
        total_score = total_score.sort_values(['total_score'], ascending=False)
        return total_score[:limit][['title','sim_tag_score','sim_genre_score','correlation','total_score']]
        