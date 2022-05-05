import warnings
from CollabItemRecommender import CollabItemRecommender
from CollabUserRecommender import CollabUserRecommender
from ContentTagRecommender import ContentTagRecommender
import pandas as pd

from HybridRecommender import HybridRecommender

def printSplit():
    print('')
    print('=================- NEW RECOMMENDATION -==================')
    print('')
    
def readData():
    tags =  pd.read_csv('./data-large/tags.csv')
    movies = pd.read_csv('./data-large/movies.csv')
    ratings = pd.read_csv('./data-large/ratings.csv')
    return tags,movies,ratings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    tags,movies,ratings = readData()

    user_item_recommender = CollabUserRecommender(movies,ratings)
    print(user_item_recommender.recommend(1,5))

    item_item_recommender = CollabItemRecommender(movies,ratings)
    print() 
    print(item_item_recommender.recommend('Toy Story (1995)',5))

    printSplit()
    content_recommender = ContentTagRecommender(movies,tags)
    print(content_recommender.recommend("Toy Story 2 (1999)",5))

    printSplit()
    hybrid_recommender = HybridRecommender(movies,ratings,tags)
    print(hybrid_recommender.recommend("Toy Story 2 (1999)",5))