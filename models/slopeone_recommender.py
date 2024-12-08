import pandas as pd
from surprise import Dataset, Reader, SlopeOne

# configuração para download de arquivos do conjunto de dados movielens-100K
MOVIES_URL = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
RATINGS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"

movies_columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
                  'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies_df = pd.read_csv(MOVIES_URL, sep='|', names=movies_columns, encoding='latin-1')

ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_df = pd.read_csv(RATINGS_URL, sep='\t', names=ratings_columns, encoding='latin-1')

reader = Reader(rating_scale=(1, 5))
train_data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
trainset = train_data.build_full_trainset()

slopeone = SlopeOne()
slopeone.fit(trainset)

def get_recommendations(user_id, n=5):
    predictions = slopeone.test(trainset.build_testset())
    user_predictions = [pred for pred in predictions if pred.uid == user_id]
    user_predictions.sort(key=lambda x: x.est, reverse=True)

    top_n = user_predictions[:n]
    recommendations = [
        {
            "title": movies_df[movies_df['movie_id'] == int(pred.iid)]['movie_title'].values[0].upper(),
            "rating": pred.est,
        }
        for pred in top_n
    ]
    return recommendations
