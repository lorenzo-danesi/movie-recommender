import os
import pandas as pd
import zipfile
import urllib.request
from lenskit.datasets import ML100K
from lenskit.algorithms.item_knn import ItemItem
from lenskit import batch

# configuração p/ download do conjunto de dados do movielens-100K
dataset_path = 'data/ml-100k'
url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    file_path = os.path.join(dataset_path, 'ml-100k.zip')
    urllib.request.urlretrieve(url, file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall('data/')

data = ML100K(dataset_path)
algo = ItemItem(nnbrs=30)
algo.fit(data.ratings)

def recommend_for_user(user_id, n=5):
    user_ratings = data.ratings[data.ratings['user'] == user_id]
    all_items = set(data.ratings['item'].unique())
    rated_items = set(user_ratings['item'])
    unrated_items = list(all_items - rated_items)

    candidates = pd.DataFrame({'user': user_id, 'item': unrated_items})
    preds = batch.predict(algo, candidates)

    top_preds = preds.sort_values(by='prediction', ascending=False).head(n)

    movies = pd.read_csv(
        os.path.join(dataset_path, 'u.item'),
        sep='|',
        encoding='latin-1',
        header=None,
        usecols=[0, 1],
        names=['item', 'title']
    )
    movies['title'] = movies['title'].str.upper()
    top_recommendations = top_preds.merge(movies, on='item')

    return top_recommendations[['title', 'prediction']].to_dict(orient='records')
