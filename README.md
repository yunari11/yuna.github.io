# yuna.github.io

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movies_metadata.csv', low_memory=False)

movies['overview'] = movies['overview'].fillna('')
movies['combined'] = movies['overview'] + ' ' + movies['genres'].astype(str)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def content_based_recommendations(title, top_n=10):
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'vote_average']].iloc[movie_indices]




from flask import Flask, render_template, request
import pandas as pd
from surprise import Dataset, Reader, SVD

app = Flask(__name__)

ratings = pd.read_csv('ratings.csv')   # MovieLens 평점 데이터
movies = pd.read_csv('movies.csv')     # MovieLens 영화 정보

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

def get_movie_recommendations(user_id, top_n=10):
    movie_ids = movies['movieId'].unique()
    user_rated = ratings[ratings['userId'] == int(user_id)]['movieId'].values
    candidates = [mid for mid in movie_ids if mid not in user_rated]

    predictions = [model.predict(int(user_id), mid) for mid in candidates]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_preds = predictions[:top_n]

    results = []
    for pred in top_preds:
        title = movies[movies['movieId'] == pred.iid]['title'].values[0]
        results.append((title, round(pred.est, 2)))
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        user_id = request.form['user_id']
        recommendations = get_movie_recommendations(user_id)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
