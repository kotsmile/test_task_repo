import numpy as np
from scipy.sparse.linalg import svds
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
import time


class recommend_system:

    def __init__(self, k):
        self.k = k

        self.users_embeddings = None
        self.items_embeddings = None
        self.ratings_dict = None

        self.ranking_model = None

        self.user_profile = None
        self.item_profile = None

        self.x_train = None
        self.y_train = None

    def fit(self, users, items, ratings):

        users, items, ratings = np.array(users), np.array(items), np.array(ratings)

        users_sorted = np.sort(np.unique(users))
        items_sorted = np.sort(np.unique(items))

        self.ratings_dict = {(u, i): r for u, i, r in zip(users, items, ratings)}

        # creating rating matrix
        rate_table = np.zeros((len(users_sorted), len(items_sorted)))
        for i, user in enumerate(users_sorted):
            for j, item in enumerate(items_sorted):
                try:
                    rate_table[i, j] = self.ratings_dict[(user, item)]
                except KeyError:
                    rate_table[i, j] = 0.0

        # SVD
        u, s, v = svds(rate_table, k=self.k)
        us = u @ np.diag(s)
        self.users_embeddings = us / np.max(us)
        self.items_embeddings = v.T / np.max(v)

        # creating dict with profiles
        self.user_profile = dict(zip(users_sorted, self.users_embeddings))
        self.item_profile = dict(zip(items_sorted, self.items_embeddings))

        # self.ranking_model = RandomForestRegressor(max_depth=2, n_estimators=100)
        # self.ranking_model = GradientBoostingRegressor(n_estimators=25)
        self.ranking_model = LinearRegression()

        # creating training data
        x_train = np.array([
            np.concatenate([self.user_profile[user], self.item_profile[item]]) for user, item in zip(users, items)
        ])
        y_train = ratings.T

        t1 = time.time()
        self.ranking_model.fit(x_train, y_train)
        print(f'learn time = {time.time() - t1:.2f} secs')

    def predict(self, users, items):
        result = []
        users = np.array(users)
        items = np.array(items)

        for user, item in zip(users, items):
            try:
                inp = np.concatenate([self.user_profile[user], self.item_profile[item]])
                result.append(self.ranking_model.predict([inp])[0])
            except KeyError:
                result.append(self.ranking_model.predict([[.0] * self.k * 2])[0])

        return result

    def recommend(self, user_id, N):
        new = []

        for item, _ in self.item_profile.items():
            if not (user_id, item) in self.ratings_dict.keys():
                new.append(item)

        recommends = dict(zip(new, self.predict(np.zeros_like(new) + user_id, np.array(new))))
        return sorted(recommends.keys(), key=lambda k: recommends[k], reverse=True)[:N]
