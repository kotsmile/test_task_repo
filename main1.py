import pickle
import pandas as pd
from rs import recommend_system


def main():
    data = pd.read_csv('data/ratings_train.csv')

    users = data.userId.tolist()
    items = data.movieId.tolist()

    ratings = data.rating.tolist()

    rec_sys = recommend_system(100)
    rec_sys.fit(users, items, ratings)

    with open('data/rec-sys', 'wb') as f:
        pickle.dump(rec_sys, f)


if __name__ == '__main__':
    main()
