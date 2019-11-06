import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as mae


def test(name):
    with open('data/rec-sys', 'rb') as f:
        rec_sys = pickle.load(f)

    data = pd.read_csv(name)

    users = data.userId.tolist()
    items = data.movieId.tolist()
    rating = np.array(data.rating.tolist())

    result = rec_sys.predict(users, items)

    return mae(rating, result)


def main():
    err_test = test('data/ratings_test.csv')
    print(f'mae(test) = {err_test}')

    err_train = test('data/ratings_train.csv')
    print(f'mae(train) = {err_train}')


if __name__ == '__main__':
    main()
