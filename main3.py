import pickle


def main():
    with open('data/rec-sys', 'rb') as f:
        rec_sys = pickle.load(f)

    print('===TOP===')
    for i, el in enumerate(rec_sys.recommend(1, 10)):
        print(f'{i + 1})\t {el}')


if __name__ == '__main__':
    main()
