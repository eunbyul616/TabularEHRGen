import os
import pickle


def save_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data
