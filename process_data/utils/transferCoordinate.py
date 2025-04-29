import pickle

import pandas as pd


def transform_large_point_set(large_points):
    with open('process/process files/rigid_transform.pkl', 'rb') as f:
        R, t = pickle.load(f)

    large_points[:, 0] *= -1
    large_points[:, 2] *= -1
    large_points[:, 2] += 8

    return large_points @ R.T + t

