import os
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob


npy_list = glob.glob("./data_20221208213155/image/*.npy")

for npy in npy_list:
    npy_object = np.load(npy, allow_pickle=True).item()
    pot_all = npy_object["pot_all"]
    route_grid = npy_object["route"]
    obst_grid = npy_object["obst_grid"]

    pot_all  = np.clip(pot_all , -2, 100)

    fig, ax = plt.subplots(figsize=(100, 100))
    ax.invert_yaxis()
    pot_all_normal = cv2.flip(pot_all, 0)
    sns.heatmap(pot_all_normal, square=True, cmap='coolwarm')
    plt.scatter(route_grid[:, 1], route_grid[:, 0], s=100, marker="x", linewidths=5, c="green")
    plt.scatter(obst_grid[:, 1], obst_grid[:, 0], s=100, marker="^", linewidths=5, c="green")
    plt.savefig("{}.jpg".format(npy[:-4]))
    # plt.show()
