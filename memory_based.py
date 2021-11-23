#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import statistics 
import json
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import mean_squared_error

CURRENT_USER = 399
USERS = 500
DISHES = 1000
NUM_CLUSTERS = 34

# Load data and convert ratings to json object
ingredient_matrix = pd.read_csv("dishes.csv")
f = open('user_ratings_train.json')
data = json.load(f)

a = data.items()
b = tuple(a)

raw_matrix = np.zeros((USERS, DISHES))
np.ndarray.fill(raw_matrix, np.nan)

for y in range (USERS):
    for i in b[y][1]:
        raw_matrix[y][i[0]] = i[1] 

# Remove extraneous attributes from the dataset and covert it to numpy array
ingredient_matrix = ingredient_matrix.drop(columns=['dish_name', 'dish_id'])
ingredient_array = ingredient_matrix.to_numpy()
weighted_matrix = raw_matrix.copy()

# Find the row-wise mean and store the value in corresponding index
for y in range (USERS): 
    mean = np.nanmean(weighted_matrix[y])
    weighted_matrix[y] -= mean

# Train the model using KMedoids algorithm 
kmedoids = KMedoids(n_clusters=NUM_CLUSTERS, random_state=0, metric='correlation').fit_predict(ingredient_matrix)

def predict_dish(current_user, dish_to_predict):
    described = 0
    cluster_average = [0] * NUM_CLUSTERS

    for y in range (NUM_CLUSTERS):
        indices = [i for i, x in enumerate(kmedoids) if x == y]
        ratings = list()
        current_rating = list(raw_matrix[current_user])

        for index in indices:     
            if np.isnan(current_rating[index]) == False:
                ratings.append(current_rating[index])  
                described += 1

        if (len(ratings) > 0):
            cluster_average[y] = round(np.mean(ratings))

    cluster_grp = kmedoids[dish_to_predict]
    predicted_val = cluster_average[cluster_grp]

    return predicted_val


def reccomendations(n, user_to_find):
    final_ratings = list()

    for i in range (1000):
        if np.isnan(raw_matrix[i]) == True:
            final_ratings += predict_dish(user_to_find, i)

    indices = np.argpartition(final_ratings, -n)[-n:]
    
    return (indices)