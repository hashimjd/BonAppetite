#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import json
import sys
from sklearn.metrics.pairwise import pairwise_distances
from numpy import nan
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error

USERS = 500
DISHES = 1000

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

def findCosine(index_a, index_b):
    a = weighted_matrix[index_a]
    b = weighted_matrix[index_b]

    cord_a = list()
    cord_b = list()

    for i in range (a.size):
        if np.isnan(a[i]) == False and np.isnan(b[i])== False:
            cord_a.append(a[i])
            cord_b.append(b[i])

    cosine_similarity = dot(cord_a, cord_b)/(norm(cord_a)*norm(cord_b))
    return cosine_similarity


def find_row_average(row_num):
    new_a = list()
    for i in raw_matrix[row_num]:
        if np.isnan(i) == False:
            new_a.append(i)
    return np.mean(new_a)


def find_predicted_score(row, column, cosine_vector_users):
    row_a = find_row_average(row)
    numerator, denominator = 0,0

    for i in range (users):
        cosine_similarity = cosine_vector_users[i]

        if np.isnan(weighted_matrix[i][column]) == False:
            numerator += cosine_similarity*weighted_matrix[i][column]
            denominator += abs(cosine_similarity)    

    predicted_score = round(row_a + (numerator/denominator))
    return predicted_score


def find_missing_values(user_to_find):
    user_final_ratings = raw_matrix[user_to_find]
    cosine_vector_users = list()

    #Pre-process values for cosine similarity with that user rather than calculating everytime
    for i in range(users):
        cosine_vector_users.append(findCosine(user_to_find, i))


    for i in range(dishes):
        if np.isnan(user_final_ratings[i]):
            user_final_ratings[i] = find_predicted_score(user_to_find, i,cosine_vector_users)

    return user_final_ratings


def reccomendations(n, user_to_find):
    final_ratings = find_missing_values(user_to_find)
    indices = np.argpartition(final_ratings, -n)[-n:]
    return (indices)

