# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 14:31:50 2025

@author: ksben
"""

from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
import numpy as np
import pandas as pd
import os


train_file = "C:\\Users\\ksben\\Documents\\santa clara uni docs\\Academics\\Assignments\\WSIR\\Project 2 Test files\\train.txt"
test_files = [
    "C:\\Users\\ksben\\Documents\\santa clara uni docs\\Academics\\Assignments\\WSIR\\Project 2 Test files\\test5.txt",
    "C:\\Users\\ksben\\Documents\\santa clara uni docs\\Academics\\Assignments\\WSIR\\Project 2 Test files\\test10.txt",
    "C:\\Users\\ksben\\Documents\\santa clara uni docs\\Academics\\Assignments\\WSIR\\Project 2 Test files\\test20.txt"
]


embedding_size = 50


def build_ncf_model(num_users, num_movies, embedding_size):
    user_input = Input(shape=(1,), name="user_input")
    movie_input = Input(shape=(1,), name="movie_input")

    user_embedding = Embedding(num_users, embedding_size, name="user_embedding")(user_input)
    movie_embedding = Embedding(num_movies, embedding_size, name="movie_embedding")(movie_input)

    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)

    merged = Concatenate()([user_vec, movie_vec])
    dense1 = Dense(64, activation='relu')(merged)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(1, activation='linear')(dense2)

    model = keras.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


num_users = 200  
num_movies = 1000
model = build_ncf_model(num_users, num_movies, embedding_size)


train_data = pd.read_csv(train_file, delim_whitespace=True, header=None, names=['user', 'movie', 'rating'])
train_data['user'] -= 1 
train_data['movie'] -= 1 


users_train = train_data['user'].values
movies_train = train_data['movie'].values
ratings_train = train_data['rating'].values
model.fit([users_train, movies_train], ratings_train, epochs=15, batch_size=32, verbose=1)


def item_based_prediction(movie_id, train_data, global_mean):
    movie_ratings = train_data[train_data['movie'] == movie_id]['rating']
    return np.mean(movie_ratings) if len(movie_ratings) > 0 else global_mean


def predict_and_save(model, test_files, train_data):
    global_mean = np.mean(train_data['rating'])
    
    for test_file in test_files:
        test_data = pd.read_csv(test_file, delim_whitespace=True, header=None, names=['user', 'movie', 'rating'])
        
        test_data['movie'] -= 1  

        predictions = []
        for _, row in test_data.iterrows():
            if row['rating'] == 0:
                user_id = row['user']  
                movie_id = row['movie']  
                
                
                if 1 <= user_id <= 200:
                    
                    user_id_zero = user_id - 1
                    pred = model.predict([np.array([user_id_zero]), np.array([movie_id])], verbose=0)[0][0]
                else:
                    
                    pred = item_based_prediction(movie_id, train_data, global_mean)
                
                
                pred_clamped = max(1.0, min(5.0, pred))
                pred_rounded = int(round(pred_clamped))
                predictions.append(f"{user_id} {movie_id + 1} {pred_rounded}")  

        
        test_filename = os.path.basename(test_file).replace(".txt", "_predictions.txt")
        with open(test_filename, "w") as f:
            f.write("\n".join(predictions))
        print(f"Predictions saved to {test_filename}")


predict_and_save(model, test_files, train_data)