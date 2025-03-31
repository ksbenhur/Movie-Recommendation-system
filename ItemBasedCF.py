# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 05:12:28 2025

@author: ksben
"""

import math
from collections import defaultdict

def load_train_data(train_file):
    user_ratings = defaultdict(dict)
    item_ratings = defaultdict(list)
    
    with open(train_file, 'r') as f:
        for line in f:
            u, m, r = map(int, line.split())
            user_ratings[u][m] = r
            item_ratings[m].append(r)
    
    user_averages = {u: sum(r.values()) / len(r) for u, r in user_ratings.items()}
    item_averages = {m: sum(r) / len(r) for m, r in item_ratings.items()}
    
    return user_ratings, item_ratings, user_averages, item_averages

def compute_item_similarities(user_ratings, threshold=0.01):
    sum_numerator = defaultdict(lambda: defaultdict(float))
    sum_i_sq = defaultdict(float)
    
    for u, ratings in user_ratings.items():
        avg_u = sum(ratings.values()) / len(ratings)
        items = list(ratings.keys())
        for i in range(len(items)):
            m_i = items[i]
            adj_i = ratings[m_i] - avg_u
            sum_i_sq[m_i] += adj_i ** 2
            for j in range(i + 1, len(items)):
                m_j = items[j]
                adj_j = ratings[m_j] - avg_u
                product = adj_i * adj_j
                sum_numerator[m_i][m_j] += product
                sum_numerator[m_j][m_i] += product  # Symmetric
    
    def get_similarity(m1, m2):
        if m1 == m2:
            return 1.0
        numerator = sum_numerator[m1].get(m2, 0)
        denom = math.sqrt(sum_i_sq[m1] * sum_i_sq[m2])
        similarity = numerator / denom if denom != 0 else 0.0
        return similarity if similarity > threshold else 0.0
    
    return get_similarity

def predict_ratings(test_file, get_similarity, user_ratings, user_averages, item_averages):
    test_users = []
    with open(test_file, 'r') as f:
        for line in f:
            u, m, r = map(int, line.split())
            if not test_users or test_users[-1]['user_id'] != u:
                test_users.append({'user_id': u, 'known': {}, 'to_predict': []})
            if r == 0:
                test_users[-1]['to_predict'].append(m)
            else:
                test_users[-1]['known'][m] = r
    
    predictions = []
    for user in test_users:
        known = user['known']
        user_avg = sum(known.values()) / len(known) if known else 3
        for m in user['to_predict']:
            sum_rsim, sum_abs_sim = 0.0, 0.0
            for j, r in known.items():
                sim = get_similarity(m, j)
                if sim > 0:  
                    sum_rsim += sim * r
                    sum_abs_sim += abs(sim)
            
            if sum_abs_sim > 0:
                pred = sum_rsim / sum_abs_sim
            else:
                pred = item_averages.get(m, user_avg)  
            
            pred = max(1.0, min(5.0, pred))  
            predictions.append((user['user_id'], m, round(pred)))
    
    return predictions

def save_predictions(predictions, output_file):
    with open(output_file, 'w') as fw:
        for user, movie, rating in predictions:
            fw.write(f"{user} {movie} {rating}\n")

def main():
    train_file = 'C:\\Users\\ksben\\Documents\\santa clara uni docs\\Academics\\Assignments\\WSIR\\Project 2 Test files\\train.txt'
    test_file = 'C:\\Users\\ksben\\Documents\\santa clara uni docs\\Academics\\Assignments\\WSIR\\Project 2 Test files\\test20.txt'
    output_file = 'IBCFAdjustedCosine20.txt'
    
    user_ratings, item_ratings, user_averages, item_averages = load_train_data(train_file)
    get_similarity = compute_item_similarities(user_ratings)
    predictions = predict_ratings(test_file, get_similarity, user_ratings, user_averages, item_averages)
    save_predictions(predictions, output_file)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
