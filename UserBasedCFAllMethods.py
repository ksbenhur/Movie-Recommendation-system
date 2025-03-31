import pandas as pd
import math

# Helper functions
average = lambda x: float(sum(x)) / len(x)


def pearson_coefficient_with_iuf_and_amplification(v1, v2, iuf_weight, amplification_factor=2.5):
    assert len(v1) == len(v2) and len(v1) > 0
    sumxx, sumyy, sumxy = 0, 0, 0
    x_avg, y_avg = average(v1), average(v2)
    for x, y in zip(v1, v2):
        x = x - x_avg
        y = y - y_avg
        sumxy += x * y * iuf_weight  
        sumxx += x * x * iuf_weight
        sumyy += y * y * iuf_weight
    if math.sqrt(sumxx * sumyy) == 0:
        return 0
    else:
        similarity = sumxy / math.sqrt(sumxx * sumyy)
        return abs(similarity) ** amplification_factor  

def pearson_coefficient_with_amplification(v1, v2, amplification_factor=2.5):
    similarity = pearson_coefficient(v1, v2)
    similarity = (similarity + 1) / 2  # Normalize to [0, 1]
    return abs(similarity) ** amplification_factor

def calculate_iuf_weights(train_matrix):
    num_items = train_matrix.shape[1]  
    iuf_weights = {}
    for user in train_matrix.index:
        num_rated = (train_matrix.loc[user] > 0).sum()  
        if num_rated == 0:
            iuf_weights[user] = 0
        else:
            iuf_weights[user] = math.log(num_items / num_rated)
    return iuf_weights


def pearson_coefficient_with_iuf(v1, v2, iuf_weight):
    assert len(v1) == len(v2) and len(v1) > 0
    sumxx, sumyy, sumxy = 0, 0, 0
    x_avg, y_avg = average(v1), average(v2)
    for x, y in zip(v1, v2):
        x = x - x_avg
        y = y - y_avg
        sumxy += x * y * iuf_weight  
        sumxx += x * x * iuf_weight
        sumyy += y * y * iuf_weight
    if math.sqrt(sumxx * sumyy) == 0:
        return 0
    else:
        return sumxy / math.sqrt(sumxx * sumyy)
    
def pearson_coefficient(v1, v2):
    assert len(v1) == len(v2) and len(v1) > 0
    sumxx, sumyy, sumxy = 0, 0, 0
    x_avg, y_avg = average(v1), average(v2)
    for x, y in zip(v1, v2):
        x = x - x_avg
        y = y - y_avg
        sumxy += x * y
        sumxx += x * x
        sumyy += y * y
    if math.sqrt(sumxx * sumyy) == 0:
        return 0
    else:
        return sumxy / math.sqrt(sumxx * sumyy)


def cosine_similarity(v1, v2):
    assert len(v1) == len(v2) and len(v1) > 0
    sumxx, sumxy, sumyy = 0, 0, 0
    for x, y in zip(v1, v2):
        sumxx += x * x
        sumxy += x * y
        sumyy += y * y
    return sumxy / math.sqrt(sumxx * sumyy)


def predict_ratings(train_matrix, test_data, sim='cosine', amplification_factor=2.5):
    predictions = []
    global_average = train_matrix.mean().mean()  
    if sim in ['pearson_iuf', 'pearson_iuf_amplified']:
        iuf_weights = calculate_iuf_weights(train_matrix)

    for user, ratings in test_data.items():
        v1 = [rating for movie, rating in ratings if rating != 0][:10]
        v1_avg = average(v1) if len(v1) > 0 else global_average

        for movie, rating in ratings:
            if rating == 0:
                
                if movie not in train_matrix.columns:
                    
                    predicted_rating = global_average
                else:
                    numerator, denominator = 0, 0
                    similar_users = []

                    for train_user in train_matrix.index:
                        if train_matrix.loc[train_user, movie] == 0:
                            continue  
                        tmpv = train_matrix.loc[train_user, train_matrix.loc[train_user] > 0].values[:len(v1)]

                        
                        if sim == 'cosine':
                            similarity = cosine_similarity(v1, tmpv)
                        elif sim == 'pearson':
                            similarity = pearson_coefficient(v1, tmpv)
                        elif sim == 'pearson_iuf':
                            similarity = pearson_coefficient_with_iuf(v1, tmpv, iuf_weights[train_user])
                        elif sim == 'pearson_amplified':
                            similarity = pearson_coefficient_with_amplification(v1, tmpv, amplification_factor)
                        elif sim == 'pearson_iuf_amplified':
                            similarity = pearson_coefficient_with_iuf_and_amplification(v1, tmpv, iuf_weights[train_user], amplification_factor)
                        else:
                            raise ValueError(f"Unknown similarity metric: {sim}")

                        similar_users.append((train_user, similarity))

                    
                    similar_users.sort(key=lambda x: x[1], reverse=True)

                    
                    for u, r in similar_users[:20]:
                        v2 = train_matrix.loc[u, train_matrix.loc[u] > 0].values[:len(v1)]
                        user_avg = average(v2)

                        
                        if sim == 'cosine':
                            similarity = cosine_similarity(v1, v2)
                        elif sim == 'pearson':
                            similarity = pearson_coefficient(v1, v2)
                        elif sim == 'pearson_iuf':
                            similarity = pearson_coefficient_with_iuf(v1, v2, iuf_weights[u])
                        elif sim == 'pearson_amplified':
                            similarity = pearson_coefficient_with_amplification(v1, v2, amplification_factor)
                        elif sim == 'pearson_iuf_amplified':
                            similarity = pearson_coefficient_with_iuf_and_amplification(v1, v2, iuf_weights[u], amplification_factor)

                        numerator += similarity * (train_matrix.loc[u, movie] - user_avg)
                        denominator += abs(similarity)

                    if denominator == 0:
                        predicted_rating = v1_avg
                    else:
                        predicted_rating = v1_avg + (numerator / denominator)

                
                predicted_rating = max(1, min(predicted_rating, 5))
                predictions.append((user, movie, int(round(predicted_rating))))
    return predictions


def load_train_data(filepath):
    train_data = pd.read_csv(filepath, sep=' ', header=None, names=['user_id', 'movie_id', 'rating'])
    train_matrix = train_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    return train_matrix

def load_test_data(filepath):
    test_data = pd.read_csv(filepath, sep=' ', header=None, names=['user_id', 'movie_id', 'rating'])
    test_dict = {}
    for user, group in test_data.groupby('user_id'):
        test_dict[user] = list(zip(group['movie_id'], group['rating']))
    return test_dict

def save_predictions(predictions, filepath):
    with open(filepath, 'w') as fw:
        for user, movie, rating in predictions:
            fw.write(f"{user} {movie} {rating}\n")


def main(train_file, test_file, output_file, sim='cosine', amplification_factor=1.5):
   
    train_matrix = load_train_data(train_file)
    test_data = load_test_data(test_file)

    
    predictions = predict_ratings(train_matrix, test_data, sim, amplification_factor)

    
    save_predictions(predictions, output_file)
    print(f"Predictions saved to {output_file}")


if __name__ == '__main__':
    train_file = 'C:\\Users\\ksben\\Documents\\santa clara uni docs\\Academics\\Assignments\\WSIR\\Project 2 Test files\\train.txt'
    test_file = 'C:\\Users\\ksben\\Documents\\santa clara uni docs\\Academics\\Assignments\\WSIR\\Project 2 Test files\\test20.txt'
    output_file = 'UBCFpearsoniufamplified20.txt'
    similarity_metric = 'pearson_iuf_amplified' # Options: 'cosine', 'pearson', 'pearson_iuf', 'pearson_amplified' , 'pearson_iuf_amplified'
    amplification_factor = 1.5 
    main(train_file, test_file, output_file, similarity_metric,amplification_factor)