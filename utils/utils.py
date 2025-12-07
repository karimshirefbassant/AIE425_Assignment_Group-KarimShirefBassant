import pandas as pd
import numpy as np
import math
import random

# ==========================================
# Team
# Shiref Khaled Elhalawany -  221100944
# Karim Ashraf Elsayed - 221100391
# Bassant Kamal Mesilam - 221100244
# ==========================================

# ==========================================
# Statistical Functions (Manual Implementation)
# ==========================================

def manual_mean(matrix, axis=0):
    """
    Computes the arithmetic mean along the specified axis.
    """
    matrix = np.asarray(matrix, dtype=float)
    
    if axis == 0:  
        n = matrix.shape[0]
        sums = np.sum(matrix, axis=0)
        return sums / n
    elif axis == 1:  
        n = matrix.shape[1]  
        sums = np.sum(matrix, axis=1)
        return sums / n
    else:
        raise ValueError("axis must be 0 or 1")

def manual_std(matrix, axis=0, ddof=0):
    """
    Computes the standard deviation along the specified axis.
    """
    matrix = np.asarray(matrix, dtype=float)
    mean = manual_mean(matrix, axis=axis)

    if axis == 0:
        diff = matrix - mean
        n = matrix.shape[0]
        var = np.sum(diff ** 2, axis=0) / (n - ddof)
        return np.sqrt(var)
    elif axis == 1: 
        diff = matrix - mean[:, None]
        n = matrix.shape[1]
        var = np.sum(diff ** 2, axis=1) / (n - ddof)
        return np.sqrt(var)
    else:
        raise ValueError("axis must be 0 or 1")

def manual_zscore(matrix, means=None, stds=None, axis=0):
    """
    Computes the Z-score standardization of the matrix.
    """
    matrix = np.asarray(matrix, dtype=float)

    if means is None:
        means = manual_mean(matrix, axis=axis)
    if stds is None:
        stds = manual_std(matrix, axis=axis)

    stds_safe = stds.copy()
    stds_safe[stds_safe == 0] = 1.0

    return (matrix - means) / stds_safe, means, stds_safe

def manual_pearson_corr(ratings1, ratings2):
    """
    Calculates the Pearson correlation coefficient between two dictionaries of ratings.
    """
    common_items = list(set(ratings1.keys()) & set(ratings2.keys()))
    
    if len(common_items) < 2:
        return 0.0 
    
    r1_vals = np.array([ratings1[i] for i in common_items])
    r2_vals = np.array([ratings2[i] for i in common_items])
    
    mean1 = np.mean(r1_vals)
    mean2 = np.mean(r2_vals)
    
    numerator = np.sum((r1_vals - mean1) * (r2_vals - mean2))
    denominator = np.sqrt(np.sum((r1_vals - mean1)**2)) * np.sqrt(np.sum((r2_vals - mean2)**2))
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator

def manual_euclidean_distance(p1, p2):
    """
    Calculates the Euclidean distance between two points.
    """
    return np.sqrt(np.sum((p1 - p2)**2))

# ==========================================
# Clustering Functions (Manual Implementation)
# ==========================================

def manual_kmeans(X, k, max_iters=100, tol=1e-4, random_state=42):
    """
    Performs K-Means clustering manually.
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    
    labels = np.zeros(n_samples) # Initialize labels
    
    for i in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centroids[j] for j in range(k)])
        
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
            
        centroids = new_centroids
        
    wcss = 0
    for j in range(k):
        cluster_points = X[labels == j]
        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - centroids[j])**2)
            
    return labels, centroids, wcss

def manual_silhouette_score(X, labels):
    """
    Calculates the Silhouette Score manually.
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1 or n_clusters >= n_samples:
        return 0.0
    
    silhouette_vals = []
    clusters = {l: X[labels == l] for l in unique_labels}
    
    for i in range(n_samples):
        point = X[i]
        label = labels[i]
        
        own_cluster_points = clusters[label]
        if len(own_cluster_points) > 1:
            dists_a = np.linalg.norm(own_cluster_points - point, axis=1)
            a_i = np.sum(dists_a) / (len(own_cluster_points) - 1)
        else:
            a_i = 0
            
        b_i = np.inf
        for other_label in unique_labels:
            if other_label == label:
                continue
                
            other_cluster_points = clusters[other_label]
            if len(other_cluster_points) > 0:
                dists_b = np.linalg.norm(other_cluster_points - point, axis=1)
                mean_dist_b = np.mean(dists_b)
                if mean_dist_b < b_i:
                    b_i = mean_dist_b
        
        if b_i == np.inf: 
            b_i = 0
            
        max_ab = max(a_i, b_i)
        if max_ab == 0:
            s_i = 0
        else:
            s_i = (b_i - a_i) / max_ab
            
        silhouette_vals.append(s_i)
        
    return np.mean(silhouette_vals)

# ==========================================
# Similarity Functions
# ==========================================

def calculate_raw_cosine_similarity(user1, user2, user_ratings):
    """
    Calculates the raw Cosine Similarity between two users.
    """
    u1_items = user_ratings.get(user1, {})
    u2_items = user_ratings.get(user2, {})

    common_items = set(u1_items.keys()) & set(u2_items.keys())

    if not common_items:
        return 0.0, 0

    dot_product = 0.0
    sum_sq_1 = 0.0
    sum_sq_2 = 0.0

    for item in common_items:
        r1 = u1_items[item]
        r2 = u2_items[item]

        dot_product += r1 * r2
        sum_sq_1 += r1 ** 2
        sum_sq_2 += r2 ** 2

    norm1 = sum_sq_1** 0.5
    norm2 = sum_sq_2** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0, len(common_items)

    similarity = dot_product / (norm1 * norm2)
    return similarity, len(common_items)

def calculate_adjusted_cosine_similarity(item1, item2, item_user_ratings, user_avgs):
    """
    Calculates the Adjusted Cosine Similarity between two items (subtracting user means).
    """
    u1_users = item_user_ratings.get(item1, {})
    u2_users = item_user_ratings.get(item2, {})
    
    common_users = set(u1_users.keys()) & set(u2_users.keys())
    
    if not common_users:
        return 0.0, 0
        
    numerator = 0.0
    sum_sq_1 = 0.0
    sum_sq_2 = 0.0
    
    for user in common_users:
        r1 = u1_users[user]
        r2 = u2_users[user]
        user_avg = user_avgs[user]
        
        r1_centered = r1 - user_avg
        r2_centered = r2 - user_avg
        
        numerator += r1_centered * r2_centered
        sum_sq_1 += r1_centered ** 2
        sum_sq_2 += r2_centered ** 2
        
    norm1 = sum_sq_1 ** 0.5
    norm2 = sum_sq_2 ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0, len(common_users)
        
    similarity = numerator / (norm1 * norm2)
    return similarity, len(common_users)

def calculate_discounted_similarity(df_similarities, user_item_ratings, beta_pct=0.3, sim_col='Similarity'):
    """
    Calculates Discounted Similarity based on the number of common items.
    """
    ds_list = []

    for target_user, group in df_similarities.groupby('TargetUser'):
        num_rated_by_target = len(user_item_ratings.get(target_user, {}))
        
        beta = math.ceil(num_rated_by_target * beta_pct)
        
        for _, row in group.iterrows():
            other_user = row['OtherUser']
            similarity = row[sim_col]
            common_items = row['CommonItems']
            
            if beta > 0:
                df = min(common_items / beta, 1.0)
            else:
                df = 1.0 
                
            ds = similarity * df
            
            ds_entry = row.to_dict()
            ds_entry['DiscountFactor'] = round(df, 2)
            ds_entry['DiscountedSimilarity'] = round(ds, 2)
            
            ds_list.append(ds_entry)

    return pd.DataFrame(ds_list)

def calculate_discounted_similarity_items(df_similarities, item_user_ratings, beta_pct=0.3, sim_col='Similarity'):
    """
    Calculates Discounted Similarity for items based on common users.
    """
    ds_list = []

    for target_item, group in df_similarities.groupby('TargetItem'):
        num_users_for_target = len(item_user_ratings.get(target_item, {}))

        beta = math.ceil(num_users_for_target * beta_pct)

        for _, row in group.iterrows():
            other_item = row['OtherItem']
            similarity = row[sim_col]
            common_users = row['CommonUsers']

            if beta > 0:
                df = min(common_users / beta, 1.0)
            else:
                df = 1.0

            ds = similarity * df

            ds_entry = row.to_dict()
            ds_entry['DiscountFactor'] = round(df, 2)
            ds_entry['DiscountedSimilarity'] = round(ds, 2)

            ds_list.append(ds_entry)

    return pd.DataFrame(ds_list)

# ==========================================
# Prediction & Helper Functions
# ==========================================

def compute_user_averages(user_ratings_list):
    """
    Computes the average rating for each user.
    """
    user_avgs = {}\
    for user, ratings in user_ratings_list.items():
        if len(ratings) > 0:
            user_avgs[user] = sum(ratings) / len(ratings)
        else:
            user_avgs[user] = 0.0
    return user_avgs

def get_top_n_similar_users(df_similarities, n_percentage=0.20, similarity_col='Similarity'):
    """
    Selects the top N% similar users for each target user.
    """
    top_similar_users = []

    for target_user, group in df_similarities.groupby('TargetUser'):
        sorted_group = group.sort_values(by=similarity_col, ascending=False)
        
        n_top = math.ceil(len(sorted_group) * n_percentage)
        
        top_users = sorted_group.head(n_top)
        
        top_similar_users.append(top_users)

    if top_similar_users:
        return pd.concat(top_similar_users)
    else:
        return pd.DataFrame()

def get_top_n_similar_items(df_similarities, n_percentage=0.20, similarity_col='Similarity'):
    """
    Selects the top N% similar items for each target item.
    """
    top_similar_items = []

    for target_item, group in df_similarities.groupby('TargetItem'):
        sorted_group = group.sort_values(by=similarity_col, ascending=False)

        n_top = math.ceil(len(sorted_group) * n_percentage)

        top_items = sorted_group.head(n_top)

        top_similar_items.append(top_items)

    if top_similar_items:
        return pd.concat(top_similar_items)
    else:
        return pd.DataFrame()

def predict_ratings(df_top_users, user_item_ratings, sim_col='Similarity'):
    """
    Predicts ratings for unrated items using User-Based CF.
    """
    predictions = []

    for target_user, group in df_top_users.groupby('TargetUser'):
        target_user_items = set(user_item_ratings.get(target_user, {}).keys())
        
        candidate_items = set()
        for _, row in group.iterrows():
            other_user = row['OtherUser']
            other_user_items = user_item_ratings.get(other_user, {}).keys()
            candidate_items.update(other_user_items)
        
        unknown_items = candidate_items - target_user_items
        
        for item in unknown_items:
            numerator = 0.0
            denominator = 0.0
            
            for _, row in group.iterrows():
                other_user = row['OtherUser']
                
                similarity = row[sim_col]
                
                rating = user_item_ratings.get(other_user, {}).get(item)
                
                if rating is not None:
                    numerator += similarity * rating
                    denominator += abs(similarity)
            
            if denominator > 0:
                predicted_rating = numerator / denominator
                predictions.append({
                    'TargetUser': target_user,
                    'Item': item,
                    'PredictedRating': round(predicted_rating, 2),
                    'SimilarityType': sim_col
                })
                
    return pd.DataFrame(predictions)

def predict_ratings_item_based(df_top_items, item_user_ratings, sim_col='Similarity'):
    """
    Predicts ratings for unrated items using Item-Based CF.
    """
    predictions = []

    for target_item, group in df_top_items.groupby('TargetItem'):
        target_item_users = set(item_user_ratings.get(target_item, {}).keys())

        candidate_users = set()
        for _, row in group.iterrows():
            other_item = row['OtherItem']
            other_item_users = item_user_ratings.get(other_item, {}).keys()
            candidate_users.update(other_item_users)

        unknown_users = candidate_users - target_item_users

        for user in unknown_users:
            numerator = 0.0
            denominator = 0.0

            for _, row in group.iterrows():
                other_item = row['OtherItem']

                similarity = row[sim_col]

                rating = item_user_ratings.get(other_item, {}).get(user)

                if rating is not None:
                    numerator += similarity * rating
                    denominator += abs(similarity)

            if denominator > 0:
                predicted_rating = numerator / denominator
                predictions.append({
                    'UserID': user,
                    'Item': target_item,
                    'PredictedRating': round(predicted_rating, 2),
                    'SimilarityType': sim_col
                })

    return pd.DataFrame(predictions)
