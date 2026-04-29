import pandas as pd
import torch
import numpy as np

def load_and_process_data(csv_path="dataset.csv"):
    df = pd.read_csv(csv_path)
    
    # Calculate pseudo-labels based on anomaly scoring
    # swipe_right_ratio > 0.85 -> bot_swipe_flag
    bot_swipe_flag = (df['swipe_right_ratio'] > 0.85).astype(int)
    
    # app_usage_time_min > 240 AND mutual_matches < 5 -> high_usage_low_match_flag
    high_usage_low_match_flag = ((df['app_usage_time_min'] > 240) & (df['mutual_matches'] < 5)).astype(int)
    
    # message_sent_count > 200 AND mutual_matches < 10 -> spam_message_flag
    spam_message_flag = ((df['message_sent_count'] > 200) & (df['mutual_matches'] < 10)).astype(int)
    
    # profile_pics_count <= 1 AND bio_length < 20 -> incomplete_profile_flag
    incomplete_profile_flag = ((df['profile_pics_count'] <= 1) & (df['bio_length'] < 20)).astype(int)
    
    # emoji_usage_rate > 0.75 -> emoji_spam_flag
    emoji_spam_flag = (df['emoji_usage_rate'] > 0.75).astype(int)
    
    # Risk score = weighted sum: 0.25, 0.25, 0.20, 0.15, 0.15
    risk_score = (
        bot_swipe_flag * 0.25 +
        high_usage_low_match_flag * 0.25 +
        spam_message_flag * 0.20 +
        incomplete_profile_flag * 0.15 +
        emoji_spam_flag * 0.15
    )
    
    # Convert risk score
    # >= 0.7 -> 2 (Inauthentic)
    # >= 0.4 -> 1 (Potentially Inauthentic)
    # else -> 0 (Authentic)
    labels = np.where(risk_score >= 0.7, 2, np.where(risk_score >= 0.4, 1, 0))
    
    # Select features required for models
    feature_cols = [
        'app_usage_time_min',
        'message_sent_count',
        'swipe_right_ratio',
        'likes_received',
        'mutual_matches'
    ]
    
    # Expand features to include all available stats for the GCN static features,
    # or just use the same 5 core continuous features.
    features = df[feature_cols].copy()
    features = features.fillna(0)
    
    # Standardize features (Z-score normalization)
    features = (features - features.mean()) / (features.std() + 1e-8)
    
    feature_matrix = torch.tensor(features.values, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return feature_matrix, labels_tensor, df

if __name__ == "__main__":
    # Local execution test
    # x, y, df = load_and_process_data("dataset.csv")
    pass
