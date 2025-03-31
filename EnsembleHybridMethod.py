# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 16:52:27 2025

@author: ksben
"""

import numpy as np

def ensemble_predictions(test_case, model_files, mae_dict):
   
    weights = {}
    for file in model_files:
        model_name = file.replace('.txt', '')  
        weights[file] = 1 / mae_dict[model_name]
    
    
    total_weight = sum(weights.values())
    normalized_weights = {file: w / total_weight for file, w in weights.items()}
    
    
    predictions = []
    for file in model_files:
        with open(file, 'r') as f:
            preds = [float(line.split()[2]) for line in f]
        predictions.append(preds)
    
    weighted_avg = np.zeros_like(predictions[0])
    for i, preds in enumerate(predictions):
        weighted_avg += np.array(preds) * normalized_weights[model_files[i]]
    
    
    weighted_avg = np.clip(weighted_avg, 1.0, 5.0)
    weighted_avg = np.round(weighted_avg).astype(int)
    
    
    output_file = f"ENSEMBLE_FINAL_{test_case}.txt"
    with open(output_file, 'w') as f:
        for u_line, pred in zip(open(model_files[0], 'r'), weighted_avg):
            u, m, _ = u_line.strip().split()
            f.write(f"{u} {m} {pred}\n")
    print(f"Ensemble predictions saved to {output_file}")



test_config = {
    'test5': {
        'model_files': ['UBCFpearsonamplified5.txt', 'test5_predictions.txt'],
        'mae_dict': {'UBCFpearsonamplified5': 0.846567, 'test5_predictions': 0.816181}
    },
    'test10': {
        'model_files': ['UBCFpearsonamplified10.txt', 'test10_predictions.txt'],
        'mae_dict': {'UBCFpearsonamplified10': 0.780333, 'test10_predictions': 0.782}
    },
    'test20': {
        'model_files': ['IBCFAdjustedCosine20.txt', 'test20_predictions.txt'],
        'mae_dict': {'IBCFAdjustedCosine20': 0.769461, 'test20_predictions': 0.761262}
    }
}


for test_case in ['test5', 'test10', 'test20']:
    ensemble_predictions(
        test_case,
        test_config[test_case]['model_files'],
        test_config[test_case]['mae_dict']
    )