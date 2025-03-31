import numpy as np

mae_weights = {
    'test5': {
        'UBCFcosine5': 0.853945,
        'UBCFpearson5': 0.906215,
        'UBCFpearsoniuf5': 0.906715,
        'UBCFpearsonamplified5': 0.846567,
        'UBCFpearsoniufamplified5': 0.853820,
        'IBCFAdjustedCosine5': 0.920720,
        'test5_predictions':0.816181067900463,
    },
    'test10': {
        'UBCFcosine10': 0.765333,
        'UBCFpearson10': 0.8325,
        'UBCFpearsoniuf10': 0.832667,
        'UBCFpearsonamplified10': 0.780333,
        'UBCFpearsoniufamplified10': 0.794167,
        'IBCFAdjustedCosine10': 0.826833,
        'test10_predictions':0.782,
    },
    'test20': {
        'UBCFcosine20': 0.779493,
        'UBCFpearson20': 0.828494,
        'UBCFpearsoniuf20': 0.828591,
        'UBCFpearsonamplified20': 0.774187,
        'UBCFpearsoniufamplified20': 0.784123,
        'IBCFAdjustedCosine20': 0.769461,
        'test20_predictions':0.761261695765409,
    }
}

def ensemble_predictions(test_case, model_files):
    mae_dict = mae_weights[test_case]
    
    weights = {}
    for file in model_files:
        model_name = file.replace('.txt', '')  
        weights[file] = 1 / mae_dict[model_name]
    total_weight = sum(weights.values())
    weights = {file: w / total_weight for file, w in weights.items()}
    
    
    predictions = []
    for file in model_files:
        with open(file, 'r') as f:
            preds = [float(line.split()[2]) for line in f]
        predictions.append(preds)
    
    
    weighted_avg = np.zeros_like(predictions[0])
    for i, preds in enumerate(predictions):
        weighted_avg += np.array(preds) * weights[model_files[i]]
    
    
    weighted_avg = np.clip(weighted_avg, 1.0, 5.0)
    
    
    output_file = 'Ensemble5.txt'  # e.g., Ensemble5.txt
    with open(output_file, 'w') as f:
        for u_line, pred in zip(open(model_files[0], 'r'), weighted_avg):
            u, m, _ = u_line.strip().split()
            pred=round(pred)
            f.write(f"{u} {m} {pred}\n")


test_case = 'test5'
model_files = [
    'UBCFcosine20.txt',
    'UBCFpearson20.txt',
    'UBCFpearsoniuf20.txt',
    'UBCFpearsonamplified20.txt',
    'UBCFpearsoniufamplified20.txt',
    'IBCFAdjustedCosine20.txt',
    'test20_predictions.txt'
]
model_files2=['UBCFpearsonamplified5.txt','test5_predictions.txt']
ensemble_predictions(test_case, model_files2)