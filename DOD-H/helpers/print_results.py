import json
import os

results_files = '../results'
dataset = 'DOD-H'
epochs = [5, 10, 30]

for file in os.listdir(results_files):
    if file.startswith(dataset) and file.endswith('new_data'):
        results_file = os.path.join(results_files, file, 'results.json')
        if not os.path.exists(results_file):
            print(f"Results file not found for {file}")
            continue
        with open(results_file, 'r') as f:
            results = json.load(f)
            seq_results = results['seq']
            print(f"Results for {file}: {seq_results['per_class_f1']}")
            print(f"ACC: {seq_results['accuracy_mean']} | MF1: {seq_results['mf1_mean']} | Kappa: {seq_results['kappa_mean']}\n")