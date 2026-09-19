import time
import json
import os
import urllib.request
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# We will dynamically import XGBoost in main function

# Set style for publication quality plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

DATASET_PATH = "diabetes.csv"
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

def load_and_preprocess_data():
    if not os.path.exists(DATASET_PATH):
        print("Downloading Pima Indians Diabetes Dataset...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    
    # Load dataset
    df = pd.read_csv(DATASET_PATH, names=COLUMN_NAMES) if not pd.read_csv(DATASET_PATH).columns[0] == 'Pregnancies' else pd.read_csv(DATASET_PATH)
    if 'Pregnancies' not in df.columns:
        df.columns = COLUMN_NAMES
        
    print(f"Dataset Loaded. Shape: {df.shape}")
    
    # Impute missing values represented as 0 in physiological parameters
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        median_val = df[df[col] != 0][col].median()
        df[col] = df[col].replace(0, median_val)
        
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, X.columns.tolist()

def get_models():
    import xgboost as xgb
    models = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'MLP (Dense NN)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss'),
        'Gaussian Naive Bayes': GaussianNB()
    }
    return models

def run_scenario_a(X_train, y_train, X_test, y_test):
    print("\n--- Running Scenario A: Sample Size Variation (N) ---")
    percentages = [0.10, 0.25, 0.50, 0.75, 1.00]
    n_samples_list = [int(p * len(X_train)) for p in percentages]
    
    results = {model_name: {'fit_time': [], 'predict_time': []} for model_name in get_models().keys()}
    
    for p, n_samples in zip(percentages, n_samples_list):
        # Subset training data
        indices = np.random.choice(len(X_train), size=n_samples, replace=False)
        X_sub, y_sub = X_train[indices], y_train[indices]
        
        models = get_models()
        for name, model in models.items():
            # Time training (.fit)
            start_fit = time.perf_counter()
            model.fit(X_sub, y_sub)
            end_fit = time.perf_counter()
            fit_time = end_fit - start_fit
            
            # Time prediction (.predict)
            start_pred = time.perf_counter()
            _ = model.predict(X_test)
            end_pred = time.perf_counter()
            pred_time = end_pred - start_pred
            
            results[name]['fit_time'].append(fit_time)
            results[name]['predict_time'].append(pred_time)
            print(f"Percent: {int(p*100)}% ({n_samples} samples) | {name:<20} | Fit: {fit_time*1000:.2f} ms | Pred: {pred_time*1000:.2f} ms")
            
    return percentages, n_samples_list, results

def run_scenario_b(X_train, y_train, X_test, y_test):
    print("\n--- Running Scenario B: Feature Size Variation (D) ---")
    percentages = [0.20, 0.40, 0.60, 0.80, 1.00]
    total_features = X_train.shape[1]
    k_features_list = [max(1, int(round(p * total_features))) for p in percentages]
    
    results = {model_name: {'fit_time': [], 'predict_time': []} for model_name in get_models().keys()}
    
    for p, k in zip(percentages, k_features_list):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_k = selector.fit_transform(X_train, y_train)
        X_test_k = selector.transform(X_test)
        
        models = get_models()
        for name, model in models.items():
            # Time training (.fit)
            start_fit = time.perf_counter()
            model.fit(X_train_k, y_train)
            end_fit = time.perf_counter()
            fit_time = end_fit - start_fit
            
            # Time prediction (.predict)
            start_pred = time.perf_counter()
            _ = model.predict(X_test_k)
            end_pred = time.perf_counter()
            pred_time = end_pred - start_pred
            
            results[name]['fit_time'].append(fit_time)
            results[name]['predict_time'].append(pred_time)
            print(f"Feature %: {int(p*100)}% ({k} features) | {name:<20} | Fit: {fit_time*1000:.2f} ms | Pred: {pred_time*1000:.2f} ms")
            
    return percentages, k_features_list, results

def run_scenario_c(X_train, y_train, X_test):
    print("\n--- Running Scenario C: Inference Batch Scalability ---")
    batch_sizes = [1, 32, len(X_test)]
    batch_labels = ['1 (Single Sample)', '32 (Small Batch)', f'{len(X_test)} (Bulk Test Set)']
    
    models = get_models()
    for model in models.values():
        model.fit(X_train, y_train)
        
    results = {model_name: [] for model_name in models.keys()}
    
    for bs in batch_sizes:
        X_batch = X_test[:bs]
        for name, model in models.items():
            if bs == 1:
                # Average over multiple runs for sub-ms precision on single sample
                iterations = 50
                start_pred = time.perf_counter()
                for _ in range(iterations):
                    _ = model.predict(X_batch)
                end_pred = time.perf_counter()
                pred_time = (end_pred - start_pred) / iterations
            else:
                start_pred = time.perf_counter()
                _ = model.predict(X_batch)
                end_pred = time.perf_counter()
                pred_time = end_pred - start_pred
                
            results[name].append(pred_time)
            print(f"Batch Size: {bs:<20} | {name:<20} | Pred Time: {pred_time*1000:.4f} ms")
            
    return batch_sizes, batch_labels, results

def run_scenario_d(X_train, y_train):
    print("\n--- Running Scenario D: Convergence Normal vs Noisy Data ---")
    # Add Gaussian noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, size=X_train.shape)
    X_train_noisy = X_train + noise
    
    results = {model_name: {'clean_fit': 0.0, 'noisy_fit': 0.0} for model_name in get_models().keys()}
    
    models_clean = get_models()
    models_noisy = get_models()
    
    for name in models_clean.keys():
        # Fit Clean
        start = time.perf_counter()
        models_clean[name].fit(X_train, y_train)
        fit_clean = time.perf_counter() - start
        
        # Fit Noisy
        start = time.perf_counter()
        models_noisy[name].fit(X_train_noisy, y_train)
        fit_noisy = time.perf_counter() - start
        
        results[name]['clean_fit'] = fit_clean
        results[name]['noisy_fit'] = fit_noisy
        diff_pct = ((fit_noisy - fit_clean) / fit_clean) * 100
        print(f"{name:<20} | Clean Fit: {fit_clean*1000:.2f} ms | Noisy Fit: {fit_noisy*1000:.2f} ms | Diff: {diff_pct:+.2f}%")
        
    return results

def calculate_final_metrics(X_train, y_train, X_test, y_test):
    print("\n--- Summary Metrics & Final Evaluation ---")
    models = get_models()
    summary = []
    
    for name, model in models.items():
        # Fit
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t_fit = time.perf_counter() - t0
        
        # Predict
        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        t_pred = time.perf_counter() - t1
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        summary.append({
            'Algorithm': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Fit Time (s)': t_fit,
            'Fit Time (ms)': t_fit * 1000,
            'Predict Time (s)': t_pred,
            'Predict Time (ms)': t_pred * 1000
        })
        
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    return summary_df

def generate_plots(scen_a, scen_b, scen_c, scen_d):
    colors = {
        'SVM': '#1f77b4',
        'MLP (Dense NN)': '#ff7f0e',
        'Random Forest': '#2ca02c',
        'XGBoost': '#d62728',
        'Gaussian Naive Bayes': '#9467bd'
    }
    markers = {
        'SVM': 'o',
        'MLP (Dense NN)': 's',
        'Random Forest': '^',
        'XGBoost': 'D',
        'Gaussian Naive Bayes': 'v'
    }
    
    # 1. Scenario A Plot
    p_a, n_a, res_a = scen_a
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    x_axis = [int(p*100) for p in p_a]
    for model_name, data in res_a.items():
        ax[0].plot(x_axis, [t*1000 for t in data['fit_time']], label=model_name, color=colors[model_name], marker=markers[model_name], linewidth=2)
        ax[1].plot(x_axis, [t*1000 for t in data['predict_time']], label=model_name, color=colors[model_name], marker=markers[model_name], linewidth=2)
        
    ax[0].set_title('Skenario A: Training Time vs Sample Percentage (N)', fontweight='bold')
    ax[0].set_xlabel('Persentase Data Latih (%)')
    ax[0].set_ylabel('Training Time (ms)')
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.7)
    
    ax[1].set_title('Skenario A: Inference Time vs Sample Percentage (N)', fontweight='bold')
    ax[1].set_xlabel('Persentase Data Latih (%)')
    ax[1].set_ylabel('Inference Time (ms)')
    ax[1].legend()
    ax[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('scenario_a_samples.png', dpi=300)
    plt.close()
    
    # 2. Scenario B Plot
    p_b, k_b, res_b = scen_b
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    x_axis_b = [int(p*100) for p in p_b]
    for model_name, data in res_b.items():
        ax[0].plot(x_axis_b, [t*1000 for t in data['fit_time']], label=model_name, color=colors[model_name], marker=markers[model_name], linewidth=2)
        ax[1].plot(x_axis_b, [t*1000 for t in data['predict_time']], label=model_name, color=colors[model_name], marker=markers[model_name], linewidth=2)
        
    ax[0].set_title('Skenario B: Training Time vs Feature Percentage (D)', fontweight='bold')
    ax[0].set_xlabel('Persentase Fitur Terpilih (%)')
    ax[0].set_ylabel('Training Time (ms)')
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.7)
    
    ax[1].set_title('Skenario B: Inference Time vs Feature Percentage (D)', fontweight='bold')
    ax[1].set_xlabel('Persentase Fitur Terpilih (%)')
    ax[1].set_ylabel('Inference Time (ms)')
    ax[1].legend()
    ax[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('scenario_b_features.png', dpi=300)
    plt.close()
    
    # 3. Scenario C Plot
    bs_c, labels_c, res_c = scen_c
    plt.figure(figsize=(9, 5))
    for model_name, times in res_c.items():
        plt.plot(labels_c, [t*1000 for t in times], label=model_name, color=colors[model_name], marker=markers[model_name], linewidth=2)
        
    plt.title('Skenario C: Inference Time Scalability across Batch Sizes', fontweight='bold')
    plt.xlabel('Ukuran Batch Prediksi (Batch Size)')
    plt.ylabel('Inference Time (ms)')
    plt.yscale('log') # Log scale for better visibility of 1 vs 32 vs bulk
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('scenario_c_inference.png', dpi=300)
    plt.close()
    
    # 4. Scenario D Plot
    res_d = scen_d
    models_list = list(res_d.keys())
    clean_times = [res_d[m]['clean_fit']*1000 for m in models_list]
    noisy_times = [res_d[m]['noisy_fit']*1000 for m in models_list]
    
    x = np.arange(len(models_list))
    width = 0.35
    
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, clean_times, width, label='Data Normal (Clean)', color='#2b5c8f')
    plt.bar(x + width/2, noisy_times, width, label='Data Noisy (+Gaussian Noise)', color='#e05d5d')
    
    plt.title('Skenario D: Waktu Training Data Normal vs Data Noisy', fontweight='bold')
    plt.xticks(x, models_list, rotation=15)
    plt.ylabel('Training Time (ms)')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('scenario_d_noise.png', dpi=300)
    plt.close()
    
    print("All plots saved successfully!")

if __name__ == '__main__':
    np.random.seed(42)
    X_train, X_test, y_train, y_test, feat_names = load_and_preprocess_data()
    
    scen_a = run_scenario_a(X_train, y_train, X_test, y_test)
    scen_b = run_scenario_b(X_train, y_train, X_test, y_test)
    scen_c = run_scenario_c(X_train, y_train, X_test)
    scen_d = run_scenario_d(X_train, y_train)
    
    summary_df = calculate_final_metrics(X_train, y_train, X_test, y_test)
    generate_plots(scen_a, scen_b, scen_c, scen_d)
    
    # Save data json
    export_data = {
        'scen_a': {
            'percentages': scen_a[0],
            'n_samples': scen_a[1],
            'results': scen_a[2]
        },
        'scen_b': {
            'percentages': scen_b[0],
            'k_features': scen_b[1],
            'results': scen_b[2]
        },
        'scen_c': {
            'batch_sizes': scen_c[0],
            'batch_labels': scen_c[1],
            'results': scen_c[2]
        },
        'scen_d': scen_d,
        'summary': summary_df.to_dict(orient='records')
    }
    with open('benchmark_results.json', 'w') as f:
        json.dump(export_data, f, indent=2)
        
    print("\nBenchmark experiment completed successfully!")
