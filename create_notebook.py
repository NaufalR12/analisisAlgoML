import nbformat as nbf
import json
import os

nb = nbf.v4.new_notebook()

# Read benchmark results json to embed exact outputs if needed
with open('benchmark_results.json', 'r') as f:
    benchmark_data = json.load(f)

cells = []

# Title & Introduction
cells.append(nbf.v4.new_markdown_cell("""# Time Complexity Benchmarking & Evaluasi Performa 5 Algoritma Machine Learning

**Mata Kuliah**: Analisis Algoritma Lanjut  
**Dataset**: Pima Indians Diabetes (`diabetes.csv`)  
**Algoritma yang Diuji**:
1. Support Vector Machine (SVM)
2. Multi-Layer Perceptron / Dense Neural Network (MLP)
3. Random Forest (RF)
4. XGBoost
5. Gaussian Naive Bayes (GNB)

---

## 1. Import Library & Pengaturan Awal
Pada tahap ini, kita mengimpor library utama seperti `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, dan `seaborn` serta mengonfigurasi presisi pencatatan waktu dengan `time.perf_counter()`.
"""))

cells.append(nbf.v4.new_code_cell("""import time
import json
import os
import urllib.request
import numpy as np
import pandas as pd
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
import xgboost as xgb

# Konfigurasi Tampilan Visualisasi
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})
print("Seluruh library berhasil diimpor!")
"""))

# Data Loading & Preprocessing
cells.append(nbf.v4.new_markdown_cell("""## 2. Pemuatan Data & Imputasi Preprocessing
Dataset *Pima Indians Diabetes* memiliki 768 baris data tabular. Beberapa kolom fisiologis mengandung nilai `0` implisit yang sebenarnya merupakan *missing values*. Kita melakukan imputasi menggunakan *median* kolom non-zero, kemudian membagi dataset menjadi data latih (80%) dan data uji (20%), diikuti skalisasi `StandardScaler`.
"""))

cells.append(nbf.v4.new_code_cell("""DATASET_PATH = "diabetes.csv"
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

if not os.path.exists(DATASET_PATH):
    print("Mengunduh Pima Indians Diabetes Dataset...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)

df = pd.read_csv(DATASET_PATH, names=COLUMN_NAMES) if not pd.read_csv(DATASET_PATH).columns[0] == 'Pregnancies' else pd.read_csv(DATASET_PATH)
if 'Pregnancies' not in df.columns:
    df.columns = COLUMN_NAMES

# Imputasi nilai 0 dengan Median
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

print(f"Data Latih: {X_train_scaled.shape}, Data Uji: {X_test_scaled.shape}")
df.head()
"""))

# Model Architecture Definition
cells.append(nbf.v4.new_markdown_cell("""## 3. Inisialisasi 5 Algoritma Machine Learning
Kita mendefinisikan fungsi pembantu untuk menghasilkan instance baru dari 5 algoritma dengan *hyperparameter* standar yang adil (*apple-to-apple*).
"""))

cells.append(nbf.v4.new_code_cell("""def get_models():
    return {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'MLP (Dense NN)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss'),
        'Gaussian Naive Bayes': GaussianNB()
    }

for name, model in get_models().items():
    print(f"Model Siap: {name:<20} -> {model.__class__.__name__}")
"""))

# Scenario A
cells.append(nbf.v4.new_markdown_cell("""## 4. Skenario A: Variasi Jumlah Sampel ($N$)
Mengevaluasi waktu *fit* dan *predict* ketika data latih bervariasi dari 10%, 25%, 50%, 75%, hingga 100% dari total baris.
"""))

cells.append(nbf.v4.new_code_cell("""percentages = [0.10, 0.25, 0.50, 0.75, 1.00]
n_samples_list = [int(p * len(X_train_scaled)) for p in percentages]
results_a = {model_name: {'fit_time': [], 'predict_time': []} for model_name in get_models().keys()}

for p, n_samples in zip(percentages, n_samples_list):
    np.random.seed(42)
    indices = np.random.choice(len(X_train_scaled), size=n_samples, replace=False)
    X_sub, y_sub = X_train_scaled[indices], y_train.values[indices]
    
    models = get_models()
    for name, model in models.items():
        # Measured with time.perf_counter()
        t0 = time.perf_counter()
        model.fit(X_sub, y_sub)
        t_fit = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        _ = model.predict(X_test_scaled)
        t_pred = time.perf_counter() - t1
        
        results_a[name]['fit_time'].append(t_fit)
        results_a[name]['predict_time'].append(t_pred)

print("Skenario A Selesai Eksplorasi!")
"""))

# Scenario B
cells.append(nbf.v4.new_markdown_cell("""## 5. Skenario B: Variasi Jumlah Fitur ($D$)
Menggunakan `SelectKBest` dengan skor ANOVA F-value, kita bervariasi fitur dari 20%, 40%, 60%, 80%, hingga 100% dimensi fitur.
"""))

cells.append(nbf.v4.new_code_cell("""percentages_b = [0.20, 0.40, 0.60, 0.80, 1.00]
total_features = X_train_scaled.shape[1]
k_features_list = [max(1, int(round(p * total_features))) for p in percentages_b]
results_b = {model_name: {'fit_time': [], 'predict_time': []} for model_name in get_models().keys()}

for p, k in zip(percentages_b, k_features_list):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_k = selector.fit_transform(X_train_scaled, y_train.values)
    X_test_k = selector.transform(X_test_scaled)
    
    models = get_models()
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train_k, y_train.values)
        t_fit = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        _ = model.predict(X_test_k)
        t_pred = time.perf_counter() - t1
        
        results_b[name]['fit_time'].append(t_fit)
        results_b[name]['predict_time'].append(t_pred)

print("Skenario B Selesai Eksplorasi!")
"""))

# Scenario C
cells.append(nbf.v4.new_markdown_cell("""## 6. Skenario C: Skalabilitas Batch Prediksi (*Inference Speed*)
Mengukur waktu prediksi untuk Single Sample (Batch Size = 1), Small Batch (Batch Size = 32), dan Bulk Test Set (Batch Size = 154).
"""))

cells.append(nbf.v4.new_code_cell("""batch_sizes = [1, 32, len(X_test_scaled)]
batch_labels = ['1 (Single Sample)', '32 (Small Batch)', f'{len(X_test_scaled)} (Bulk Test Set)']

models_trained = get_models()
for model in models_trained.values():
    model.fit(X_train_scaled, y_train.values)

results_c = {model_name: [] for model_name in models_trained.keys()}

for bs in batch_sizes:
    X_batch = X_test_scaled[:bs]
    for name, model in models_trained.items():
        if bs == 1:
            iterations = 50
            t0 = time.perf_counter()
            for _ in range(iterations):
                _ = model.predict(X_batch)
            t_pred = (time.perf_counter() - t0) / iterations
        else:
            t0 = time.perf_counter()
            _ = model.predict(X_batch)
            t_pred = time.perf_counter() - t0
            
        results_c[name].append(t_pred)

print("Skenario C Selesai Eksplorasi!")
"""))

# Scenario D
cells.append(nbf.v4.new_markdown_cell("""## 7. Skenario D: Sensitivitas Data Normal vs Data Noisy
Mengukur pengaruh penambahan *Gaussian Noise* $\mathcal{N}(0, 0.5^2)$ pada fitur latih terhadap waktu *training*.
"""))

cells.append(nbf.v4.new_code_cell("""np.random.seed(42)
noise = np.random.normal(0, 0.5, size=X_train_scaled.shape)
X_train_noisy = X_train_scaled + noise

results_d = {model_name: {'clean_fit': 0.0, 'noisy_fit': 0.0} for model_name in get_models().keys()}

models_clean = get_models()
models_noisy = get_models()

for name in models_clean.keys():
    # Clean
    t0 = time.perf_counter()
    models_clean[name].fit(X_train_scaled, y_train.values)
    results_d[name]['clean_fit'] = time.perf_counter() - t0
    
    # Noisy
    t0 = time.perf_counter()
    models_noisy[name].fit(X_train_noisy, y_train.values)
    results_d[name]['noisy_fit'] = time.perf_counter() - t0

print("Skenario D Selesai Eksplorasi!")
"""))

# Visualizations
cells.append(nbf.v4.new_markdown_cell("""## 8. Visualisasi Kurva Pertumbuhan Waktu (Figur 1 - 4)
Menampilkan 4 grafik line plot terpisah untuk membandingkan kurva pertumbuhan waktu latih dan prediksi secara visual.
"""))

cells.append(nbf.v4.new_code_cell("""colors = {'SVM': '#1f77b4', 'MLP (Dense NN)': '#ff7f0e', 'Random Forest': '#2ca02c', 'XGBoost': '#d62728', 'Gaussian Naive Bayes': '#9467bd'}
markers = {'SVM': 'o', 'MLP (Dense NN)': 's', 'Random Forest': '^', 'XGBoost': 'D', 'Gaussian Naive Bayes': 'v'}

# Figur 1: Skenario A
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
x_a = [int(p*100) for p in percentages]
for name, data in results_a.items():
    ax[0].plot(x_a, [t*1000 for t in data['fit_time']], label=name, color=colors[name], marker=markers[name], linewidth=2)
    ax[1].plot(x_a, [t*1000 for t in data['predict_time']], label=name, color=colors[name], marker=markers[name], linewidth=2)

ax[0].set_title('Skenario A: Training Time vs Sample % (N)', fontweight='bold')
ax[0].set_xlabel('Persentase Data Latih (%)')
ax[0].set_ylabel('Training Time (ms)')
ax[0].legend(); ax[0].grid(True, linestyle='--')

ax[1].set_title('Skenario A: Inference Time vs Sample % (N)', fontweight='bold')
ax[1].set_xlabel('Persentase Data Latih (%)')
ax[1].set_ylabel('Inference Time (ms)')
ax[1].legend(); ax[1].grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# Figur 2: Skenario B
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
x_b = [int(p*100) for p in percentages_b]
for name, data in results_b.items():
    ax[0].plot(x_b, [t*1000 for t in data['fit_time']], label=name, color=colors[name], marker=markers[name], linewidth=2)
    ax[1].plot(x_b, [t*1000 for t in data['predict_time']], label=name, color=colors[name], marker=markers[name], linewidth=2)

ax[0].set_title('Skenario B: Training Time vs Feature % (D)', fontweight='bold')
ax[0].set_xlabel('Persentase Fitur Terpilih (%)')
ax[0].set_ylabel('Training Time (ms)')
ax[0].legend(); ax[0].grid(True, linestyle='--')

ax[1].set_title('Skenario B: Inference Time vs Feature % (D)', fontweight='bold')
ax[1].set_xlabel('Persentase Fitur Terpilih (%)')
ax[1].set_ylabel('Inference Time (ms)')
ax[1].legend(); ax[1].grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# Figur 3: Skenario C
plt.figure(figsize=(9, 5))
for name, times in results_c.items():
    plt.plot(batch_labels, [t*1000 for t in times], label=name, color=colors[name], marker=markers[name], linewidth=2)
plt.title('Skenario C: Inference Time Scalability across Batch Sizes', fontweight='bold')
plt.xlabel('Ukuran Batch Prediksi')
plt.ylabel('Inference Time (ms)')
plt.yscale('log')
plt.legend(); plt.grid(True, which="both", linestyle='--')
plt.tight_layout()
plt.show()

# Figur 4: Skenario D
plt.figure(figsize=(10, 5))
m_names = list(results_d.keys())
clean_t = [results_d[m]['clean_fit']*1000 for m in m_names]
noisy_t = [results_d[m]['noisy_fit']*1000 for m in m_names]
x = np.arange(len(m_names)); width = 0.35
plt.bar(x - width/2, clean_t, width, label='Data Normal (Clean)', color='#2b5c8f')
plt.bar(x + width/2, noisy_t, width, label='Data Noisy (+Noise)', color='#e05d5d')
plt.title('Skenario D: Waktu Training Data Normal vs Data Noisy', fontweight='bold')
plt.xticks(x, m_names, rotation=15); plt.ylabel('Training Time (ms)')
plt.legend(); plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.show()
"""))

# Final Metrics Summary Table
cells.append(nbf.v4.new_markdown_cell("""## 9. Tabel Ringkasan Komparasi Performa Akhir
Menampilkan hasil komparasi akhir antara Metrik Efektivitas (Accuracy, Precision, Recall, F1-Score) dan Metrik Efisiensi (Fit Time & Predict Time) pada 100% dataset.
"""))

cells.append(nbf.v4.new_code_cell("""models = get_models()
summary = []

for name, model in models.items():
    t0 = time.perf_counter()
    model.fit(X_train_scaled, y_train.values)
    t_fit = time.perf_counter() - t0
    
    t1 = time.perf_counter()
    y_pred = model.predict(X_test_scaled)
    t_pred = time.perf_counter() - t1
    
    summary.append({
        'Algorithm': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Fit Time (ms)': t_fit * 1000,
        'Predict Time (ms)': t_pred * 1000
    })

summary_df = pd.DataFrame(summary)
summary_df.sort_values(by='F1-Score', ascending=False, inplace=True)
summary_df.style.highlight_max(subset=['Accuracy', 'F1-Score'], color='lightgreen')\
          .highlight_min(subset=['Fit Time (ms)', 'Predict Time (ms)'], color='lightblue')
"""))

# Academic Report Markdown Cell inside Notebook
cells.append(nbf.v4.new_markdown_cell("""## 10. Draf Laporan Analisis Akademis (Teoritis vs Empiris)

### 10.1. Notasi Big-O teoritis
1. **Gaussian Naive Bayes**: Fit $\mathcal{O}(N \cdot D)$, Predict $\mathcal{O}(C \cdot D)$. Menghitung statistik mean/variansi per fitur. Algoritma paling efisien.
2. **Random Forest**: Fit $\mathcal{O}(K \cdot N \log N \cdot \sqrt{D})$, Predict $\mathcal{O}(K \cdot \text{depth})$.
3. **XGBoost**: Fit $\mathcal{O}(K \cdot d \cdot N \log N \cdot D)$, Predict $\mathcal{O}(K \cdot d)$.
4. **SVM (Kernel RBF)**: Fit $\mathcal{O}(N^2 \cdot D)$ s.d. $\mathcal{O}(N^3 \cdot D)$, Predict $\mathcal{O}(N_{sv} \cdot D)$.
5. **MLP (Dense NN)**: Fit $\mathcal{O}(E \cdot N \cdot \sum M_l M_{l+1})$, Predict $\mathcal{O}(\sum M_l M_{l+1})$.

### 10.2. Pembahasan Empiris Grafik
- **Skenario A & B**: Gaussian Naive Bayes menunjukkan garis datar (linier sempurna). SVM dan MLP memiliki tingkat kecuraman paling tinggi seiring penambahan sampel $N$.
- **Skenario C**: MLP dan GNB sangat ideal untuk *single sample inference* (< 0.5 ms). Random Forest dan XGBoost membutuhkan waktu lebih tinggi saat melakukan traversal pada 100 pohon decision tree.
- **Skenario D**: Penambahan *Gaussian Noise* meningkatkan waktu pelatihan pada SVM dan MLP karena terganggunya margin hyperplane dan terhambatnya konvergensi *gradient descent*.

### 10.3. Trade-off Efisiensi vs Efektivitas
- **Accuracy & F1 Supremacy**: **Random Forest** dan **XGBoost** meraih akurasi tertinggi (~76% - 78%).
- **Speed Supremacy**: **Gaussian Naive Bayes** mencatatkan waktu *fit* dan *predict* paling minim (< 1 ms).
- **Kesimpulan Praktis**: Pada aplikasi berorientasi kecepatan respon (*ultra-low latency*), GNB dipilih. Pada aplikasi medis yang memprioritaskan skor F1/Akurasi, Random Forest dan XGBoost merupakan pilihan utama.
"""))

nb['cells'] = cells

with open('benchmarking_ml_algorithms.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("File benchmarking_ml_algorithms.ipynb berhasil dibuat!")
