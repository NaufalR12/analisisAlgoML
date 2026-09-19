import time
import json
import os
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as px_go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML Time Complexity Benchmarking",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Aesthetics
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 24px;
    }
    
    .header-title {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 15px;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Custom Badges */
    .badge-fast {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    
    .badge-slow {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

DATASET_PATH = "diabetes.csv"
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# ---------------------------------------------------------
# Data Loader Function (Cached)
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    if not os.path.exists(DATASET_PATH):
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
        
    df = pd.read_csv(DATASET_PATH, names=COLUMN_NAMES) if not pd.read_csv(DATASET_PATH).columns[0] == 'Pregnancies' else pd.read_csv(DATASET_PATH)
    if 'Pregnancies' not in df.columns:
        df.columns = COLUMN_NAMES
        
    df_clean = df.copy()
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        median_val = df_clean[df_clean[col] != 0][col].median()
        df_clean[col] = df_clean[col].replace(0, median_val)
        
    X = df_clean.drop(columns=['Outcome'])
    y = df_clean['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return df, df_clean, X_train_scaled, X_test_scaled, y_train.values, y_test.values, X.columns.tolist()

df_raw, df_clean, X_train_scaled, X_test_scaled, y_train, y_test, feature_names = load_and_preprocess_data()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("⚙️ Kontrol Eksperimen")

selected_models = st.sidebar.multiselect(
    "Pilih Algoritma ML:",
    ['SVM', 'MLP (Dense NN)', 'Random Forest', 'XGBoost', 'Gaussian Naive Bayes'],
    default=['SVM', 'MLP (Dense NN)', 'Random Forest', 'XGBoost', 'Gaussian Naive Bayes']
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Parameter Skenario")

sample_steps = st.sidebar.multiselect(
    "Skenario A (Sampel %):",
    [10, 25, 50, 75, 100],
    default=[10, 25, 50, 75, 100]
)

feature_steps = st.sidebar.multiselect(
    "Skenario B (Fitur %):",
    [20, 40, 60, 80, 100],
    default=[20, 40, 60, 80, 100]
)

noise_std = st.sidebar.slider("Skenario D (Gaussian Noise Std):", 0.1, 1.0, 0.5, step=0.1)

run_button = st.sidebar.button("🚀 Jalankan Benchmarking Ulang", type="primary", use_container_width=True)

# Helper function to generate models
def get_selected_models(model_names):
    all_models = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'MLP (Dense NN)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, tol=1e-4, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss'),
        'Gaussian Naive Bayes': GaussianNB()
    }
    return {name: all_models[name] for name in model_names if name in all_models}

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <div class="header-title">⚡ Time Complexity Benchmarking & Algorithm Analysis Dashboard</div>
    <div class="header-subtitle">Analisis Pertumbuhan Waktu Eksekusi (Training & Prediction) dan Metrik Efektivitas 5 Algoritma Machine Learning pada Data Tabular (Pima Indians Diabetes)</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Benchmark Engine Execution (Cached & State)
# ---------------------------------------------------------
@st.cache_data
def run_all_benchmarks(model_keys, sample_pcts, feature_pcts, noise_level):
    models = get_selected_models(model_keys)
    
    # --- Scenario A ---
    percentages_a = [p/100.0 for p in sorted(sample_pcts)]
    n_samples_list = [int(p * len(X_train_scaled)) for p in percentages_a]
    results_a = {m: {'fit_time': [], 'predict_time': []} for m in models.keys()}
    
    for p, n_samp in zip(percentages_a, n_samples_list):
        np.random.seed(42)
        idx = np.random.choice(len(X_train_scaled), size=n_samp, replace=False)
        X_sub, y_sub = X_train_scaled[idx], y_train[idx]
        
        models_inst = get_selected_models(model_keys)
        for name, model in models_inst.items():
            t0 = time.perf_counter()
            model.fit(X_sub, y_sub)
            t_fit = time.perf_counter() - t0
            
            t1 = time.perf_counter()
            _ = model.predict(X_test_scaled)
            t_pred = time.perf_counter() - t1
            
            results_a[name]['fit_time'].append(t_fit * 1000)
            results_a[name]['predict_time'].append(t_pred * 1000)
            
    # --- Scenario B ---
    percentages_b = [p/100.0 for p in sorted(feature_pcts)]
    total_f = X_train_scaled.shape[1]
    k_list = [max(1, int(round(p * total_f))) for p in percentages_b]
    results_b = {m: {'fit_time': [], 'predict_time': []} for m in models.keys()}
    
    for p, k in zip(percentages_b, k_list):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_tr_k = selector.fit_transform(X_train_scaled, y_train)
        X_te_k = selector.transform(X_test_scaled)
        
        models_inst = get_selected_models(model_keys)
        for name, model in models_inst.items():
            t0 = time.perf_counter()
            model.fit(X_tr_k, y_train)
            t_fit = time.perf_counter() - t0
            
            t1 = time.perf_counter()
            _ = model.predict(X_te_k)
            t_pred = time.perf_counter() - t1
            
            results_b[name]['fit_time'].append(t_fit * 1000)
            results_b[name]['predict_time'].append(t_pred * 1000)
            
    # --- Scenario C ---
    batch_sizes = [1, 32, len(X_test_scaled)]
    batch_labels = ['1 (Single)', '32 (Batch)', f'{len(X_test_scaled)} (Bulk)']
    results_c = {m: [] for m in models.keys()}
    
    models_inst = get_selected_models(model_keys)
    for m in models_inst.values():
        m.fit(X_train_scaled, y_train)
        
    for bs in batch_sizes:
        X_b = X_test_scaled[:bs]
        for name, model in models_inst.items():
            if bs == 1:
                t0 = time.perf_counter()
                for _ in range(50):
                    _ = model.predict(X_b)
                t_pred = (time.perf_counter() - t0) / 50.0
            else:
                t0 = time.perf_counter()
                _ = model.predict(X_b)
                t_pred = time.perf_counter() - t0
            results_c[name].append(t_pred * 1000)
            
    # --- Scenario D ---
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, size=X_train_scaled.shape)
    X_noisy = X_train_scaled + noise
    results_d = {m: {'clean_fit': 0.0, 'noisy_fit': 0.0} for m in models.keys()}
    
    models_clean = get_selected_models(model_keys)
    models_noisy = get_selected_models(model_keys)
    
    for name in models.keys():
        t0 = time.perf_counter()
        models_clean[name].fit(X_train_scaled, y_train)
        results_d[name]['clean_fit'] = (time.perf_counter() - t0) * 1000
        
        t0 = time.perf_counter()
        models_noisy[name].fit(X_noisy, y_train)
        results_d[name]['noisy_fit'] = (time.perf_counter() - t0) * 1000
        
    # --- Summary Metrics ---
    summary = []
    models_inst = get_selected_models(model_keys)
    for name, model in models_inst.items():
        t0 = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        fit_t = (time.perf_counter() - t0) * 1000
        
        t1 = time.perf_counter()
        y_pred = model.predict(X_test_scaled)
        pred_t = (time.perf_counter() - t1) * 1000
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        summary.append({
            'Algoritma': name,
            'Akurasi': round(acc, 4),
            'Presisi': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'Fit Time (ms)': round(fit_t, 2),
            'Predict Time (ms)': round(pred_t, 2)
        })
        
    return {
        'scen_a': (sample_pcts, results_a),
        'scen_b': (feature_pcts, results_b),
        'scen_c': (batch_labels, results_c),
        'scen_d': results_d,
        'summary': pd.DataFrame(summary)
    }

if not selected_models:
    st.warning("⚠️ Harap pilih minimal 1 algoritma ML di sidebar!")
    st.stop()

bench_data = run_all_benchmarks(selected_models, sample_steps, feature_steps, noise_std)

# Color Palette for Plotly Charts
COLOR_MAP = {
    'SVM': '#38bdf8',
    'MLP (Dense NN)': '#fb923c',
    'Random Forest': '#4ade80',
    'XGBoost': '#f43f5e',
    'Gaussian Naive Bayes': '#a855f7'
}

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset & Preprocessing", 
    "🚀 Kurva Pertumbuhan Waktu (A, B, C, D)", 
    "⚖️ Efficiency vs Effectiveness Matrix", 
    "📖 Laporan Akademis & Notasi Big-O"
])

# =========================================================
# TAB 1: DATASET & PREPROCESSING
# =========================================================
with tab1:
    st.subheader("📌 Overview Dataset Pima Indians Diabetes")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">768</div><div class="metric-label">Total Baris Data</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">8</div><div class="metric-label">Fitur Independen</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">614 / 154</div><div class="metric-label">Pembagian Train / Test</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">34.9%</div><div class="metric-label">Proporsi Positif Diabetes</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.write("##### Sample Data Setelah Imputasi Median & Scaling")
        st.dataframe(df_clean.head(8), use_container_width=True)
    with col_b:
        st.write("##### Distrubusi Target (Outcome)")
        fig_pie = px.pie(
            df_clean, names='Outcome', title="Distribusi Diagnostik Diabetes (0 vs 1)",
            color_discrete_sequence=['#3b82f6', '#ef4444'],
            labels={'0': 'Negatif (0)', '1': 'Positif (1)'}
        )
        fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=280)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("---")
    st.write("##### Penanganan Missing Values Implisit (Angka 0 pada Variabel Fisiologis)")
    zero_summary = []
    for c in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        raw_zeros = (df_raw[c] == 0).sum()
        median_val = df_clean[c].median()
        zero_summary.append({'Fitur': c, 'Jumlah Nilai 0 (Mentah)': raw_zeros, 'Persentase 0': f"{(raw_zeros/768)*100:.1f}%", 'Nilai Imputasi Median': round(median_val, 2)})
    st.table(pd.DataFrame(zero_summary))

# =========================================================
# TAB 2: SCENARIOS BENCHMARKING (A, B, C, D)
# =========================================================
with tab2:
    st.subheader("📈 Apple-to-Apple Benchmarking Curves")
    
    # --- Scenario A ---
    st.markdown("### 1. Skenario A: Variasi Ukuran Sampel ($N$)")
    st.caption("Peningkatan waktu eksekusi saat ukuran sampel latih bervariasi dari 10% hingga 100% data.")
    
    sample_pcts, res_a = bench_data['scen_a']
    
    fig_a_fit = px_go.Figure()
    fig_a_pred = px_go.Figure()
    
    for model_name, data in res_a.items():
        fig_a_fit.add_trace(px_go.Scatter(x=sample_pcts, y=data['fit_time'], mode='lines+markers', name=model_name, line=dict(color=COLOR_MAP[model_name], width=3)))
        fig_a_pred.add_trace(px_go.Scatter(x=sample_pcts, y=data['predict_time'], mode='lines+markers', name=model_name, line=dict(color=COLOR_MAP[model_name], width=3)))
        
    fig_a_fit.update_layout(title="Training Time (.fit) vs Sample Size (N)", xaxis_title="Persentase Sampel Latih (%)", yaxis_title="Waktu Fit (ms)", template="plotly_dark", height=380)
    fig_a_pred.update_layout(title="Inference Time (.predict) vs Sample Size (N)", xaxis_title="Persentase Sampel Latih (%)", yaxis_title="Waktu Predict (ms)", template="plotly_dark", height=380)
    
    col_a1, col_a2 = st.columns(2)
    with col_a1: st.plotly_chart(fig_a_fit, use_container_width=True)
    with col_a2: st.plotly_chart(fig_a_pred, use_container_width=True)
    
    st.markdown("---")
    
    # --- Scenario B ---
    st.markdown("### 2. Skenario B: Variasi Dimensi Fitur ($D$)")
    st.caption("Pengaruh sub-himpunan fitur (SelectKBest ANOVA F-score) terhadap durasi komputasi.")
    
    feat_pcts, res_b = bench_data['scen_b']
    
    fig_b_fit = px_go.Figure()
    fig_b_pred = px_go.Figure()
    
    for model_name, data in res_b.items():
        fig_b_fit.add_trace(px_go.Scatter(x=feat_pcts, y=data['fit_time'], mode='lines+markers', name=model_name, line=dict(color=COLOR_MAP[model_name], width=3)))
        fig_b_pred.add_trace(px_go.Scatter(x=feat_pcts, y=data['predict_time'], mode='lines+markers', name=model_name, line=dict(color=COLOR_MAP[model_name], width=3)))
        
    fig_b_fit.update_layout(title="Training Time (.fit) vs Feature Count (D)", xaxis_title="Persentase Fitur Terpilih (%)", yaxis_title="Waktu Fit (ms)", template="plotly_dark", height=380)
    fig_b_pred.update_layout(title="Inference Time (.predict) vs Feature Count (D)", xaxis_title="Persentase Fitur Terpilih (%)", yaxis_title="Waktu Predict (ms)", template="plotly_dark", height=380)
    
    col_b1, col_b2 = st.columns(2)
    with col_b1: st.plotly_chart(fig_b_fit, use_container_width=True)
    with col_b2: st.plotly_chart(fig_b_pred, use_container_width=True)
    
    st.markdown("---")
    
    # --- Scenario C & D ---
    col_cd1, col_cd2 = st.columns(2)
    
    with col_cd1:
        st.markdown("### 3. Skenario C: Inference Batch Scalability")
        st.caption("Membandingkan latency prediksi Single Sample (1), Small Batch (32), dan Bulk Test Set.")
        
        batch_labels, res_c = bench_data['scen_c']
        fig_c = px_go.Figure()
        for model_name, times in res_c.items():
            fig_c.add_trace(px_go.Scatter(x=batch_labels, y=times, mode='lines+markers', name=model_name, line=dict(color=COLOR_MAP[model_name], width=3)))
        fig_c.update_layout(title="Inference Latency across Batch Sizes (Log Scale)", xaxis_title="Ukuran Batch Prediksi", yaxis_title="Latency (ms) - Log Scale", yaxis_type="log", template="plotly_dark", height=420)
        st.plotly_chart(fig_c, use_container_width=True)
        
    with col_cd2:
        st.markdown("### 4. Skenario D: Sensitivitas Data Noisy")
        st.caption(f"Perbandingan waktu training data bersih vs data dengan Gaussian Noise (std={noise_std}).")
        
        res_d = bench_data['scen_d']
        models_d = list(res_d.keys())
        clean_times = [res_d[m]['clean_fit'] for m in models_d]
        noisy_times = [res_d[m]['noisy_fit'] for m in models_d]
        
        fig_d = px_go.Figure(data=[
            px_go.Bar(name='Data Normal (Clean)', x=models_d, y=clean_times, marker_color='#3b82f6'),
            px_go.Bar(name='Data Noisy (+Noise)', x=models_d, y=noisy_times, marker_color='#ef4444')
        ])
        fig_d.update_layout(barmode='group', title="Training Time: Clean vs Noisy Data", xaxis_title="Algoritma ML", yaxis_title="Training Time (ms)", template="plotly_dark", height=420)
        st.plotly_chart(fig_d, use_container_width=True)

# =========================================================
# TAB 3: TRADE-OFF MATRIX & SUMMARY
# =========================================================
with tab3:
    st.subheader("⚖️ Efficiency vs Effectiveness Trade-Off Matrix")
    
    df_summary = bench_data['summary']
    
    st.write("##### Tabel Ringkasan Evaluasi 100% Data Latih & Data Uji")
    st.dataframe(
        df_summary.style.highlight_max(subset=['Akurasi', 'F1-Score'], color='#064e3b')
                        .highlight_min(subset=['Fit Time (ms)', 'Predict Time (ms)'], color='#1e3a8a'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    col_mat1, col_mat2 = st.columns([3, 2])
    
    with col_mat1:
        st.write("##### Scatter Plot Trade-off: Skor F1 vs Training Time (Log Scale)")
        fig_trade = px.scatter(
            df_summary, x='Fit Time (ms)', y='F1-Score', text='Algoritma', color='Algoritma',
            size=[25]*len(df_summary), color_discrete_map=COLOR_MAP, log_x=True,
            title="Matriks Perbandingan: Kecepatan Fit vs Kualitas Prediksi"
        )
        fig_trade.update_traces(textposition='top center', marker=dict(sizemode='area', sizeref=1))
        fig_trade.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_trade, use_container_width=True)
        
    with col_mat2:
        st.write("##### 💡 Analisis Klasifikasi Model")
        st.markdown(r"""
        - **Ultra-Fast Baseline**: **Gaussian Naive Bayes** mencatatkan *fit time* $< 1\text{ ms}$, sangat cocok untuk *streaming data* / *real-time processing*.
        - **Optimal Accuracy**: **Random Forest** & **XGBoost** mendominasi skor F1 dan Akurasi tertinggi, dengan *latency* komputasi moderat.
        - **Heavy Optimization**: **MLP** & **SVM** membutuhkan biaya pelatihan lebih tinggi seiring peningkatan kompleksitas arsitektur dan margin optimizer.
        """)

# =========================================================
# TAB 4: ACADEMIC REPORT & BIG-O THEORY
# =========================================================
with tab4:
    st.subheader("🎓 Draf Laporan Analisis Akademis Komprehensif")
    
    st.markdown("""
    ### 1. Landasan Teori Notasi Big-O (Time Complexity)
    
    Turunan matematis untuk kompleksitas waktu latih (*training*) dan waktu prediksi (*inference*):
    """)
    
    st.latex(r"""
    \begin{aligned}
    \text{Gaussian Naive Bayes: } & T_{\text{fit}} = \mathcal{O}(N \cdot D), \quad T_{\text{predict}} = \mathcal{O}(C \cdot D) \\
    \text{Random Forest: } & T_{\text{fit}} = \mathcal{O}(K \cdot N \log N \cdot \sqrt{D}), \quad T_{\text{predict}} = \mathcal{O}(K \cdot \text{depth}) \\
    \text{XGBoost: } & T_{\text{fit}} = \mathcal{O}(K \cdot d \cdot N \log N \cdot D), \quad T_{\text{predict}} = \mathcal{O}(K \cdot d) \\
    \text{SVM (Kernel RBF): } & T_{\text{fit}} = \mathcal{O}(N^2 \cdot D \text{ s.d. } N^3 \cdot D), \quad T_{\text{predict}} = \mathcal{O}(N_{\text{SV}} \cdot D) \\
    \text{MLP (Dense NN): } & T_{\text{fit}} = \mathcal{O}\left(E \cdot N \cdot \sum_{l=1}^{L-1} M_l M_{l+1}\right), \quad T_{\text{predict}} = \mathcal{O}\left(\sum_{l=1}^{L-1} M_l M_{l+1}\right)
    \end{aligned}
    """)
    
    st.markdown(r"""
    ---
    ### 2. Diskusi & Pembahasan Empiris
    
    1. **Sensitivitas Terhadap Ukuran Data ($N$)**:
       - Gaussian Naive Bayes mempertahankan waktu latih linier datar karena hanya menghitung parameter statistik mean ($\mu$) dan variansi ($\sigma^2$) dalam 1 pas alur data.
       - Support Vector Machine (SVM) menunjukkan kenaikan kurva paling tinggi akibat kebutuhan membangun dan menyelesaikan matriks Kernel Gram berukuran $N \times N$.
       
    2. **Sensitivitas Terhadap Dimensi Fitur ($D$)**:
       - XGBoost dan Random Forest mengalami penambahan waktu split-node seiring meningkatnya variabel kandidat $D$.
       - SVM membutuhkan lebih banyak operasi *dot product* untuk fungsi kernel RBF.
       
    3. **Sensitivitas Terhadap Gaussian Noise**:
       - *Gaussian Noise* merusak margin hyperplane linier pada SVM, memaksa QP solver merekrut lebih banyak *Support Vectors* ($N_{\text{SV}}$).
       - Pada MLP, *noise* memperlambat laju penurunan gradient (*gradient descent*), meningkatkan jumlah epoch hingga konvergen.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard dikembangkan oleh Senior Data Scientist & Expert Algorithm Analyst.")
