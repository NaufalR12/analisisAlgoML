import time
import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as px_go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# ---------------------------------------------------------
# Sidebar Controls & File Upload
# ---------------------------------------------------------
st.sidebar.title("⚙️ Kontrol Eksperimen")
st.sidebar.subheader("📂 Upload Dataset CSV")

uploaded_file = st.sidebar.file_uploader(
    "Unggah File CSV:",
    type=["csv"],
    help="Upload file dataset bertipe .csv dari komputer Anda."
)

if uploaded_file is None:
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">⚡ Time Complexity Benchmarking & Algorithm Analysis Dashboard</div>
        <div class="header-subtitle">Analisis Pertumbuhan Waktu Eksekusi (Training & Prediction) dan Metrik Efektivitas Algoritma Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👋 **Selamat Datang!** Silakan upload file dataset bertipe **.csv** di sidebar sebelah kiri untuk memulai eksperimen.")
    st.warning("⚠️ **Catatan**: Aplikasi **tidak akan langsung berjalan** secara otomatis. Setelah mengunggah file CSV dan memilih parameter, silakan klik tombol **🚀 Jalankan Benchmarking** di sidebar.")
    st.stop()

# Read uploaded CSV dataframe
try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Gagal membaca file CSV: {e}")
    st.stop()

if df_raw.empty or len(df_raw.columns) < 2:
    st.error("❌ File CSV harus memiliki minimal 2 kolom (fitur dan target) dan tidak boleh kosong.")
    st.stop()

# Target column selection
target_col = st.sidebar.selectbox(
    "Pilih Kolom Target (Label/Class):",
    options=df_raw.columns.tolist(),
    index=len(df_raw.columns) - 1,
    help="Pilih kolom yang menjadi variabel dependen / target klasifikasi."
)

st.sidebar.markdown("---")
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

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Jalankan Benchmarking", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Dynamic Data Preprocessing Function
# ---------------------------------------------------------
def preprocess_uploaded_data(df_input, target_column):
    df = df_input.copy()
    
    # Drop rows where target is missing
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    
    # Encode target vector y
    y_raw = df[target_column]
    if y_raw.dtype == 'object' or not np.issubdtype(y_raw.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
    X_df = df.drop(columns=[target_column])
    feature_names = X_df.columns.tolist()
    
    # Process feature columns
    X_processed = pd.DataFrame()
    for col in feature_names:
        if X_df[col].dtype == 'object' or not np.issubdtype(X_df[col].dtype, np.number):
            X_processed[col] = pd.factorize(X_df[col])[0]
        else:
            col_series = X_df[col].copy()
            if col_series.isnull().any():
                col_series = col_series.fillna(col_series.median())
            X_processed[col] = col_series
            
    df_clean = X_processed.copy()
    df_clean[target_column] = y
    
    X = X_processed.values
    
    # Train / Test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return df_input, df_clean, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_column

df_raw, df_clean, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_col = preprocess_uploaded_data(df_raw, target_col)

# ---------------------------------------------------------
# Helper function to instantiate selected ML models
# ---------------------------------------------------------
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
# Benchmark Engine Function
# ---------------------------------------------------------
def run_all_benchmarks(model_keys, sample_pcts, feature_pcts, noise_level, X_tr, X_te, y_tr, y_te, progress_callback=None):
    models = get_selected_models(model_keys)
    n_models = len(models)
    
    # Calculate total steps for percentage progress
    steps_a = len(sample_pcts) * n_models
    steps_b = len(feature_pcts) * n_models
    steps_c = n_models * 4 # 1 fit + 3 batch sizes
    steps_d = n_models * 2 # clean + noisy
    steps_summary = n_models
    
    total_steps = steps_a + steps_b + steps_c + steps_d + steps_summary
    completed_steps = 0
    
    def report_step(status_msg):
        nonlocal completed_steps
        completed_steps += 1
        if progress_callback and total_steps > 0:
            pct = int(min(100, (completed_steps / total_steps) * 100))
            progress_callback(pct, status_msg)
    
    # --- Scenario A ---
    percentages_a = [p/100.0 for p in sorted(sample_pcts)]
    n_samples_list = [max(2, int(p * len(X_tr))) for p in percentages_a]
    results_a = {m: {'fit_time': [], 'predict_time': []} for m in models.keys()}
    
    for p, n_samp in zip(percentages_a, n_samples_list):
        np.random.seed(42)
        idx = np.random.choice(len(X_tr), size=n_samp, replace=False)
        X_sub, y_sub = X_tr[idx], y_tr[idx]
        
        models_inst = get_selected_models(model_keys)
        for name, model in models_inst.items():
            t0 = time.perf_counter()
            model.fit(X_sub, y_sub)
            t_fit = time.perf_counter() - t0
            
            t1 = time.perf_counter()
            _ = model.predict(X_te)
            t_pred = time.perf_counter() - t1
            
            results_a[name]['fit_time'].append(t_fit * 1000)
            results_a[name]['predict_time'].append(t_pred * 1000)
            report_step(f"Skenario A ({int(p*100)}% Sampel) - {name}")
            
    # --- Scenario B ---
    percentages_b = [p/100.0 for p in sorted(feature_pcts)]
    total_f = X_tr.shape[1]
    k_list = [max(1, min(int(round(p * total_f)), total_f)) for p in percentages_b]
    results_b = {m: {'fit_time': [], 'predict_time': []} for m in models.keys()}
    
    for p, k in zip(percentages_b, k_list):
        actual_k = min(k, total_f)
        selector = SelectKBest(score_func=f_classif, k=actual_k)
        try:
            X_tr_k = selector.fit_transform(X_tr, y_tr)
            X_te_k = selector.transform(X_te)
        except Exception:
            X_tr_k = X_tr
            X_te_k = X_te
        
        models_inst = get_selected_models(model_keys)
        for name, model in models_inst.items():
            t0 = time.perf_counter()
            model.fit(X_tr_k, y_tr)
            t_fit = time.perf_counter() - t0
            
            t1 = time.perf_counter()
            _ = model.predict(X_te_k)
            t_pred = time.perf_counter() - t1
            
            results_b[name]['fit_time'].append(t_fit * 1000)
            results_b[name]['predict_time'].append(t_pred * 1000)
            report_step(f"Skenario B ({int(p*100)}% Fitur) - {name}")
            
    # --- Scenario C ---
    test_len = len(X_te)
    b_single = 1
    b_batch = min(32, test_len)
    b_bulk = test_len
    batch_sizes = [b_single, b_batch, b_bulk]
    batch_labels = ['1 (Single)', f'{b_batch} (Batch)', f'{b_bulk} (Bulk)']
    results_c = {m: [] for m in models.keys()}
    
    models_inst = get_selected_models(model_keys)
    for name, m in models_inst.items():
        m.fit(X_tr, y_tr)
        report_step(f"Skenario C (Initial Fit) - {name}")
        
    for bs, blabel in zip(batch_sizes, batch_labels):
        X_b = X_te[:bs]
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
            report_step(f"Skenario C (Batch {blabel}) - {name}")
            
    # --- Scenario D ---
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, size=X_tr.shape)
    X_noisy = X_tr + noise
    results_d = {m: {'clean_fit': 0.0, 'noisy_fit': 0.0} for m in models.keys()}
    
    models_clean = get_selected_models(model_keys)
    models_noisy = get_selected_models(model_keys)
    
    for name in models.keys():
        t0 = time.perf_counter()
        models_clean[name].fit(X_tr, y_tr)
        results_d[name]['clean_fit'] = (time.perf_counter() - t0) * 1000
        report_step(f"Skenario D (Data Clean) - {name}")
        
        t0 = time.perf_counter()
        models_noisy[name].fit(X_noisy, y_tr)
        results_d[name]['noisy_fit'] = (time.perf_counter() - t0) * 1000
        report_step(f"Skenario D (Data Noisy) - {name}")
        
    # --- Summary Metrics ---
    summary = []
    models_inst = get_selected_models(model_keys)
    is_multiclass = len(np.unique(y_tr)) > 2
    avg_setting = 'weighted' if is_multiclass else 'binary'
    
    for name, model in models_inst.items():
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_t = (time.perf_counter() - t0) * 1000
        
        t1 = time.perf_counter()
        y_pred = model.predict(X_te)
        pred_t = (time.perf_counter() - t1) * 1000
        
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, average=avg_setting, zero_division=0)
        rec = recall_score(y_te, y_pred, average=avg_setting, zero_division=0)
        f1 = f1_score(y_te, y_pred, average=avg_setting, zero_division=0)
        
        summary.append({
            'Algoritma': name,
            'Akurasi': round(acc, 4),
            'Presisi': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'Fit Time (ms)': round(fit_t, 2),
            'Predict Time (ms)': round(pred_t, 2)
        })
        report_step(f"Evaluasi Akhir & Summary - {name}")
        
    return {
        'scen_a': (sample_pcts, results_a),
        'scen_b': (feature_pcts, results_b),
        'scen_c': (batch_labels, results_c),
        'scen_d': results_d,
        'summary': pd.DataFrame(summary)
    }

# ---------------------------------------------------------
# Execution State Control ("Tidak Langsung Jalan")
# ---------------------------------------------------------
if 'bench_data' not in st.session_state:
    st.session_state['bench_data'] = None

if run_button:
    if not selected_models:
        st.sidebar.error("⚠️ Pilih minimal 1 algoritma ML di sidebar!")
    elif not sample_steps:
        st.sidebar.error("⚠️ Pilih minimal 1 persentase sampel di Skenario A!")
    elif not feature_steps:
        st.sidebar.error("⚠️ Pilih minimal 1 persentase fitur di Skenario B!")
    else:
        progress_bar = st.progress(0)
        status_box = st.empty()
        
        def update_progress(pct, status_msg):
            progress_bar.progress(pct)
            status_box.markdown(f"⏳ **Menjalankan Benchmarking: {pct}%** — *{status_msg}*")
            
        st.session_state['bench_data'] = run_all_benchmarks(
            selected_models, sample_steps, feature_steps, noise_std,
            X_train_scaled, X_test_scaled, y_train, y_test,
            progress_callback=update_progress
        )
        
        progress_bar.progress(100)
        status_box.success("🎉 **Benchmarking 100% Selesai!** Hasil analisis dan grafik siap dilihat.")

# Color Palette for Plotly Charts
COLOR_MAP = {
    'SVM': '#38bdf8',
    'MLP (Dense NN)': '#fb923c',
    'Random Forest': '#4ade80',
    'XGBoost': '#f43f5e',
    'Gaussian Naive Bayes': '#a855f7'
}

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown(f"""
<div class="header-banner">
    <div class="header-title">⚡ Time Complexity Benchmarking & Algorithm Analysis Dashboard</div>
    <div class="header-subtitle">Analisis Pertumbuhan Waktu Eksekusi (Training & Prediction) dan Metrik Efektivitas Algoritma Machine Learning pada Dataset Custom (File: <b>{uploaded_file.name}</b>)</div>
</div>
""", unsafe_allow_html=True)

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
    st.subheader(f"📌 Overview Dataset: {uploaded_file.name}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_raw)}</div><div class="metric-label">Total Baris Data</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(feature_names)}</div><div class="metric-label">Fitur Independen</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(X_train_scaled)} / {len(X_test_scaled)}</div><div class="metric-label">Pembagian Train / Test</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(np.unique(y_train))}</div><div class="metric-label">Jumlah Kelas Target ({target_col})</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.write("##### Sample Data Setelah Imputasi & Preprocessing")
        st.dataframe(df_clean.head(8), use_container_width=True)
    with col_b:
        st.write(f"##### Distribusi Target ({target_col})")
        df_target_counts = pd.DataFrame(df_clean[target_col].value_counts()).reset_index()
        df_target_counts.columns = ['Class', 'Count']
        fig_pie = px.pie(
            df_target_counts, values='Count', names='Class',
            title=f"Distribusi Kelas Target ({target_col})",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=280)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("---")
    st.write("##### Penanganan Missing Values & Nilai 0 pada Kolom Fitur")
    null_summary = []
    for c in feature_names:
        if np.issubdtype(df_raw[c].dtype, np.number):
            raw_zeros = (df_raw[c] == 0).sum()
            raw_nulls = df_raw[c].isnull().sum()
            med_val = df_clean[c].median()
            null_summary.append({
                'Fitur': c, 
                'Jumlah Missing (NaN)': raw_nulls,
                'Jumlah Nilai 0': raw_zeros, 
                'Nilai Imputasi Median': round(med_val, 2)
            })
    if null_summary:
        st.table(pd.DataFrame(null_summary))
    else:
        st.info("Seluruh fitur kategorikal di-encode secara otomatis.")

# =========================================================
# TAB 2: SCENARIOS BENCHMARKING (A, B, C, D)
# =========================================================
with tab2:
    if st.session_state['bench_data'] is None:
        st.info("👈 **Benchmarking Belum Dijalankan.**")
        st.warning("Silakan sesuaikan parameter di sidebar dan klik **🚀 Jalankan Benchmarking** untuk memulai analisis dan meng-generate grafik.")
    else:
        bench_data = st.session_state['bench_data']
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
            st.caption("Membandingkan latency prediksi Single Sample (1), Small Batch, dan Bulk Test Set.")
            
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
    if st.session_state['bench_data'] is None:
        st.info("👈 **Benchmarking Belum Dijalankan.**")
        st.warning("Silakan sesuaikan parameter di sidebar dan klik **🚀 Jalankan Benchmarking** untuk melihat matriks evaluasi efisiensi vs efektivitas.")
    else:
        st.subheader("秤 Efficiency vs Effectiveness Trade-Off Matrix")
        bench_data = st.session_state['bench_data']
        df_summary = bench_data['summary']
        
        st.write("##### Tabel Ringkasan Evaluasi Data Latih & Data Uji")
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
            - **Ultra-Fast Baseline**: **Gaussian Naive Bayes** mencatatkan *fit time* amat cepat, sangat cocok untuk *streaming data* / *real-time processing*.
            - **Optimal Accuracy**: **Random Forest** & **XGBoost** umumnya mendominasi skor F1 dan Akurasi pada data tabular, dengan *latency* komputasi moderat.
            - **Heavy Optimization**: **MLP** & **SVM** membutuhkan biaya pelatihan lebih tinggi seiring peningkatan kompleksitas data.
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
