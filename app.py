"""
BigDataPlatform v1.1
Bank faoliyatini tahlil qilish uchun katta ma'lumotlarni qayta ishlash platformasi
Muallif: Turabova Shakhnoza
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import os
import time
import io
import json

from modules.data_generator import generate_all_data
from modules.data_cleaner import clean_dataset
from modules.data_validator import validate_dataset
from modules.data_transformer import calculate_kpis
from modules.data_mart import build_marts, export_to_sqlite
from modules.data_loader import (
    load_file, load_file_in_chunks, detect_file_type,
    auto_detect_table_type, get_memory_usage, optimize_dtypes,
)
from modules.advanced_analytics import comprehensive_analysis
from modules.nosql_support import (
    load_json_nosql, load_bson, flatten_nested_json,
    nosql_to_tabular, detect_nosql_structure, get_nosql_sample_data,
)

st.set_page_config(
    page_title="BigDataPlatform — Bank ma'lumotlarini tahlil platformasi",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f172a 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); border-right: 1px solid #334155; }
    .platform-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
        padding: 2.5rem 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(96, 165, 250, 0.3);
    }
    .platform-header h1 { margin: 0; font-size: 3rem; font-weight: 700; letter-spacing: -1px; }
    .platform-header .subtitle { margin: 0.75rem 0 0 0; font-size: 1.1rem; opacity: 0.95; font-weight: 300; }
    .platform-header .author { margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2); font-size: 0.9rem; opacity: 0.85; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); height: 100%;
    }
    .metric-card h4 { margin: 0 0 0.5rem 0; color: #60a5fa; font-size: 1rem; }
    .metric-card p { margin: 0; color: #cbd5e1; font-size: 0.9rem; }
    .stage-badge { display: inline-block; padding: 0.4rem 1rem;
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        color: white; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .new-badge { display: inline-block; padding: 0.2rem 0.6rem; margin-left: 0.5rem;
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white; border-radius: 12px; font-size: 0.7rem; font-weight: 700; }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white; border: none; border-radius: 8px; padding: 0.6rem 1.5rem;
        font-weight: 600; transition: all 0.3s; box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-2px); box-shadow: 0 6px 15px rgba(59, 130, 246, 0.4);
    }
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #60a5fa; }
    [data-testid="stMetricLabel"] { color: #cbd5e1; }
    h1, h2, h3 { color: #f1f5f9; }
    .streamlit-expanderHeader { background: #1e293b; border-radius: 8px; }
    .status-line { padding: 0.5rem 0; color: #cbd5e1; font-size: 0.95rem; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Session state
for key in ['raw_data', 'clean_data', 'clean_reports', 'validation_results',
            'kpis', 'marts', 'analytics', 'data_source', 'nosql_data']:
    if key not in st.session_state:
        st.session_state[key] = None


# ============= FIX: Fayl nomi bo'yicha tur aniqlash =============
def detect_type_by_filename(filename: str) -> str | None:
    """Fayl nomiga qarab ma'lumot turini aniqlaydi — ustunlardan oldin tekshiriladi."""
    fname = filename.lower()
    # Raqam prefikslari va bo'shliqlarni olib tashlash: "5_deposits.csv" → "deposits"
    fname_clean = fname.replace('-', '_').split('/')[-1]  # faqat fayl nomi
    
    deposit_keywords = ['deposit', 'depozit', 'saving', 'jamg', 'omonat']
    loan_keywords = ['loan', 'kredit', 'credit', 'qarz', 'ssuda']
    client_keywords = ['client', 'mijoz', 'customer', 'user', 'klient']
    account_keywords = ['account', 'hisob', 'счёт']
    transaction_keywords = ['transaction', 'operatsiya', 'payment', 'transfer', 'trx', 'tx_']
    
    for kw in deposit_keywords:
        if kw in fname_clean:
            return 'deposits'
    for kw in loan_keywords:
        if kw in fname_clean:
            return 'loans'
    for kw in client_keywords:
        if kw in fname_clean:
            return 'clients'
    for kw in account_keywords:
        if kw in fname_clean:
            return 'accounts'
    for kw in transaction_keywords:
        if kw in fname_clean:
            return 'transactions'
    return None  # Aniqlanmadi → ustunlarga qarab aniqlaydi


ARCHITECTURE_SVG = """
<svg viewBox="0 0 900 700" xmlns="http://www.w3.org/2000/svg" style="width: 100%; max-width: 900px;">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#1e40af;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#3b82f6;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#0891b2;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#06b6d4;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad3" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#10b981;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#34d399;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad4" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#f59e0b;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#fbbf24;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad5" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#8b5cf6;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#a78bfa;stop-opacity:1" />
    </linearGradient>
    <marker id="arrowblue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#60a5fa" />
    </marker>
  </defs>
  <rect x="50" y="30" width="800" height="100" rx="15" fill="url(#grad1)" opacity="0.9"/>
  <text x="450" y="60" text-anchor="middle" fill="white" font-size="18" font-weight="bold">1. MA'LUMOT MANBALARI</text>
  <rect x="80" y="75" width="130" height="45" rx="8" fill="white" opacity="0.95"/>
  <text x="145" y="93" text-anchor="middle" fill="#1e40af" font-size="11" font-weight="bold">SQL</text>
  <text x="145" y="108" text-anchor="middle" fill="#475569" font-size="10">CSV · Excel · DB</text>
  <rect x="230" y="75" width="130" height="45" rx="8" fill="white" opacity="0.95"/>
  <text x="295" y="93" text-anchor="middle" fill="#1e40af" font-size="11" font-weight="bold">NoSQL</text>
  <text x="295" y="108" text-anchor="middle" fill="#475569" font-size="10">JSON · BSON · Mongo</text>
  <rect x="380" y="75" width="130" height="45" rx="8" fill="white" opacity="0.95"/>
  <text x="445" y="93" text-anchor="middle" fill="#1e40af" font-size="11" font-weight="bold">Big Data</text>
  <text x="445" y="108" text-anchor="middle" fill="#475569" font-size="10">Parquet · Stata</text>
  <rect x="530" y="75" width="130" height="45" rx="8" fill="white" opacity="0.95"/>
  <text x="595" y="93" text-anchor="middle" fill="#1e40af" font-size="11" font-weight="bold">API</text>
  <text x="595" y="108" text-anchor="middle" fill="#475569" font-size="10">MB RUz · Open Data</text>
  <rect x="680" y="75" width="140" height="45" rx="8" fill="white" opacity="0.95"/>
  <text x="750" y="93" text-anchor="middle" fill="#1e40af" font-size="11" font-weight="bold">Real-time</text>
  <text x="750" y="108" text-anchor="middle" fill="#475569" font-size="10">Streaming data</text>
  <line x1="450" y1="130" x2="450" y2="155" stroke="#60a5fa" stroke-width="3" marker-end="url(#arrowblue)"/>
  <rect x="50" y="165" width="800" height="80" rx="15" fill="url(#grad2)" opacity="0.9"/>
  <text x="450" y="195" text-anchor="middle" fill="white" font-size="16" font-weight="bold">2. YUKLASH MODULI</text>
  <text x="450" y="220" text-anchor="middle" fill="white" font-size="12">Avtomatik aniqlash · Chunked reading · Multi-format</text>
  <line x1="450" y1="245" x2="450" y2="270" stroke="#60a5fa" stroke-width="3" marker-end="url(#arrowblue)"/>
  <rect x="50" y="280" width="385" height="120" rx="15" fill="url(#grad3)" opacity="0.9"/>
  <text x="242" y="310" text-anchor="middle" fill="white" font-size="16" font-weight="bold">3. TOZALASH</text>
  <text x="242" y="335" text-anchor="middle" fill="white" font-size="11">Dublikatlar · NULL</text>
  <text x="242" y="355" text-anchor="middle" fill="white" font-size="11">Anomaliyalar · Normalizatsiya</text>
  <rect x="465" y="280" width="385" height="120" rx="15" fill="url(#grad3)" opacity="0.9"/>
  <text x="657" y="310" text-anchor="middle" fill="white" font-size="16" font-weight="bold">4. VALIDATSIYA</text>
  <text x="657" y="335" text-anchor="middle" fill="white" font-size="11">Bank qoidalari (15+)</text>
  <text x="657" y="355" text-anchor="middle" fill="white" font-size="11">Ma'lumotlar yaxlitligi</text>
  <line x1="450" y1="400" x2="450" y2="425" stroke="#60a5fa" stroke-width="3" marker-end="url(#arrowblue)"/>
  <rect x="50" y="435" width="800" height="110" rx="15" fill="url(#grad4)" opacity="0.9"/>
  <text x="450" y="465" text-anchor="middle" fill="white" font-size="16" font-weight="bold">5. TAHLIL MODULI</text>
  <rect x="80" y="480" width="115" height="50" rx="8" fill="white" opacity="0.95"/>
  <text x="137" y="500" text-anchor="middle" fill="#92400e" font-size="11" font-weight="bold">KPI</text>
  <text x="137" y="515" text-anchor="middle" fill="#475569" font-size="9">NPL·ROA·ROE·CAR</text>
  <rect x="215" y="480" width="115" height="50" rx="8" fill="white" opacity="0.95"/>
  <text x="272" y="500" text-anchor="middle" fill="#92400e" font-size="11" font-weight="bold">ABC-tahlil</text>
  <text x="272" y="515" text-anchor="middle" fill="#475569" font-size="9">Mijozlar segmenti</text>
  <rect x="350" y="480" width="115" height="50" rx="8" fill="white" opacity="0.95"/>
  <text x="407" y="500" text-anchor="middle" fill="#92400e" font-size="11" font-weight="bold">Churn</text>
  <text x="407" y="515" text-anchor="middle" fill="#475569" font-size="9">Mijozlar oqimi</text>
  <rect x="485" y="480" width="115" height="50" rx="8" fill="white" opacity="0.95"/>
  <text x="542" y="500" text-anchor="middle" fill="#92400e" font-size="11" font-weight="bold">Risk</text>
  <text x="542" y="515" text-anchor="middle" fill="#475569" font-size="9">Konsentratsiya</text>
  <rect x="620" y="480" width="115" height="50" rx="8" fill="white" opacity="0.95"/>
  <text x="677" y="500" text-anchor="middle" fill="#92400e" font-size="11" font-weight="bold">Likvidlik</text>
  <text x="677" y="515" text-anchor="middle" fill="#475569" font-size="9">Muddat GAP</text>
  <line x1="450" y1="545" x2="450" y2="570" stroke="#60a5fa" stroke-width="3" marker-end="url(#arrowblue)"/>
  <rect x="50" y="580" width="800" height="95" rx="15" fill="url(#grad5)" opacity="0.9"/>
  <text x="450" y="610" text-anchor="middle" fill="white" font-size="16" font-weight="bold">6. MA'LUMOT VITRINALARI → BI TIZIMLARI</text>
  <rect x="100" y="625" width="160" height="35" rx="8" fill="white" opacity="0.95"/>
  <text x="180" y="648" text-anchor="middle" fill="#5b21b6" font-size="12" font-weight="bold">Star Schema (9 vitrina)</text>
  <text x="290" y="648" fill="white" font-size="16">→</text>
  <rect x="320" y="625" width="140" height="35" rx="8" fill="white" opacity="0.95"/>
  <text x="390" y="648" text-anchor="middle" fill="#5b21b6" font-size="12" font-weight="bold">Apache Superset</text>
  <rect x="475" y="625" width="120" height="35" rx="8" fill="white" opacity="0.95"/>
  <text x="535" y="648" text-anchor="middle" fill="#5b21b6" font-size="12" font-weight="bold">Power BI</text>
  <rect x="610" y="625" width="120" height="35" rx="8" fill="white" opacity="0.95"/>
  <text x="670" y="648" text-anchor="middle" fill="#5b21b6" font-size="12" font-weight="bold">Tableau</text>
  <rect x="745" y="625" width="80" height="35" rx="8" fill="white" opacity="0.95"/>
  <text x="785" y="648" text-anchor="middle" fill="#5b21b6" font-size="12" font-weight="bold">SQL</text>
</svg>
"""


# ============= SIDEBAR =============
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <div style='font-size: 3rem;'>🏦</div>
        <div style='font-size: 1.4rem; font-weight: 700; color: #60a5fa; margin-top: 0.5rem;'>BigDataPlatform</div>
        <div style='font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem;'>v1.1 · Bank tahlil platformasi</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio("**NAVIGATSIYA**", [
        "🏠 Platforma haqida",
        "⚡ Tezkor ishga tushirish",
        "📁 Fayllarni yuklash",
        "🌐 NoSQL ma'lumotlari",
        "1️⃣ Ma'lumot generatsiyasi",
        "2️⃣ Sifat tekshiruvi",
        "3️⃣ Ma'lumotlarni tozalash",
        "4️⃣ Validatsiya",
        "5️⃣ KPI hisoblash",
        "6️⃣ Kengaytirilgan tahlil",
        "7️⃣ Ma'lumot vitrinalari",
        "8️⃣ Superset ga eksport",
        "❓ Qanday foydalanish",
    ])
    
    st.markdown("---")
    st.markdown("### 📊 PIPELINE HOLATI")
    stages = [
        ("Ma'lumotlar yuklandi", st.session_state.raw_data),
        ("Tozalandi", st.session_state.clean_data),
        ("Validatsiya", st.session_state.validation_results),
        ("KPI", st.session_state.kpis),
        ("Tahlil", st.session_state.analytics),
        ("Vitrinalar", st.session_state.marts),
    ]
    for stage_name, stage_data in stages:
        icon = "✅" if stage_data is not None else "⚪"
        color = "#10b981" if stage_data is not None else "#64748b"
        st.markdown(f"<div class='status-line' style='color:{color}'>{icon} {stage_name}</div>",
                    unsafe_allow_html=True)
    progress = sum(1 for _, s in stages if s is not None) / len(stages)
    st.progress(progress)
    st.caption(f"Tayyorlik: {int(progress * 100)}%")
    
    if st.session_state.data_source:
        st.markdown(f"**Manba:** {st.session_state.data_source}")
    
    # FIX: Joriy ma'lumotlarni ko'rsatish
    if st.session_state.raw_data:
        st.markdown("---")
        st.markdown("### 📋 Joriy ma'lumotlar")
        for tname, tdf in st.session_state.raw_data.items():
            st.caption(f"• **{tname}**: {len(tdf):,} qator")
        if st.button("🗑️ Barcha ma'lumotlarni tozalash", use_container_width=True):
            for k in ['raw_data','clean_data','clean_reports','validation_results',
                      'kpis','analytics','marts','data_source','nosql_data']:
                st.session_state[k] = None
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.8rem; color: #94a3b8; text-align: center;'>
        <b>Muallif:</b><br>Turabova Shakhnoza<br><i>Xalq Bank · TDIU</i><br><br>
        <b>Mutaxassislik:</b><br>08.00.16 — Raqamli iqtisodiyot<br><br>© 2026
    </div>
    """, unsafe_allow_html=True)


# ============= ASOSIY SAHIFA =============
if page == "🏠 Platforma haqida":
    st.markdown("""
    <div class="platform-header">
        <h1>🏦 BigDataPlatform</h1>
        <div class="subtitle">Tijorat banki faoliyatini tahlil qilish uchun katta ma'lumotlarni qayta ishlash platformasi</div>
        <div class="author">Muallif: <b>Turabova Shakhnoza</b> · PhD tadqiqoti · Toshkent davlat iqtisodiyot universiteti</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Platforma imkoniyatlari")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="metric-card"><h4>📁 Har xil manbalar</h4>
        <p>CSV · Excel · JSON · Parquet · Stata · NoSQL (MongoDB/BSON)</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card"><h4>⚡ Big Data</h4>
        <p>Millionlab yozuvlarni chunk-lar bo'yicha qayta ishlash. Xotirani 70% gacha tejash</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card"><h4>🔬 8 turdagi tahlil</h4>
        <p>KPI · ABC · Churn · Risk · Likvidlik · Hududlar · Kanallar · Vaqt tahlili</p></div>""", unsafe_allow_html=True)
    
    st.markdown("## 🏗️ Platformaning to'liq arxitekturasi")
    st.markdown(f'<div style="background: white; padding: 1rem; border-radius: 15px; margin: 1rem 0;">{ARCHITECTURE_SVG}</div>',
                unsafe_allow_html=True)
    
    st.info("💡 **Boshlash uchun:** Chap menyuda **⚡ Tezkor ishga tushirish** yoki **📁 Fayllarni yuklash** sahifasiga o'ting")

# ============= TEZKOR ISHGA TUSHIRISH =============
elif page == "⚡ Tezkor ishga tushirish":
    st.markdown("""
    <div class="platform-header">
        <h1>⚡ Tezkor Pipeline ishga tushirish</h1>
        <div class="subtitle">Butun ma'lumotlarni qayta ishlash tsiklini avtomatik ishga tushirish</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, _, _ = st.columns([1, 1, 1])
    with col1:
        size = st.selectbox("Demo ma'lumot hajmi", ["Kichik (1K)", "O'rta (5K)", "Katta (20K)"], index=1)
    size_map = {
        "Kichik (1K)": (500, 800, 1000, 200, 300),
        "O'rta (5K)": (1000, 1500, 5000, 500, 800),
        "Katta (20K)": (3000, 4500, 20000, 1500, 2500),
    }
    n_c, n_a, n_t, n_l, n_d = size_map[size]
    
    if st.button("🚀 BUTUN PIPELINE NI AVTOMATIK ISHGA TUSHIRISH", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        
        status.info("⏳ **1/7:** Ma'lumot generatsiyasi...")
        progress.progress(10)
        data = generate_all_data(n_c, n_a, n_t, n_l, n_d, 0.15)
        st.session_state.raw_data = data
        st.session_state.data_source = "Generatsiya qilingan ma'lumotlar"
        total = sum(len(df) for df in data.values())
        status.success(f"✅ **1/7:** {total:,} ta yozuv yaratildi")
        progress.progress(20); time.sleep(0.3)
        
        status.info("⏳ **2/7:** Tozalash...")
        clean_data, reports = {}, {}
        for n, df in data.items():
            c, r = clean_dataset(df, n)
            clean_data[n], reports[n] = c, r
        st.session_state.clean_data = clean_data
        st.session_state.clean_reports = reports
        td = sum(r['duplicates_removed'] for r in reports.values())
        tn = sum(r['nulls_filled'] for r in reports.values())
        status.success(f"✅ **2/7:** {td:,} ta dublikat, {tn:,} ta NULL")
        progress.progress(35); time.sleep(0.3)
        
        status.info("⏳ **3/7:** Validatsiya...")
        v = validate_dataset(clean_data)
        st.session_state.validation_results = v
        p = sum(1 for c in v['checks'] if c['status'] == 'passed')
        status.success(f"✅ **3/7:** {p}/{len(v['checks'])} ta qoida o'tdi")
        progress.progress(50); time.sleep(0.3)
        
        status.info("⏳ **4/7:** KPI hisoblash...")
        k = calculate_kpis(clean_data)
        st.session_state.kpis = k
        status.success(f"✅ **4/7:** NPL={k['npl_ratio']:.2f}% · ROA={k['roa']:.2f}%")
        progress.progress(65); time.sleep(0.3)
        
        status.info("⏳ **5/7:** Kengaytirilgan tahlil...")
        a = comprehensive_analysis(clean_data)
        st.session_state.analytics = a
        status.success("✅ **5/7:** 8 turdagi tahlil bajarildi")
        progress.progress(80); time.sleep(0.3)
        
        status.info("⏳ **6/7:** Vitrinalar...")
        m = build_marts(clean_data)
        st.session_state.marts = m
        status.success(f"✅ **6/7:** {len(m)} ta vitrina qurildi")
        progress.progress(92); time.sleep(0.3)
        
        status.info("⏳ **7/7:** SQLite eksporti...")
        os.makedirs("data", exist_ok=True)
        export_to_sqlite(m, "data/bigdataplatform.db")
        sz = os.path.getsize("data/bigdataplatform.db") / 1024 / 1024
        status.success(f"✅ **7/7:** SQLite bazasi ({sz:.2f} MB)")
        progress.progress(100); time.sleep(0.4)
        status.empty(); progress.empty()
        st.success("### 🎉 Pipeline muvaffaqiyatli bajarildi!")
    
    if st.session_state.kpis is not None:
        k = st.session_state.kpis
        st.markdown("---")
        cols = st.columns(5)
        with cols[0]: st.metric("NPL", f"{k['npl_ratio']:.2f}%")
        with cols[1]: st.metric("LDR", f"{k['ldr']:.1f}%")
        with cols[2]: st.metric("ROA", f"{k['roa']:.2f}%")
        with cols[3]: st.metric("ROE", f"{k['roe']:.2f}%")
        with cols[4]: st.metric("CAR", f"{k['car']:.2f}%")

# ============= FAYLLARNI YUKLASH (FIXED) =============
elif page == "📁 Fayllarni yuklash":
    st.markdown("""
    <div class="platform-header">
        <h1>📁 Haqiqiy ma'lumotlarni yuklash</h1>
        <div class="subtitle">CSV · Excel · JSON · Parquet · Stata fayllari</div>
    </div>
    """, unsafe_allow_html=True)
    
    # FIX: Mavjud ma'lumotlar haqida ogohlantirish
    if st.session_state.raw_data:
        existing = list(st.session_state.raw_data.keys())
        st.warning(
            f"⚠️ **Sessiyada mavjud ma'lumotlar:** {', '.join(existing)}. "
            f"Yangi fayl yuklasangiz, ular **qo'shiladi** (almashtirilmaydi). "
            f"Tozalash uchun chap menyu pastidagi **🗑️ Barcha ma'lumotlarni tozalash** tugmasini bosing."
        )
    
    st.info("""
    **Qo'llanma:**
    1. Bank ma'lumotlari fayllarini yuklang
    2. Platforma fayl **nomi va ustunlari** bo'yicha turini avtomatik aniqlaydi
    3. Kerak bo'lsa qo'lda tur tanlang
    4. Ma'lumotlar avtomatik saqlanadi → **2️⃣ Sifat tekshiruvi** ga o'ting

    **Big Data:** 100 MB dan katta fayllar chunk-lar bo'yicha qayta ishlanadi.
    """)
    
    uploaded_files = st.file_uploader(
        "Fayllarni tanlang",
        type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'tsv', 'dta', 'txt'],
        accept_multiple_files=True)
    
    if uploaded_files:
        st.markdown(f"### 📋 Yuklandi: **{len(uploaded_files)}** ta fayl")
        loaded_data = {}
        
        for idx, uf in enumerate(uploaded_files):
            size_mb = uf.size / (1024 * 1024)
            ftype = detect_file_type(uf.name)
            is_big = size_mb > 100
            
            with st.expander(f"📄 **{uf.name}** · {size_mb:.2f} MB · {ftype}", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Hajm", f"{size_mb:.2f} MB")
                with c2: st.metric("Format", ftype)
                with c3: st.metric("Rejim", "🚀 Big Data" if is_big else "⚡ Oddiy")
                
                try:
                    with st.spinner(f"{uf.name} yuklanmoqda..."):
                        if is_big and ftype in ('CSV', 'TSV'):
                            df = load_file_in_chunks(uf, file_type=ftype, filename=uf.name, chunksize=100_000)
                        else:
                            df = load_file(uf, file_type=ftype, filename=uf.name)
                    
                    # FIX: Avval fayl nomiga qarab aniqlash, keyin ustunlarga
                    fname_detected = detect_type_by_filename(uf.name)
                    col_detected = auto_detect_table_type(df)
                    detected = fname_detected if fname_detected is not None else col_detected
                    
                    # FIX: Aniqlash manbasini ko'rsatish
                    if fname_detected is not None:
                        detect_source = f"fayl nomidan aniqlandi: **{uf.name}**"
                    else:
                        detect_source = f"ustunlar bo'yicha aniqlandi"
                    
                    types = ['clients', 'accounts', 'transactions', 'loans', 'deposits', 'unknown']
                    type_labels = {
                        'clients': 'Mijozlar (clients)',
                        'accounts': 'Hisoblar (accounts)',
                        'transactions': 'Operatsiyalar (transactions)',
                        'loans': 'Kreditlar (loans)',
                        'deposits': 'Depozitlar (deposits)',
                        'unknown': "Noma'lum"
                    }
                    default_idx = types.index(detected) if detected in types else 5
                    manual_type = st.selectbox(
                        f"Ma'lumot turi ({detect_source}: **{type_labels.get(detected, detected)}**)",
                        types, index=default_idx, key=f"type_{idx}",
                        format_func=lambda x: type_labels[x])
                    
                    mem = get_memory_usage(df)
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("Qatorlar", f"{len(df):,}")
                    with c2: st.metric("Ustunlar", f"{len(df.columns)}")
                    with c3: st.metric("Xotira", f"{mem:.2f} MB")
                    with c4: st.metric("NULL", f"{df.isna().sum().sum():,}")
                    
                    if is_big:
                        if st.checkbox("🚀 Optimallash", key=f"opt_{idx}"):
                            with st.spinner("..."):
                                df, info = optimize_dtypes(df)
                                st.success(f"✅ {info['original_mb']:.2f} → {info['new_mb']:.2f} MB")
                    
                    st.markdown("**Dastlabki ma'lumotlar:**")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    if manual_type != 'unknown':
                        loaded_data[manual_type] = df
                        if st.session_state.raw_data is None:
                            st.session_state.raw_data = {}
                        st.session_state.raw_data[manual_type] = df
                        st.session_state.data_source = "Yuklangan fayllar"
                        # FIX: Tozalash kerak bo'lgan qadamlarni reset qilish
                        for rk in ['clean_data', 'clean_reports', 'validation_results',
                                   'kpis', 'analytics', 'marts']:
                            st.session_state[rk] = None
                        st.success(f"✅ **{type_labels[manual_type]}** — saqlandi")
                
                except Exception as e:
                    st.error(f"❌ Xato: {str(e)}")
        
        if loaded_data:
            st.markdown("---")
            total_tables = len(st.session_state.raw_data) if st.session_state.raw_data else 0
            st.success(
                f"✅ **{len(loaded_data)} ta yangi jadval** saqlandi! "
                f"Sessiyada jami: **{total_tables} ta jadval**. "
                f"Endi **'2️⃣ Sifat tekshiruvi'** sahifasiga o'ting."
            )

# ============= NoSQL (FIXED) =============
elif page == "🌐 NoSQL ma'lumotlari":
    st.markdown("""
    <div class="platform-header">
        <h1>🌐 NoSQL ma'lumotlarini qayta ishlash</h1>
        <div class="subtitle">JSON · BSON · MongoDB · Hujjatli ma'lumotlar bazalari</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<span class="new-badge">YANGI</span>', unsafe_allow_html=True)
    
    # FIX: Mavjud ma'lumotlar haqida ogohlantirish
    if st.session_state.raw_data:
        existing = list(st.session_state.raw_data.keys())
        st.warning(
            f"⚠️ **Sessiyada mavjud:** {', '.join(existing)}. "
            f"NoSQL qo'shsangiz, bu ma'lumotlarga **qo'shiladi**. "
            f"Yangi sessiya boshlash uchun: chap menyu → **🗑️ Barcha ma'lumotlarni tozalash**."
        )
    
    tab1, tab2, tab3 = st.tabs([
        "📤 NoSQL fayl yuklash",
        "📝 Demo NoSQL ma'lumotlari",
        "📖 NoSQL haqida"
    ])
    
    with tab1:
        st.markdown("### Fayl tanlang")
        nosql_file = st.file_uploader(
            "JSON, NDJSON yoki BSON fayl yuklang",
            type=['json', 'ndjson', 'bson', 'jsonl'],
            key="nosql_uploader")
        
        if nosql_file:
            size_mb = nosql_file.size / (1024 * 1024)
            st.markdown(f"**Fayl:** {nosql_file.name} · **Hajm:** {size_mb:.2f} MB")
            
            try:
                with st.spinner("NoSQL ma'lumotlari o'qilmoqda..."):
                    if nosql_file.name.lower().endswith('.bson'):
                        df, format_type = load_bson(nosql_file)
                    else:
                        df, format_type = load_json_nosql(nosql_file, flatten=True)
                
                st.success(f"✅ Format: **{format_type}**")
                
                info = detect_nosql_structure(df)
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Hujjatlar", f"{len(df):,}")
                with c2: st.metric("Ustunlar", info['total_columns'])
                with c3: st.metric("Nested", info['nested_columns'])
                with c4: st.metric("Arrays", info['array_columns'])
                
                if info['has_mongodb_id']:
                    st.info("🔍 MongoDB ObjectId aniqlandi")
                
                st.markdown("### 📊 Tekislangan ma'lumotlar")
                st.dataframe(df.head(20), use_container_width=True)
                
                if info['array_columns'] > 0:
                    if st.checkbox("🔄 Massivlarni alohida qatorlarga ajratish (explode)"):
                        df = nosql_to_tabular(df)
                        st.success(f"✅ {len(df):,} qator")
                        st.dataframe(df.head(20), use_container_width=True)
                
                st.markdown("### Ma'lumot turini tanlang")
                # FIX: NoSQL uchun ham fayl nomi bo'yicha aniqlash
                fname_detected = detect_type_by_filename(nosql_file.name)
                col_detected = auto_detect_table_type(df)
                detected = fname_detected if fname_detected is not None else col_detected
                
                types = ['clients', 'accounts', 'transactions', 'loans', 'deposits', 'unknown']
                type_labels = {
                    'clients': 'Mijozlar', 'accounts': 'Hisoblar',
                    'transactions': 'Operatsiyalar', 'loans': 'Kreditlar',
                    'deposits': 'Depozitlar', 'unknown': "Noma'lum"
                }
                idx = types.index(detected) if detected in types else 5
                selected_type = st.selectbox(
                    f"Tur (aniqlandi: **{type_labels.get(detected, detected)}**)",
                    types, index=idx, format_func=lambda x: type_labels[x])
                
                if selected_type != 'unknown':
                    if st.button(f"✅ Ushbu NoSQL ma'lumotlarini qo'shish ({type_labels[selected_type]})",
                                 type="primary", use_container_width=True):
                        if st.session_state.raw_data is None:
                            st.session_state.raw_data = {}
                        st.session_state.raw_data[selected_type] = df
                        st.session_state.data_source = f"NoSQL: {nosql_file.name}"
                        # FIX: Keyingi bosqichlarni reset qilish
                        for rk in ['clean_data', 'clean_reports', 'validation_results',
                                   'kpis', 'analytics', 'marts']:
                            st.session_state[rk] = None
                        st.success(f"✅ NoSQL ma'lumotlari qo'shildi: **{type_labels[selected_type]}**!")
                        st.info("➡️ Endi chap menyu → **2️⃣ Sifat tekshiruvi** ga o'ting")
            
            except Exception as e:
                st.error(f"❌ Xato: {str(e)}")
    
    with tab2:
        st.markdown("### 📝 Demo NoSQL ma'lumotlari")
        demo = get_nosql_sample_data()
        st.code(json.dumps(demo, indent=2, ensure_ascii=False), language='json')
        
        if st.button("🔄 Demo qayta ishlash", type="primary"):
            df = pd.json_normalize(demo)
            st.dataframe(df, use_container_width=True)
            info = detect_nosql_structure(df)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Hujjatlar", len(demo))
            with c2: st.metric("Nested", info['nested_columns'])
            with c3: st.metric("Arrays", info['array_columns'])
    
    with tab3:
        st.markdown("""
        **NoSQL** — MongoDB, Cassandra kabi hujjat-orientiruvchi bazalar.
        Platforma avtomatik: nested → tekislaydi, arrays → qatorlarga ajratadi, jadval formatiga aylantiradi.
        """)

# ============= 1. GENERATSIYA =============
elif page == "1️⃣ Ma'lumot generatsiyasi":
    st.markdown("""
    <div class="platform-header"><h1>1️⃣ Ma'lumot generatsiyasi</h1>
    <div class="subtitle">Test uchun sintetik ma'lumotlar</div></div>
    """, unsafe_allow_html=True)
    st.info("💡 Haqiqiy ma'lumotlaringiz bo'lsa — **📁 Fayllarni yuklash** sahifasiga o'ting")
    
    c1, c2 = st.columns(2)
    with c1:
        n_c = st.slider("👥 Mijozlar", 100, 5000, 1000, 100)
        n_a = st.slider("💳 Hisoblar", 100, 10000, 1500, 100)
        n_t = st.slider("💸 Operatsiyalar", 1000, 50000, 10000, 1000)
    with c2:
        n_l = st.slider("📑 Kreditlar", 50, 3000, 500, 50)
        n_d = st.slider("💰 Depozitlar", 50, 3000, 800, 50)
        dirty = st.slider("⚠️ 'Iflos' ma'lumotlar (%)", 0, 30, 15)
    
    if st.button("🚀 Generatsiya qilish", type="primary", use_container_width=True):
        with st.spinner("..."):
            data = generate_all_data(n_c, n_a, n_t, n_l, n_d, dirty/100)
            st.session_state.raw_data = data
            st.session_state.data_source = "Generatsiya qilingan ma'lumotlar"
            for k in ['clean_data', 'validation_results', 'kpis', 'analytics', 'marts']:
                st.session_state[k] = None
        st.success("✅ Tayyor!")
    
    if st.session_state.raw_data:
        cols = st.columns(5)
        for i, (n, df) in enumerate(st.session_state.raw_data.items()):
            with cols[i % 5]: st.metric(n.capitalize(), f"{len(df):,}")

# ============= 2. SIFAT TEKSHIRUVI (FIXED) =============
elif page == "2️⃣ Sifat tekshiruvi":
    st.markdown("""
    <div class="platform-header"><h1>2️⃣ Sifat tekshiruvi</h1>
    <div class="subtitle">Xom ma'lumotlarni tahlil qilish</div></div>
    """, unsafe_allow_html=True)
    
    if st.session_state.raw_data is None:
        st.warning("⚠️ Avval ma'lumotlarni yuklang")
        st.info("➡️ **📁 Fayllarni yuklash** yoki **🌐 NoSQL ma'lumotlari** sahifasiga o'ting")
        st.stop()
    
    # FIX: Qaysi ma'lumotlar ko'rsatilayotganini aniq ko'rsatish
    tables = list(st.session_state.raw_data.keys())
    st.success(f"✅ **{len(tables)} ta jadval** mavjud: {', '.join(tables)}")
    src = st.session_state.data_source or "Noma'lum"
    st.caption(f"Manba: {src}")
    
    for name, df in st.session_state.raw_data.items():
        with st.expander(f"📋 **{name}** ({len(df):,} qator)", expanded=True):
            try:
                hashable_cols = [c for c in df.columns
                                 if not any(isinstance(v, (list, dict))
                                            for v in df[c].dropna().head(10))]
                dupes_count = df[hashable_cols].duplicated().sum() if hashable_cols else 0
            except Exception:
                dupes_count = 0
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Qatorlar", f"{len(df):,}")
            with c2: st.metric("Dublikatlar", f"{dupes_count:,}")
            with c3: st.metric("NULL", f"{df.isna().sum().sum():,}")
            with c4:
                total = len(df) * len(df.columns)
                q = ((total - df.isna().sum().sum()) / total * 100) if total > 0 else 0
                st.metric("Sifat", f"{q:.1f}%")
            
            info = pd.DataFrame({
                'Ustun': df.columns,
                'Tur': df.dtypes.astype(str).values,
                "To'ldirilgan": df.notna().sum().values,
                'NULL': df.isna().sum().values,
                '% NULL': (df.isna().sum() / len(df) * 100).round(1).values
            })
            st.dataframe(info, use_container_width=True, hide_index=True)
            
            st.markdown("**Namuna (10 qator):**")
            st.dataframe(df.head(10), use_container_width=True)

# ============= 3. TOZALASH =============
elif page == "3️⃣ Ma'lumotlarni tozalash":
    st.markdown("""
    <div class="platform-header"><h1>3️⃣ Ma'lumotlarni tozalash</h1>
    <div class="subtitle">Dublikatlar · NULL · anomaliyalar · normalizatsiya</div></div>
    """, unsafe_allow_html=True)
    
    if st.session_state.raw_data is None:
        st.warning("⚠️ Avval ma'lumotlarni yuklang"); st.stop()
    
    if st.button("🧹 Tozalashni boshlash", type="primary", use_container_width=True):
        with st.spinner("..."):
            clean_data, reports = {}, {}
            for n, df in st.session_state.raw_data.items():
                c, r = clean_dataset(df, n)
                clean_data[n], reports[n] = c, r
            st.session_state.clean_data = clean_data
            st.session_state.clean_reports = reports
        st.success("✅ Tozalandi!")
    
    if st.session_state.clean_data is not None:
        for name in st.session_state.raw_data.keys():
            if name not in st.session_state.clean_data:
                continue
            raw, cln = st.session_state.raw_data[name], st.session_state.clean_data[name]
            r = st.session_state.clean_reports[name]
            with st.expander(f"📋 **{name}**: {len(raw):,} → {len(cln):,}"):
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Oldin", f"{len(raw):,}")
                with c2: st.metric("Keyin", f"{len(cln):,}")
                with c3: st.metric("Dublikat", f"{r['duplicates_removed']:,}")
                with c4: st.metric("NULL to'ldirildi", f"{r['nulls_filled']:,}")
                for a in r['actions']: st.write(f"✓ {a}")

# ============= 4. VALIDATSIYA =============
elif page == "4️⃣ Validatsiya":
    st.markdown("""
    <div class="platform-header"><h1>4️⃣ Validatsiya</h1>
    <div class="subtitle">15+ bank qoidalari</div></div>
    """, unsafe_allow_html=True)
    
    if st.session_state.clean_data is None:
        st.warning("⚠️ Avval ma'lumotlarni tozalang"); st.stop()
    
    if st.button("✔️ Validatsiya", type="primary", use_container_width=True):
        with st.spinner("..."):
            st.session_state.validation_results = validate_dataset(st.session_state.clean_data)
        st.success("✅ Tayyor!")
    
    if st.session_state.validation_results is not None:
        r = st.session_state.validation_results
        total = len(r['checks'])
        p = sum(1 for c in r['checks'] if c['status'] == 'passed')
        w = sum(1 for c in r['checks'] if c['status'] == 'warning')
        f = sum(1 for c in r['checks'] if c['status'] == 'failed')
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Jami", total)
        with c2: st.metric("✅", p)
        with c3: st.metric("⚠️", w)
        with c4: st.metric("❌", f)
        for c in r['checks']:
            icon = {"passed": "✅", "warning": "⚠️", "failed": "❌"}[c['status']]
            st.markdown(f"{icon} **{c['rule']}** — {c['message']}")

# ============= 5. KPI =============
elif page == "5️⃣ KPI hisoblash":
    st.markdown("""
    <div class="platform-header"><h1>5️⃣ Bank KPI</h1>
    <div class="subtitle">NPL · LDR · ROA · ROE · CAR</div></div>
    """, unsafe_allow_html=True)
    
    if st.session_state.clean_data is None:
        st.warning("⚠️ Avval ma'lumotlarni tozalang"); st.stop()
    
    if st.button("📊 KPI hisoblash", type="primary", use_container_width=True):
        with st.spinner("..."):
            st.session_state.kpis = calculate_kpis(st.session_state.clean_data)
        st.success("✅ Tayyor!")
    
    if st.session_state.kpis is not None:
        k = st.session_state.kpis
        cols = st.columns(5)
        with cols[0]: st.metric("NPL", f"{k['npl_ratio']:.2f}%")
        with cols[1]: st.metric("LDR", f"{k['ldr']:.1f}%")
        with cols[2]: st.metric("ROA", f"{k['roa']:.2f}%")
        with cols[3]: st.metric("ROE", f"{k['roe']:.2f}%")
        with cols[4]: st.metric("CAR", f"{k['car']:.2f}%")
        
        fig = go.Figure(go.Bar(
            x=['NPL', 'LDR', 'ROA', 'ROE', 'CAR'],
            y=[k['npl_ratio'], k['ldr'], k['roa'], k['roe'], k['car']],
            marker_color=['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'],
            text=[f"{v:.1f}%" for v in [k['npl_ratio'], k['ldr'], k['roa'], k['roe'], k['car']]],
            textposition='outside', textfont=dict(color='white')))
        fig.update_layout(title="Moliyaviy koeffitsiyentlar", height=400,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,41,59,0.5)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155'))
        st.plotly_chart(fig, use_container_width=True)

# ============= 6. KENGAYTIRILGAN TAHLIL =============
elif page == "6️⃣ Kengaytirilgan tahlil":
    st.markdown("""
    <div class="platform-header"><h1>6️⃣ Kengaytirilgan bank tahlili</h1>
    <div class="subtitle">ABC · Churn · Risk · Likvidlik · Hudud · Kanallar</div></div>
    """, unsafe_allow_html=True)
    
    if st.session_state.clean_data is None:
        st.warning("⚠️ Avval ma'lumotlarni tozalang"); st.stop()
    
    if st.button("🔬 Kompleks tahlil", type="primary", use_container_width=True):
        with st.spinner("Bajarilmoqda..."):
            st.session_state.analytics = comprehensive_analysis(st.session_state.clean_data)
        st.success("✅ Tahlil tugadi!")
    
    if st.session_state.analytics is not None:
        a = st.session_state.analytics
        
        if a.get('churn'):
            st.markdown("### 📉 Mijozlar oqimi")
            ch = a['churn']
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Mijozlar", f"{ch.get('total_clients', 0):,}")
            with c2: st.metric("Churn Rate", f"{ch.get('churn_rate_pct', 0):.2f}%")
            with c3:
                if 'dormant_accounts' in ch:
                    st.metric("Uyqudagi", f"{ch['dormant_accounts']:,}")
        
        if a.get('abc') and a['abc'].get('summary') is not None:
            st.markdown("### 🎯 ABC-tahlil")
            abc_sum = a['abc']['summary']
            st.dataframe(abc_sum, use_container_width=True, hide_index=True)
        
        if a.get('concentration'):
            st.markdown("### ⚠️ Kredit riski konsentratsiyasi")
            con = a['concentration']
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if 'top_10_share_pct' in con: st.metric("Top-10", f"{con['top_10_share_pct']:.1f}%")
            with c2:
                if 'top_20_share_pct' in con: st.metric("Top-20", f"{con['top_20_share_pct']:.1f}%")
            with c3:
                if 'top_50_share_pct' in con: st.metric("Top-50", f"{con['top_50_share_pct']:.1f}%")
            with c4:
                if 'top_100_share_pct' in con: st.metric("Top-100", f"{con['top_100_share_pct']:.1f}%")
        
        if a.get('liquidity'):
            st.markdown("### 💧 Likvidlik")
            liq = a['liquidity']
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Kreditlar", f"{liq['total_loans']/1e9:.2f} mlrd")
            with c2: st.metric("Depozitlar", f"{liq['total_deposits']/1e9:.2f} mlrd")
            with c3: st.metric("Gap", f"{liq['liquidity_gap']/1e9:.2f} mlrd")

# ============= 7. VITRINALAR =============
elif page == "7️⃣ Ma'lumot vitrinalari":
    st.markdown("""
    <div class="platform-header"><h1>7️⃣ Ma'lumot vitrinalari</h1>
    <div class="subtitle">Star Schema · Dim · Fact · Aggregates</div></div>
    """, unsafe_allow_html=True)
    
    if st.session_state.clean_data is None:
        st.warning("⚠️ Avval ma'lumotlarni tozalang"); st.stop()
    
    if st.button("🏗️ Vitrinalarni qurish", type="primary", use_container_width=True):
        with st.spinner("..."):
            st.session_state.marts = build_marts(st.session_state.clean_data)
        st.success("✅ Tayyor!")
    
    if st.session_state.marts is not None:
        st.markdown(f"### 📦 Vitrinalar: **{len(st.session_state.marts)}**")
        for name, df in st.session_state.marts.items():
            t = "📐 O'lchov" if name.startswith('dim') else "📊 Fakt" if name.startswith('fact') else "📈 Agregat"
            with st.expander(f"{t} · **{name}** · {len(df):,} × {len(df.columns)}"):
                st.dataframe(df.head(50), use_container_width=True)
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(f"⬇️ {name}.csv", data=csv,
                    file_name=f"{name}.csv", mime="text/csv", key=f"dl_{name}")

# ============= 8. EKSPORT =============
elif page == "8️⃣ Superset ga eksport":
    st.markdown("""
    <div class="platform-header"><h1>8️⃣ Apache Superset ga eksport</h1>
    <div class="subtitle">BI uchun SQLite ma'lumotlar bazasi</div></div>
    """, unsafe_allow_html=True)
    
    if st.session_state.marts is None:
        st.warning("⚠️ Avval vitrinalarni quring"); st.stop()
    
    if st.button("💾 SQLite ga eksport", type="primary", use_container_width=True):
        with st.spinner("..."):
            os.makedirs("data", exist_ok=True)
            export_to_sqlite(st.session_state.marts, "data/bigdataplatform.db")
        st.success("✅ Tayyor!")
    
    if os.path.exists("data/bigdataplatform.db"):
        sz = os.path.getsize("data/bigdataplatform.db") / 1024 / 1024
        st.markdown(f"### 📦 {sz:.2f} MB")
        with open("data/bigdataplatform.db", "rb") as f:
            st.download_button("💾 SQLite bazani yuklab olish", data=f,
                file_name="bigdataplatform.db", mime="application/x-sqlite3",
                use_container_width=True)

# ============= YORDAM =============
elif page == "❓ Qanday foydalanish":
    st.markdown("""
    <div class="platform-header">
        <h1>❓ Qo'llanma</h1>
        <div class="subtitle">Ma'lumotlarni yuklashdan tahlilgacha</div>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🚀 Tez demo", "📁 Haqiqiy ma'lumotlar", "🌐 NoSQL"])
    
    with tab1:
        st.markdown("""
        1. **⚡ Tezkor ishga tushirish** → O'rta (5K) → **🚀 BUTUN PIPELINE** tugmasi
        2. 10 soniyada: 30 000+ yozuv → tozalash → validatsiya → KPI → 8 tahlil → vitrinalar
        """)
    
    with tab2:
        st.markdown("""
        1. **📁 Fayllarni yuklash** → CSV/Excel yuklang
        2. Platforma **fayl nomi va ustunlar** bo'yicha turini aniqlaydi
        3. **2️⃣ → 3️⃣ → 4️⃣ → 5️⃣ → 6️⃣ → 7️⃣ → 8️⃣** tartibida o'ting
        
        **Tavsiya etilgan ustunlar:**
        - Mijozlar: `client_id, first_name, inn, region`
        - Kreditlar: `loan_id, principal_amount, interest_rate, status`
        - Depozitlar: `deposit_id, principal_amount, term_months`
        """)
    
    with tab3:
        st.markdown("""
        1. **🌐 NoSQL ma'lumotlari** → JSON/BSON fayl yuklang
        2. Platforma nested strukturalarni tekislaydi
        3. Tur tanlang → **✅ Qo'shish** → **2️⃣ Sifat tekshiruvi**
        """)

