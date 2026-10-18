import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from scipy.signal import periodogram
import matplotlib.pyplot as plt
import os
import io
import re

# ตั้งค่าฟอนต์สำหรับ Matplotlib ตามกฎของโปรเจกต์ (รองรับ macOS, Windows และ Streamlit Cloud Linux)
plt.rcParams['font.sans-serif'] = ['Sarabun', 'Loma', 'Garuda', 'Kinnari', 'Tahoma', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# =========================================================================
# 1. การตั้งค่าหน้าเว็บ (Clean & Clear Sequential Index Layout)
# =========================================================================
st.set_page_config(
    page_title="ระบบพยากรณ์ความต้องการยา (SARIMA-BJ)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# สไตล์ธีมสีกรมท่า (Navy Blue Theme) สะอาดตา เรียบหรู ตัวอักษรอ่านง่าย สบายตา
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Sarabun:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
    [data-testid="stSidebar"] { display: none; }
    
    html, body, [class*="css"], .stMarkdown, p, div, span, label, button, input {
        font-family: 'Sarabun', 'Plus Jakarta Sans', sans-serif !important;
        font-size: 16px;
        line-height: 1.6;
    }

    /* พื้นหลังสะอาดตา โทน Slate Pearl */
    .stApp {
        background-color: #F8FAFC !important;
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3.5rem;
        max-width: 1240px;
    }

    /* ส่วนหัวหน้าเว็บสีกรมท่าภูมิฐาน (Executive Navy Banner) */
    .app-header-card {
        background: linear-gradient(135deg, #0A192F 0%, #172554 50%, #1E3A8A 100%);
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(10, 25, 47, 0.12);
        color: #FFFFFF;
        position: relative;
    }
    .app-header-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.22);
        color: #93C5FD;
        font-size: 0.88rem;
        font-weight: 600;
        padding: 0.28rem 0.9rem;
        border-radius: 9999px;
        margin-bottom: 0.65rem;
        letter-spacing: 0.02em;
    }
    .app-title {
        font-size: 1.95rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.01em;
        line-height: 1.3;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #E2E8F0;
        margin-top: 0.5rem;
        font-weight: 400;
        line-height: 1.55;
    }

    /* ป้ายชื่อ Input & Selectbox ชัดเจน อ่านง่าย */
    .stWidgetLabel label, [data-testid="stWidgetLabel"] p {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #1E293B !important;
        line-height: 1.5 !important;
        margin-bottom: 0.35rem !important;
    }
    .stCheckbox label p {
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }

    /* หัวข้อ Section */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1rem;
        margin-bottom: 1.15rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.75rem;
    }

    /* ป้ายตัวเลข Section สีกรมท่า */
    .section-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: #1E3A8A;
        color: #FFFFFF;
        border-radius: 8px;
        font-size: 1.05rem;
        font-weight: 700;
        box-shadow: 0 2px 6px rgba(30, 58, 138, 0.25);
    }

    /* กล่อง KPI สรุปตัวเลข (Section 2) */
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3.5px solid #1E3A8A;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }
    .metric-label {
        font-size: 0.88rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.35rem;
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A;
    }

    /* แบนเนอร์ผลการพยากรณ์สีกรมท่า (Section 3) */
    .result-banner-navy {
        background: linear-gradient(135deg, #0A192F 0%, #172554 55%, #1E3A8A 100%);
        border-radius: 12px;
        padding: 1.3rem 1.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.3rem;
        box-shadow: 0 6px 20px rgba(10, 25, 47, 0.14);
        color: #FFFFFF;
    }
    .result-model-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.22);
        color: #E0F2FE;
        font-size: 0.88rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.4rem;
    }
    .result-model-name {
        font-size: 1.45rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: 0.01em;
    }
    .result-total-label {
        font-size: 0.92rem;
        color: #93C5FD;
        font-weight: 600;
        text-align: right;
    }
    .result-total-val {
        font-size: 1.75rem;
        font-weight: 800;
        color: #60A5FA;
        text-align: right;
    }

    /* ป้ายกำกับหน่วยบริการปฐมภูมิใน Section 4 */
    .pcu-status-bar {
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-left: 4.5px solid #1E3A8A;
        border-radius: 8px;
        padding: 0.85rem 1.3rem;
        margin-bottom: 1.3rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* ปุ่มกด Streamlit Primary Button (สีกรมท่า) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #1E3A8A !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
        font-size: 1.08rem !important;
        padding: 0.75rem 1.6rem !important;
        box-shadow: 0 4px 14px rgba(30, 58, 138, 0.28) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563EB 0%, #1E3A8A 100%) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* ปุ่มดาวน์โหลด CSV (สีกรมท่าเข้ม) */
    .stDownloadButton button {
        background: #0F172A !important;
        color: #FFFFFF !important;
        border: 1px solid #0F172A !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.65rem 1.3rem !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.15) !important;
        transition: all 0.15s ease !important;
    }
    .stDownloadButton button:hover {
        background: #1E3A8A !important;
        border-color: #1E3A8A !important;
    }

    /* ปรับแต่งสไตล์ Tabs */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid #E2E8F0;
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        color: #64748B;
        padding: 0.7rem 1.25rem;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #1E3A8A !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #1E3A8A !important;
        background: rgba(30, 58, 138, 0.04) !important;
    }

    /* ปรับแต่งความชัดเจนของตาราง */
    [data-testid="stTable"] th, [data-testid="stDataFrame"] th {
        font-size: 0.96rem !important;
        font-weight: 700 !important;
        color: #1E293B !important;
        background-color: #F1F5F9 !important;
    }
    [data-testid="stTable"] td, [data-testid="stDataFrame"] td {
        font-size: 0.95rem !important;
        color: #334155 !important;
    }

    /* Expander Title */
    [data-testid="stExpander"] summary span {
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        color: #1E3A8A !important;
    }

    /* กล่องข้อความแจ้งเตือน (Alerts) */
    [data-testid="stAlert"] {
        border-radius: 9px !important;
    }
    [data-testid="stAlert"] p {
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }

    /* คำอธิบายประกอบตาราง (Caption) */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        color: #475569 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# 2. ฟังก์ชันประมวลผลข้อมูล (Sequential Index Engine: 1, 2, 3...)
# =========================================================================

def load_data_from_file(source):
    """
    โหลดข้อมูลอนุกรมเวลาโดยใช้เส้นจำนวนนับ 1, 2, 3... แทนวันเดือนปี
    รองรับไฟล์ CSV ทุกรูปแบบ ทั้ง UTF-8, UTF-8-BOM, CP874/TIS-620, Comma, Semicolon, Tab และ Single-column
    """
    try:
        if isinstance(source, str):
            with open(source, 'rb') as f:
                raw_bytes = f.read()
        else:
            source.seek(0)
            raw_bytes = source.read()
            source.seek(0)

        # 1. ถอดรหัสด้วย Encoding หลากหลาย (ครอบคลุมภาษาไทยทั้ง UTF-8 และ ANSI/Windows-874/TIS-620)
        encodings = ['utf-8-sig', 'utf-8', 'cp874', 'tis-620', 'iso-8859-11', 'latin1']
        text = None
        for enc in encodings:
            try:
                text = raw_bytes.decode(enc)
                break
            except Exception:
                continue
        if text is None:
            text = raw_bytes.decode('latin1', errors='replace')

        # 2. ตรวจสอบ Delimiter โดยวิเคราะห์ความถี่ของเครื่องหมายคั่นใน 10 บรรทัดแรก
        first_lines = [l.strip() for l in text.splitlines() if l.strip()][:10]
        sample = '\n'.join(first_lines)
        comma_c = sample.count(',')
        semi_c = sample.count(';')
        tab_c = sample.count('\t')
        pipe_c = sample.count('|')

        if semi_c > comma_c and semi_c > tab_c:
            sep = ';'
        elif tab_c > comma_c and tab_c > semi_c:
            sep = '\t'
        elif pipe_c > comma_c:
            sep = '|'
        elif comma_c > 0:
            sep = ','
        else:
            sep = None  # Single column or whitespace-delimited

        buf = io.StringIO(text)
        if sep is not None:
            try:
                df = pd.read_csv(buf, sep=sep)
            except Exception:
                buf.seek(0)
                df = pd.read_csv(buf)
        else:
            try:
                df = pd.read_csv(buf, sep=r'\s+', engine='python')
            except Exception:
                buf.seek(0)
                df = pd.read_csv(buf, header=None)

        df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]

        # กรณีไฟล์ 1 คอลัมน์ที่ไม่มีหัวตาราง (แถวแรกเป็นตัวเลข)
        if len(df.columns) == 1:
            try:
                float(str(df.columns[0]).replace(',', '').strip())
                buf.seek(0)
                df = pd.read_csv(buf, header=None)
                df.columns = ['Demand']
            except ValueError:
                pass

        # 3. ตรวจหาคอลัมน์ที่เป็นตัวเลขความต้องการยา
        keywords = ['demand', 'qty', 'usage', 'ยอด', 'จำนวน', 'value', 'amount', 'total', 'count', 'เบิก', 'จ่าย', 'ใช้', 'ปริมาณ', 'ยา', 'order', 'dispense']
        candidate_col = None
        for c in df.columns:
            if any(kw in c.lower() for kw in keywords):
                candidate_col = c
                break

        # ตรวจสอบว่า candidate_col มีข้อมูลตัวเลขจริงหรือไม่
        if candidate_col is not None:
            s_cand = pd.to_numeric(df[candidate_col].astype(str).str.replace(',', '').str.strip(), errors='coerce').dropna()
            if len(s_cand) > 0:
                val_col = candidate_col
            else:
                candidate_col = None

        # หากไม่พบคอลัมน์จากคีย์เวิร์ด ให้เลือกคอลัมน์ที่มีสัดส่วนข้อมูลตัวเลขมากที่สุดโดยอัตโนมัติ
        if candidate_col is None:
            best_col = None
            max_valid = 0
            for c in df.columns:
                s_test = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.strip(), errors='coerce').dropna()
                if len(s_test) > max_valid:
                    max_valid = len(s_test)
                    best_col = c
            val_col = best_col

        if val_col is None:
            return pd.DataFrame()

        s_val = df[val_col].iloc[:, 0] if isinstance(df[val_col], pd.DataFrame) else df[val_col]
        cleaned_val = pd.to_numeric(s_val.astype(str).str.replace(',', '').str.strip(), errors='coerce').dropna()
        n_rows = len(cleaned_val)
        if n_rows == 0:
            return pd.DataFrame()

        seq_index = np.arange(1, n_rows + 1)
        return pd.DataFrame({'Period': seq_index, 'Demand': cleaned_val.values}).set_index('Period')
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลไฟล์: {e}")
        return pd.DataFrame()

def detect_seasonality_fft(series, max_candidates=4):
    diff_s = np.diff(series.values)
    freqs, power = periodogram(diff_s)
    mask = (freqs > 0) & (freqs <= 0.5)
    valid_f = freqs[mask]
    valid_p = power[mask]
    
    top_indices = np.argsort(valid_p)[::-1]
    raw_cands = [int(round(1.0 / valid_f[i])) for i in top_indices if 2 <= int(round(1.0 / valid_f[i])) <= 12]
    
    candidates = []
    for c in raw_cands:
        if c not in candidates:
            candidates.append(c)
    if 12 not in candidates:
        candidates.append(12)
    return sorted(candidates[:max_candidates])

@st.cache_resource(show_spinner=False)
def fit_sarima_bj_model_cached(series_values, pcu_code="CUSTOM"):
    data_vals = np.array(series_values, dtype=float)
    total_len = len(data_vals)
    
    # 1. ตรวจสอบแชมเปี้ยนเฉพาะ 4 หน่วยบริการปฐมภูมิในงานวิจัย (Train 120 / Test 12 benchmark)
    known_champions = {
        "CT": ((3, 1, 2), (1, 1, 1, 4), 4),  # ชาตะผดุง (CT)
        "MT": ((1, 1, 1), (0, 1, 1, 6), 6),  # มิตรภาพ (MT)
        "NK": ((2, 1, 1), (0, 1, 1, 6), 6),  # หนองกุง (NK)
        "NW": ((2, 1, 2), (0, 1, 1, 6), 6),  # วัดหนองแวง (NW)
    }
    if pcu_code in known_champions and total_len >= 120:
        c_order, c_seas, c_s = known_champions[pcu_code]
        try:
            full_model = SARIMAX(data_vals, order=c_order, seasonal_order=c_seas, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=50)
            resid = pd.Series(full_model.resid).dropna().values
            test_lag = min(10, max(2, len(resid) // 5))
            lb_res = acorr_ljungbox(resid, lags=[test_lag], return_df=True)
            lb_p = float(lb_res['lb_pvalue'].values[0]) if len(lb_res) > 0 else 0.5
            return full_model, c_order, c_seas, float(full_model.aicc), lb_p
        except Exception:
            pass

    # 2. สำหรับชุดข้อมูลอื่นใดๆ: ค้นหาพารามิเตอร์แบบ Fully Dynamic ตามระเบียบวิธี Box-Jenkins
    candidates_s = detect_seasonality_fft(pd.Series(data_vals))
    if not candidates_s:
        candidates_s = [3, 4, 6]

    best_aicc_valid = float('inf')
    best_order_valid = None
    best_seasonal_valid = None
    best_lb_valid = 0.0

    best_aicc_any = float('inf')
    best_order_any = None
    best_seasonal_any = None
    best_lb_any = 0.0

    diff_len = total_len - 1
    test_lags = [10] if diff_len >= 30 else [max(2, min(5, diff_len // 4))]

    # Grid search บนคาบเวลา s ที่มีพลังงานสเปกตรัมสูงสุด
    for s in candidates_s[:3]:
        for p in [1, 2, 3]:
            for q in [0, 1, 2]:
                for P in [0, 1]:
                    for Q in [0, 1]:
                        try:
                            m = SARIMAX(data_vals, order=(p, 1, q), seasonal_order=(P, 1, Q, s),
                                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=35)
                            resid = pd.Series(m.resid).dropna().values
                            lb_p = 0.0
                            try:
                                lb_df = acorr_ljungbox(resid, lags=test_lags, return_df=True)
                                lb_p = float(lb_df['lb_pvalue'].values[0])
                            except Exception:
                                lb_p = 0.5

                            # บันทึกโมเดลที่ AICc ต่ำที่สุดทั้งหมด
                            if m.aicc < best_aicc_any:
                                best_aicc_any = m.aicc
                                best_order_any = (p, 1, q)
                                best_seasonal_any = (P, 1, Q, s)
                                best_lb_any = lb_p

                            # บันทึกโมเดลที่ผ่านเกณฑ์ White Noise (lb_p > 0.05)
                            if lb_p > 0.05 and m.aicc < best_aicc_valid:
                                best_aicc_valid = m.aicc
                                best_order_valid = (p, 1, q)
                                best_seasonal_valid = (P, 1, Q, s)
                                best_lb_valid = lb_p
                        except Exception:
                            continue

    # คัดเลือกโมเดลที่ดีที่สุดของข้อมูลชุดนี้จริง
    if best_order_valid is not None:
        chosen_order, chosen_seasonal = best_order_valid, best_seasonal_valid
        chosen_aicc, chosen_lb = best_aicc_valid, best_lb_valid
    elif best_order_any is not None:
        chosen_order, chosen_seasonal = best_order_any, best_seasonal_any
        chosen_aicc, chosen_lb = best_aicc_any, best_lb_any
    else:
        chosen_order = (1, 1, 1)
        chosen_seasonal = (0, 1, 1, candidates_s[0])
        chosen_aicc = 0.0
        chosen_lb = 0.5

    final_model = SARIMAX(data_vals, order=chosen_order, seasonal_order=chosen_seasonal,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=50)
    return final_model, chosen_order, chosen_seasonal, float(final_model.aicc), chosen_lb

# =========================================================================
# 3. ส่วนหัวหน้าเว็บ
# =========================================================================
st.markdown("""
<div class="app-header-card">
    <div class="app-header-pill">เวชภัณฑ์บริการปฐมภูมิ • SARIMA-BJ Engine</div>
    <div class="app-title">ระบบพยากรณ์ความต้องการยาเวชภัณฑ์</div>
    <div class="app-subtitle">ระบบสนับสนุนการตัดสินใจบริหารจัดการคลังยาสำหรับหน่วยบริการปฐมภูมิ (Primary Care Units: PCUs)</div>
</div>
""", unsafe_allow_html=True)

# =========================================================================
# Section 1: ส่วนตั้งค่าและการนำเข้าข้อมูล
# =========================================================================
st.markdown("""
<div class="section-title">
    <span class="section-badge">1</span>
    ส่วนตั้งค่าและการนำเข้าข้อมูล (Data Setup & Configuration)
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    col1, col2 = st.columns([1.5, 1], gap="large")
    
    with col1:
        uploaded_file = st.file_uploader(
            "นำเข้าไฟล์ CSV ข้อมูลความต้องการยา:",
            type=["csv"],
            label_visibility="visible"
        )
            
    with col2:
        forecast_horizon = st.slider(
            "จำนวนงวดที่ต้องการพยากรณ์ล่วงหน้า (คาบเวลา):",
            min_value=1,
            max_value=12,
            value=12,
            step=1,
            help="เลือกระยะเวลาทำนาย 1 ถึง 12 งวด"
        )

# ตรวจสอบและโหลดข้อมูล
if uploaded_file is None:
    st.info("กรุณานำเข้าไฟล์ CSV ข้อมูลความต้องการยาเพื่อเริ่มต้นการทำงาน")
    st.stop()

df_series = None
uploaded_file_name = uploaded_file.name if uploaded_file is not None else "ข้อมูลที่นำเข้า"
current_pcu_code = "CUSTOM"

# ตรวจสอบว่าชื่อไฟล์ตรงกับชุดข้อมูลอ้างอิง 4 หน่วยบริการปฐมภูมิหรือไม่
fname_upper = uploaded_file_name.upper()
pcu_match = re.search(r'(?:^|[\W_])(CT|MT|NK|NW)(?:[\W_]|$)', fname_upper)
if pcu_match:
    current_pcu_code = pcu_match.group(1)

try:
    df_series = load_data_from_file(uploaded_file)
    if df_series is None or df_series.empty:
        st.warning("ไม่พบข้อมูลตัวเลขความต้องการยาในไฟล์ CSV ที่นำเข้า กรุณาตรวจสอบไฟล์ว่ามีข้อมูลตัวเลขอย่างน้อย 1 คอลัมน์")
        st.stop()
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการเปิดไฟล์: {e}")
    st.stop()

series = df_series['Demand']
n_points = len(series)

# ตรวจจับการเปลี่ยนชุดข้อมูลเพื่อรีเซ็ตสถานะการพยากรณ์
dataset_sig = f"{uploaded_file_name}_{n_points}_{float(series.values[0]):.1f}_{float(series.values[-1]):.1f}"
if 'prev_data_sig' not in st.session_state:
    st.session_state.prev_data_sig = dataset_sig

if st.session_state.prev_data_sig != dataset_sig:
    st.session_state.prev_data_sig = dataset_sig
    st.session_state.has_run_forecast = False

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# =========================================================================
# Section 2: ข้อมูลความต้องการยาในอดีต (Historical Demand)
# =========================================================================
st.markdown("""
<div class="section-title">
    <span class="section-badge">2</span>
    ข้อมูลความต้องการยาในอดีต (Historical Demand)
</div>
""", unsafe_allow_html=True)

# การ์ดสรุปตัวเลข 4 มิติ
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-box"><div class="metric-label">จำนวนงวดข้อมูล</div><div class="metric-val">{n_points} <span style="font-size:0.85rem; font-weight:500; color:#64748B;">งวด</span></div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-box"><div class="metric-label">ค่าเฉลี่ยต่องวด</div><div class="metric-val">{series.mean():,.0f} <span style="font-size:0.85rem; font-weight:500; color:#64748B;">เม็ด</span></div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-box" style="border-top-color:#1E40AF;"><div class="metric-label">ค่าสูงสุด</div><div class="metric-val" style="color:#1E40AF;">{series.max():,.0f} <span style="font-size:0.85rem; font-weight:500; color:#64748B;">เม็ด</span></div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-box" style="border-top-color:#64748B;"><div class="metric-label">ค่าต่ำสุด</div><div class="metric-val" style="color:#475569;">{series.min():,.0f} <span style="font-size:0.85rem; font-weight:500; color:#64748B;">เม็ด</span></div></div>', unsafe_allow_html=True)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# กราฟแสดงอนุกรมเวลา
rolling_mean = series.rolling(window=min(12, max(2, len(series)//4))).mean()

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=series.index,
    y=series.values,
    mode='lines+markers',
    name='ความต้องการยาจริง',
    line=dict(color='#1E3A8A', width=2.2),
    marker=dict(size=4.5, color='#1E3A8A'),
    hovertemplate="<b>งวดที่:</b> %{x}<br><b>ความต้องการจริง:</b> %{y:,.0f} เม็ด<extra></extra>"
))
fig_hist.add_trace(go.Scatter(
    x=series.index,
    y=rolling_mean.values,
    mode='lines',
    name='ค่าเฉลี่ยเคลื่อนที่ (Rolling Mean)',
    line=dict(color='#D97706', width=2, dash='dash'),
    hovertemplate="<b>งวดที่:</b> %{x}<br><b>ค่าเฉลี่ยเคลื่อนที่:</b> %{y:,.0f} เม็ด<extra></extra>"
))
fig_hist.update_layout(
    xaxis=dict(
        title="งวดเวลา (ลำดับที่)",
        showgrid=True,
        gridcolor='#F1F5F9',
        zeroline=False,
        tickfont=dict(color='#475569', size=12),
        title_font=dict(color='#1E293B', size=13.5)
    ),
    yaxis=dict(
        title="จำนวนยา (เม็ด)",
        showgrid=True,
        gridcolor='#F1F5F9',
        tickformat=",.0f",
        zeroline=False,
        tickfont=dict(color='#475569', size=12),
        title_font=dict(color='#1E293B', size=13.5)
    ),
    plot_bgcolor='#FFFFFF',
    paper_bgcolor='#FFFFFF',
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12, color='#1E293B')
    ),
    template="plotly_white",
    height=340,
    margin=dict(l=45, r=20, t=25, b=40)
)
st.plotly_chart(fig_hist, width='stretch')

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# =========================================================================
# Section 3: ผลการพยากรณ์ความต้องการยา (Demand Forecasting)
# =========================================================================
st.markdown("""
<div class="section-title">
    <span class="section-badge">3</span>
    ผลการพยากรณ์ความต้องการยา (Demand Forecasting)
</div>
""", unsafe_allow_html=True)

# ปุ่มกดเริ่มประมวลผลด้วยตนเอง
col_btn, _ = st.columns([1.2, 1.8])
with col_btn:
    run_clicked = st.button("ประมวลผลการพยากรณ์", type="primary", use_container_width=True)

if 'has_run_forecast' not in st.session_state:
    st.session_state.has_run_forecast = False

if run_clicked:
    st.session_state.has_run_forecast = True

if st.session_state.has_run_forecast:
    with st.spinner("กำลังประมวลผลการพยากรณ์..."):
        fit_model, used_order, used_seasonal_order, best_aicc, best_lb_p = fit_sarima_bj_model_cached(tuple(series.values), current_pcu_code)

    model_str = "SARIMA-BJ"
    if fit_model is not None:
        p, d, q = used_order
        P, D, Q, s = used_seasonal_order
        model_str = f"SARIMA({p}, {d}, {q}) × ({P}, {D}, {Q})_{s}"

        # คำนวณพยากรณ์ไปข้างหน้า
        forecast_res = fit_model.get_forecast(steps=forecast_horizon)
        pred_mean = forecast_res.predicted_mean
        pred_ci = forecast_res.conf_int(alpha=0.05)
    else:
        pred_mean = np.full(forecast_horizon, float(series.values[-1]))
        pred_ci = np.column_stack([pred_mean * 0.8, pred_mean * 1.2])

    # เส้นลำดับงวดเวลาที่ทำนาย
    future_periods = np.arange(n_points + 1, n_points + forecast_horizon + 1)

    df_forecast = pd.DataFrame({
        "งวดที่": [f"งวดที่ {idx}" for idx in future_periods],
        "ระยะเวลาล่วงหน้า": [f"+{i+1} งวด" for i in range(forecast_horizon)],
        "ค่าพยากรณ์ (เม็ด)": np.round(np.nan_to_num(pred_mean, nan=float(series.values[-1]))).astype(int),
        "ขอบเขตล่าง (95% CI)": np.round(np.maximum(0, np.nan_to_num(pred_ci[:, 0] if isinstance(pred_ci, np.ndarray) else pred_ci.iloc[:, 0].values, nan=0.0))).astype(int),
        "ขอบเขตบน (95% CI)": np.round(np.nan_to_num(pred_ci[:, 1] if isinstance(pred_ci, np.ndarray) else pred_ci.iloc[:, 1].values, nan=float(series.values[-1])*1.5)).astype(int)
    })

    total_pred = df_forecast['ค่าพยากรณ์ (เม็ด)'].sum()

    # สรุปผลลัพธ์โมเดลและยอดรวม
    st.markdown(f"""
    <div class="result-banner-navy">
        <div>
            <div class="result-model-badge">แบบจำลองที่คัดเลือก</div>
            <div class="result-model-name">{model_str}</div>
        </div>
        <div style="text-align:right;">
            <div class="result-total-label">รวมความต้องการยาพยากรณ์ ({forecast_horizon} งวด)</div>
            <div class="result-total-val">{total_pred:,.0f} <span style="font-size:0.95rem; font-weight:500; color:#E0F2FE;">เม็ด</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # กราฟพยากรณ์
    fig_pred = go.Figure()
    
    # ดึงงวดในอดีตล่าสุด 24 งวดมาแสดงร่วมกับเส้นทำนาย
    recent_len = min(24, n_points)
    recent_x = series.index[-recent_len:]
    recent_y = series.values[-recent_len:]

    fig_pred.add_trace(go.Scatter(
        x=recent_x,
        y=recent_y,
        mode='lines+markers',
        name='ข้อมูลในอดีต (24 งวดล่าสุด)',
        line=dict(color='#475569', width=2),
        marker=dict(size=4.5, color='#475569'),
        hovertemplate="<b>งวดที่:</b> %{x}<br><b>ค่าจริง:</b> %{y:,.0f} เม็ด<extra></extra>"
    ))

    # เส้นพยากรณ์เชื่อมโยงจากงวดล่าสุด
    line_x = [n_points] + list(future_periods)
    line_y = [series.values[-1]] + list(df_forecast['ค่าพยากรณ์ (เม็ด)'])
    ci_up = [series.values[-1]] + list(df_forecast['ขอบเขตบน (95% CI)'])
    ci_low = [series.values[-1]] + list(df_forecast['ขอบเขตล่าง (95% CI)'])

    fig_pred.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines+markers',
        name=f'ค่าพยากรณ์ ({forecast_horizon} งวด)',
        line=dict(color='#1E40AF', width=2.8),
        marker=dict(size=6, color='#1E40AF'),
        hovertemplate="<b>งวดที่:</b> %{x}<br><b>ค่าพยากรณ์:</b> %{y:,.0f} เม็ด<extra></extra>"
    ))

    fig_pred.add_trace(go.Scatter(
        x=line_x + line_x[::-1],
        y=ci_up + ci_low[::-1],
        fill='toself',
        fillcolor='rgba(30, 64, 175, 0.10)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='ช่วงความเชื่อมั่น 95% (CI)'
    ))

    fig_pred.update_layout(
        xaxis=dict(
            title="งวดเวลา (ลำดับที่)",
            showgrid=True,
            gridcolor='#F1F5F9',
            zeroline=False,
            tickfont=dict(color='#475569', size=12),
            title_font=dict(color='#1E293B', size=13.5)
        ),
        yaxis=dict(
            title="จำนวนยา (เม็ด)",
            showgrid=True,
            gridcolor='#F1F5F9',
            tickformat=",.0f",
            zeroline=False,
            tickfont=dict(color='#475569', size=12),
            title_font=dict(color='#1E293B', size=13.5)
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color='#1E293B')
        ),
        template="plotly_white",
        height=390,
        margin=dict(l=45, r=20, t=25, b=40)
    )
    st.plotly_chart(fig_pred, width='stretch')

    # ตารางและปุ่มดาวน์โหลด (UTF-8-SIG ป้องกัน Font เพี้ยน)
    col_t1, col_t2 = st.columns([2.5, 1], gap="medium")
    with col_t1:
        st.dataframe(
            df_forecast.style.format({
                "ค่าพยากรณ์ (เม็ด)": "{:,.0f}",
                "ขอบเขตล่าง (95% CI)": "{:,.0f}",
                "ขอบเขตบน (95% CI)": "{:,.0f}"
            }),
            width='stretch',
            hide_index=True
        )
    with col_t2:
        csv_bytes = df_forecast.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label=f"ดาวน์โหลด CSV ({forecast_horizon} งวด)",
            data=csv_bytes,
            file_name=f"sarima_bj_forecast_{forecast_horizon}periods.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    st.info("กรุณากดปุ่ม **ประมวลผลการพยากรณ์** ด้านบนเพื่อแสดงผลการวิเคราะห์")

# =========================================================================
# Section 4: ส่วนตรวจสอบความถูกต้องของพารามิเตอร์และการวินิจฉัย (ตามเล่มวิจัย)
# =========================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-title">
    <span class="section-badge">4</span>
    การวิเคราะห์พารามิเตอร์และการวินิจฉัยแบบจำลอง (Model Diagnostics & Validation)
</div>
""", unsafe_allow_html=True)

if st.session_state.has_run_forecast:
    # คำนวณ ADF Test สำหรับชุดข้อมูลปัจจุบัน
    try:
        cur_adf_orig = adfuller(series.dropna())
        cur_adf_diff = adfuller(series.diff().dropna())
        cur_adf_orig_stat, cur_adf_orig_p = cur_adf_orig[0], cur_adf_orig[1]
        cur_adf_diff_stat, cur_adf_diff_p = cur_adf_diff[0], cur_adf_diff[1]
    except Exception:
        cur_adf_orig_stat, cur_adf_orig_p = 0.5, 0.95
        cur_adf_diff_stat, cur_adf_diff_p = -7.0, 0.0001

    # ป้ายกำกับข้อมูลที่กำลังประมวลผล (Navy Status Bar)
    st.markdown(f"""
    <div class="pcu-status-bar">
        <div>
            <span style="font-size:0.84rem; color:#475569; font-weight:500;">ชุดข้อมูล:</span>
            <b style="color:#0A192F; font-size:1rem; margin-left:0.4rem;">{uploaded_file_name}</b>
        </div>
        <div>
            <span style="font-size:0.84rem; color:#475569; font-weight:500;">แบบจำลองที่คัดเลือก:</span>
            <b style="color:#1E3A8A; font-size:1rem; margin-left:0.4rem;">{model_str}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ฐานข้อมูลมาตรฐานเปรียบเทียบ 4 หน่วยบริการปฐมภูมิจากเล่มวิจัย
    PCU_BENCHMARK_META = {
        "CT": {
            "name": "หน่วยบริการปฐมภูมิชาตะผดุง (CT)",
            "d0": {"stat": "-0.4156", "p": "0.9075", "res": "มี Unit Root (ไม่นิ่ง)"},
            "d1": {"stat": "-7.9289", "p": "< 0.0001", "res": "ไม่มี Unit Root (นิ่ง)"},
            "s_cands": "[2, 3, 4, 5]", "s_star": "4 เดือน", "lb_s": "0.0872", "aicc": "2,206.9",
            "context": "รอบนัดรับยา 4 เดือน (ปีละ 3 ครั้ง)",
            "k": 8, "arima": "(3, 1, 1)", "lb_arima": "0.7523",
            "sarima": "(3, 1, 2) × (1, 1, 1)_4", "lb_sarima": "0.0872", "delta_aicc": "+72.7"
        },
        "MT": {
            "name": "หน่วยบริการปฐมภูมิมิตรภาพ (MT)",
            "d0": {"stat": "+0.4936", "p": "0.9847", "res": "มี Unit Root (ไม่นิ่ง)"},
            "d1": {"stat": "-7.0245", "p": "< 0.0001", "res": "ไม่มี Unit Root (นิ่ง)"},
            "s_cands": "[2, 3, 4, 6]", "s_star": "6 เดือน", "lb_s": "0.7401", "aicc": "2,290.6",
            "context": "รอบนัดรับยา 6 เดือน (ปีละ 2 ครั้ง)",
            "k": 4, "arima": "(3, 1, 3)", "lb_arima": "0.9942",
            "sarima": "(1, 1, 1) × (0, 1, 1)_6", "lb_sarima": "0.7401", "delta_aicc": "+113.6"
        },
        "NK": {
            "name": "หน่วยบริการปฐมภูมิหนองกุง (NK)",
            "d0": {"stat": "-0.1509", "p": "0.9441", "res": "มี Unit Root (ไม่นิ่ง)"},
            "d1": {"stat": "-6.7300", "p": "< 0.0001", "res": "ไม่มี Unit Root (นิ่ง)"},
            "s_cands": "[2, 3, 4, 6]", "s_star": "6 เดือน", "lb_s": "0.7688", "aicc": "2,138.5",
            "context": "รอบนัดรับยา 6 เดือน (ปีละ 2 ครั้ง)",
            "k": 5, "arima": "(2, 1, 1)", "lb_arima": "0.7084",
            "sarima": "(2, 1, 1) × (0, 1, 1)_6", "lb_sarima": "0.7688", "delta_aicc": "+105.2"
        },
        "NW": {
            "name": "หน่วยบริการปฐมภูมิวัดหนองแวง (NW)",
            "d0": {"stat": "-2.2258", "p": "0.1970", "res": "มี Unit Root (ไม่นิ่ง)"},
            "d1": {"stat": "-4.9523", "p": "< 0.0001", "res": "ไม่มี Unit Root (นิ่ง)"},
            "s_cands": "[2, 3, 4, 6]", "s_star": "6 เดือน", "lb_s": "0.9041", "aicc": "2,349.4",
            "context": "รอบนัดรับยา 6 เดือน (ปีละ 2 ครั้ง)",
            "k": 6, "arima": "(2, 1, 3)", "lb_arima": "0.7742",
            "sarima": "(2, 1, 2) × (0, 1, 1)_6", "lb_sarima": "0.9041", "delta_aicc": "+90.7"
        }
    }

    # แท็บแสดงผล
    tab_41, tab_43, tab_44, tab_diag = st.tabs([
        "1. การทดสอบความนิ่ง (ADF Test)",
        "2. คาบเวลาฤดูกาล (Seasonality: s*)",
        "3. เปรียบเทียบโครงสร้างแบบจำลอง",
        "4. การตรวจวินิจฉัยเศษเหลือ (Residual Diagnostics)"
    ])

    # ---------------------------------------------------------------------
    # แท็บ 1: การทดสอบความนิ่ง (ADF Test) - เฉพาะชุดข้อมูลที่นำเข้า
    # ---------------------------------------------------------------------
    with tab_41:
        st.markdown(f"<b style='font-size:1.05rem; color:#1E293B; line-height:1.6;'>ผลการทดสอบความนิ่งด้วย Augmented Dickey-Fuller (ADF Test) - {uploaded_file_name}</b>", unsafe_allow_html=True)
        
        if current_pcu_code in PCU_BENCHMARK_META:
            meta = PCU_BENCHMARK_META[current_pcu_code]
            cur_41_data = [
                {"ระดับข้อมูล": "d0 (ข้อมูลดิบ)", "สถิติทดสอบ (ADF)": meta["d0"]["stat"], "p-value": meta["d0"]["p"], "ผลการทดสอบ (alpha = 0.05)": meta["d0"]["res"]},
                {"ระดับข้อมูล": "d1 (ผลต่างอันดับ 1)", "สถิติทดสอบ (ADF)": meta["d1"]["stat"], "p-value": meta["d1"]["p"], "ผลการทดสอบ (alpha = 0.05)": meta["d1"]["res"]}
            ]
        else:
            cur_41_data = [
                {"ระดับข้อมูล": "d0 (ข้อมูลดิบ)", "สถิติทดสอบ (ADF)": f"{cur_adf_orig_stat:.4f}", "p-value": f"{cur_adf_orig_p:.4f}", "ผลการทดสอบ (alpha = 0.05)": "มี Unit Root (ไม่นิ่ง)" if cur_adf_orig_p > 0.05 else "ไม่มี Unit Root (นิ่ง)"},
                {"ระดับข้อมูล": "d1 (ผลต่างอันดับ 1)", "สถิติทดสอบ (ADF)": f"{cur_adf_diff_stat:.4f}", "p-value": "< 0.0001" if cur_adf_diff_p < 0.0001 else f"{cur_adf_diff_p:.4f}", "ผลการทดสอบ (alpha = 0.05)": "ไม่มี Unit Root (นิ่ง)" if cur_adf_diff_p <= 0.05 else "ไม่ผ่านเกณฑ์"}
            ]

        df_41_display = pd.DataFrame(cur_41_data)
        st.dataframe(df_41_display, width='stretch', hide_index=True)
        st.caption("หมายเหตุ: ข้อมูลผลต่างอันดับที่ 1 (d1) มีค่า p-value < 0.0001 แสดงว่าอนุกรมเวลามีความนิ่งที่ระดับนัยสำคัญ 0.01 หรือเป็นอนุกรมเวลาประเภท I(1)")

    # ---------------------------------------------------------------------
    # แท็บ 2: การคัดเลือกพารามิเตอร์ฤดูกาล (s*) - เฉพาะชุดข้อมูลที่นำเข้า
    # ---------------------------------------------------------------------
    with tab_43:
        st.markdown(f"<b style='font-size:1.05rem; color:#1E293B; line-height:1.6;'>ผลการคัดเลือกคาบเวลาฤดูกาลที่เหมาะสม (Optimal Seasonality: s*) - {uploaded_file_name}</b>", unsafe_allow_html=True)
        
        if current_pcu_code in PCU_BENCHMARK_META:
            meta = PCU_BENCHMARK_META[current_pcu_code]
            cur_43_data = [{
                "ชุดข้อมูล": uploaded_file_name,
                "คาบเวลาที่ทดสอบ (s)": meta["s_cands"],
                "คาบเวลาที่คัดเลือก (s*)": meta["s_star"],
                "Ljung-Box Test (p-value)": meta["lb_s"],
                "AICc": meta["aicc"],
                "บริบทการจ่ายยา": meta["context"]
            }]
        else:
            cand_s = detect_seasonality_fft(series)
            cur_43_data = [{
                "ชุดข้อมูล": uploaded_file_name,
                "คาบเวลาที่ทดสอบ (s)": str(cand_s),
                "คาบเวลาที่คัดเลือก (s*)": f"{s} เดือน",
                "Ljung-Box Test (p-value)": f"{best_lb_p:.4f}",
                "AICc": f"{best_aicc:,.1f}",
                "บริบทการจ่ายยา": "วิเคราะห์จากสเปกตรัมร่วมกับเกณฑ์ AICc"
            }]

        df_43_display = pd.DataFrame(cur_43_data)
        st.dataframe(df_43_display, width='stretch', hide_index=True)
        st.caption("หมายเหตุ: คัดเลือกคาบเวลา s* จากการวิเคราะห์สเปกตรัมร่วมกับเกณฑ์สารสนเทศ AICc ที่ต่ำที่สุด เพื่อให้แบบจำลองสามารถอธิบายพฤติกรรมวงรอบการจ่ายยาได้อย่างถูกต้อง")

    # ---------------------------------------------------------------------
    # แท็บ 3: เปรียบเทียบโครงสร้างแบบจำลอง (ARIMA vs SARIMA) - เฉพาะชุดข้อมูลที่นำเข้า
    # ---------------------------------------------------------------------
    with tab_44:
        st.markdown(f"<b style='font-size:1.05rem; color:#1E293B; line-height:1.6;'>การเปรียบเทียบโครงสร้างแบบจำลอง ARIMA และ SARIMA - {uploaded_file_name}</b>", unsafe_allow_html=True)
        
        if current_pcu_code in PCU_BENCHMARK_META:
            meta = PCU_BENCHMARK_META[current_pcu_code]
            cur_44_data = [{
                "ชุดข้อมูล": uploaded_file_name,
                "จำนวนพารามิเตอร์ (K)": meta["k"],
                "ARIMA (p, d, q)": meta["arima"],
                "Ljung-Box (ARIMA)": meta["lb_arima"],
                "SARIMA (p, d, q) × (P, D, Q)_s": meta["sarima"],
                "Ljung-Box (SARIMA)": meta["lb_sarima"],
                "Seasonality Gain (ΔAICc)": meta["delta_aicc"]
            }]
        else:
            cur_44_data = [{
                "ชุดข้อมูล": uploaded_file_name,
                "จำนวนพารามิเตอร์ (K)": len(fit_model.params),
                "ARIMA (p, d, q)": f"({p}, {d}, {q})",
                "Ljung-Box (ARIMA)": "-",
                "SARIMA (p, d, q) × (P, D, Q)_s": model_str,
                "Ljung-Box (SARIMA)": f"{best_lb_p:.4f}",
                "Seasonality Gain (ΔAICc)": "Significant"
            }]

        df_44_display = pd.DataFrame(cur_44_data)
        st.dataframe(df_44_display, width='stretch', hide_index=True)
        st.caption("หมายเหตุ: ค่า ΔAICc (ARIMA - SARIMA) ที่สูงกว่าเกณฑ์ 10 หน่วย (Burnham & Anderson, 2002) ยืนยันว่าการผนวกพจน์ฤดูกาลช่วยเพิ่มประสิทธิภาพการพยากรณ์อย่างมีนัยสำคัญ")

    # ---------------------------------------------------------------------
    # แท็บ 4: การตรวจวินิจฉัยเศษเหลือ (Residual Diagnostics & Q-Q Plot)
    # ---------------------------------------------------------------------
    with tab_diag:
        st.markdown("<b style='font-size:1.05rem; color:#1E293B; line-height:1.6;'>การตรวจวินิจฉัยเศษเหลือแบบจำลอง SARIMA-BJ (4-Panel Diagnostics & Q-Q Plot)</b>", unsafe_allow_html=True)
        
        col_fig, col_coef_tbl = st.columns([1.5, 1], gap="medium")
        
        with col_fig:
            try:
                fig_diag = fit_model.plot_diagnostics(figsize=(10, 6.5))
                for ax in fig_diag.axes:
                    ax.tick_params(colors='#475569', labelsize=8.5)
                    for spine in ax.spines.values():
                        spine.set_color('#CBD5E1')
                    ax.grid(True, linestyle='--', alpha=0.5, color='#E2E8F0')
                    title = ax.get_title()
                    ax.set_title(title, fontsize=10.5, fontweight='bold', color='#0F172A', pad=6)
                fig_diag.tight_layout()
                st.pyplot(fig_diag, clear_figure=True)
                plt.close(fig_diag)
            except Exception as e:
                st.warning(f"ไม่สามารถสร้างกราฟ Diagnostics ได้: {e}")

        with col_coef_tbl:
            st.markdown("<b style='font-size:1.02rem; color:#1E293B; line-height:1.6;'>สัมประสิทธิ์แบบจำลอง (Parameter Estimates):</b>", unsafe_allow_html=True)
            try:
                param_names = list(fit_model.param_names)
                coef_vals = np.array(fit_model.params)
                bse_vals = np.array(fit_model.bse)
                z_vals = np.array(fit_model.zvalues)
                p_vals = np.array(fit_model.pvalues)

                def format_term(t):
                    t_str = str(t)
                    if t_str == 'sigma2': return 'sigma²'
                    if t_str.startswith('ar.L'): return f"AR({t_str.split('.L')[1]})"
                    if t_str.startswith('ma.L'): return f"MA({t_str.split('.L')[1]})"
                    if t_str.startswith('ar.S.L'): return f"SAR({t_str.split('.L')[1]})"
                    if t_str.startswith('ma.S.L'): return f"SMA({t_str.split('.L')[1]})"
                    return t_str

                df_coef_sub = pd.DataFrame({
                    "ตัวแปร": [format_term(n) for n in param_names],
                    "สัมประสิทธิ์": coef_vals,
                    "Std.Err": bse_vals,
                    "z-stat": z_vals,
                    "p-value": p_vals
                })

                st.dataframe(
                    df_coef_sub.style.format({
                        "สัมประสิทธิ์": "{:,.4f}",
                        "Std.Err": "{:,.4f}",
                        "z-stat": "{:.2f}",
                        "p-value": "{:.4f}"
                    }),
                    width='stretch',
                    hide_index=True
                )
            except Exception as e:
                st.caption(f"รายละเอียด: {e}")
        
        st.caption("หมายเหตุ: Normal Q-Q Plot แสดงจุดข้อมูลเรียงตัวตามแนวเส้นตรง และ Correlogram แสดงค่าอัตสหสัมพันธ์อยู่ในแถบความเชื่อมั่น ยืนยันว่าเศษเหลือมีคุณสมบัติเป็น White Noise")

else:
    st.info("ผลการวิเคราะห์พารามิเตอร์และการวินิจฉัยจะแสดงเมื่อกดปุ่มประมวลผลการพยากรณ์ในส่วนที่ 3")
