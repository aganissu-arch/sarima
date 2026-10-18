# ระบบพยากรณ์ความต้องการยาเวชภัณฑ์ (SARIMA-BJ Engine)
### สำหรับหน่วยบริการปฐมภูมิ (Primary Care Units: PCUs)

เว็บแอปพลิเคชันสนับสนุนการตัดสินใจบริหารจัดการคลังยาและเวชภัณฑ์ พัฒนาด้วย **Streamlit** และ **SARIMA-BJ (Seasonal AutoRegressive Integrated Moving Average with Box-Jenkins Methodology & FFT Spectral Analysis)**

---

## 🚀 คุณสมบัติเด่น (Features)

1. **การนำเข้าข้อมูลยืดหยุ่น (Universal Data Ingestion):**
   - รองรับไฟล์ CSV ทุกรูปแบบ (Comma, Semicolon, Tab, UTF-8, UTF-8-BOM, CP874/TIS-620)
   - ตรวจจับคอลัมน์ความต้องการยาอัตโนมัติ
2. **การวิเคราะห์คาบเวลาฤดูกาล (Two-Tier Seasonality Detection):**
   - ขั้นที่ 1: วิเคราะห์สเปกตรัมเพอริโอโดแกรม (FFT Periodogram) เพื่อหาคาบเวลาเด่น
   - ขั้นที่ 2: ปรับจูนและคัดเลือกคาบเวลา `s*` ที่ดีที่สุดด้วยเกณฑ์สารสนเทศ AICc
3. **การพยากรณ์ความต้องการยา (Demand Forecasting):**
   - พยากรณ์ล่วงหน้า 1 ถึง 12 งวด พร้อมช่วงความเชื่อมั่น 95% (Confidence Interval)
   - ดาวน์โหลดผลการพยากรณ์เป็นไฟล์ CSV (UTF-8-SIG)
4. **การตรวจวินิจฉัยแบบจำลอง (Model Diagnostics & Validation):**
   - ผลการทดสอบความนิ่งด้วย Augmented Dickey-Fuller (ADF Test)
   - ตารางเปรียบเทียบโครงสร้างแบบจำลอง ARIMA และ SARIMA พร้อมค่า Seasonality Gain (ΔAICc)
   - กราฟตรวจวินิจฉัยเศษเหลือ 4 มิติ (Standardized Residuals, Histogram/KDE, Normal Q-Q Plot, Correlogram) และตารางสัมประสิทธิ์แบบจำลอง

---

## 🛠 การติดตั้งและรันในเครื่อง (Local Setup)

```bash
# 1. ติดตั้ง Library ที่จำเป็น
pip install -r requirements.txt

# 2. เริ่มต้นการทำงานของระบบ
streamlit run app.py
```

---

## 🌐 การนำขึ้นใช้งานจริงบน Streamlit Community Cloud (Share Streamlit)

1. **อัปโหลดโปรเจกต์ขึ้น GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Streamlit Cloud deployment"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. **เปิดใช้งานบน Streamlit Community Cloud:**
   - เข้าสู่เว็บไซต์ [share.streamlit.io](https://share.streamlit.io)
   - ล็อกอินด้วยบัญชี GitHub
   - กดปุ่ม **"Create app"**
   - ระบุการตั้งค่า:
     - **Repository:** เลือก Repository ที่เพิ่งสร้าง
     - **Branch:** `main`
     - **Main file path:** `app.py`
     - **App URL:** (ตั้งชื่อ URL ตามต้องการ)
   - กดปุ่ม **"Deploy!"**

ระบบจะติดตั้ง Library จาก `requirements.txt` และแพ็กเกจฟอนต์ภาษาไทยจาก `packages.txt` โดยอัตโนมัติ พร้อมใช้งานได้ทันที

---

## 📁 โครงสร้างไฟล์สำคัญสำหรับการ Deploy

```text
├── app.py                  # สคริปต์หลักของระบบพยากรณ์ Streamlit
├── requirements.txt        # รายการ Python Libraries สำหรับ Streamlit Cloud
├── packages.txt            # รายการ Linux Packages (ฟอนต์ภาษาไทยสำหรับ Matplotlib)
├── .streamlit/
│   └── config.toml         # การตั้งค่าธีมสีกรมท่า (Navy Theme) และความปลอดภัย
├── data/                   # ชุดข้อมูลตัวอย่างสำหรับทดสอบระบบ
└── README.md               # คู่มือการใช้งานและการติดตั้ง
```
