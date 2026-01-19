import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Pro AI Vehicle Health Advisor", layout="wide")

# --- Fixed NLP Logic with Language Support ---
def analyze_text_input(text, lang):
    text = text.lower()
    findings = []
    
    # Dictionary for Multilingual Responses
    kb = {
        "engine": {
            "en": "🔊 **Engine/Mounting:** Potential issue with engine belts or mountings detected.",
            "hi": "🔊 **इंजन/माउंटिंग:** इंजन बेल्ट या माउंटिंग में खराबी की संभावना है।"
        },
        "brake": {
            "en": "🛑 **Braking System:** Brake pads are worn out. Immediate replacement recommended.",
            "hi": "🛑 **ब्रेकिंग सिस्टम:** ब्रेक पैड्स घिस चुके हैं। इन्हें तुरंत बदलना बेहतर होगा।"
        },
        "accident": {
            "en": "⚠️ **Structural Alert:** Accident history detected. Chassis alignment check is mandatory.",
            "hi": "⚠️ **स्ट्रक्चरल अलर्ट:** एक्सीडेंट की वजह से चेसिस एलाइनमेंट चेक करवाना अनिवार्य है।"
        },
        "body": {
            "en": "🔨 **Body Work:** Dents/scratches found. Repainting needed to prevent rusting.",
            "hi": "🔨 **बॉडी वर्क:** डेंट/स्क्रैच पाए गए हैं। जंग से बचने के लिए पेंटिंग की जरूरत है।"
        },
        "battery": {
            "en": "🔋 **Electrical:** Alternator or battery voltage appears weak.",
            "hi": "🔋 **इलेक्ट्रिकल:** अल्टरनेटर या बैटरी वोल्टेज कमजोर लग रहा है।"
        }
    }

    l_code = "en" if lang == "English" else "hi"

    # Keywords Mapping
    if any(word in text for word in ["awaz", "sound", "noise", "khat", "noise"]):
        findings.append(kb["engine"][l_code])
    if any(word in text for word in ["brake", "jam", "ruk", "squeak"]):
        findings.append(kb["brake"][l_code])
    if any(word in text for word in ["accident", "thuk", "crash", "takkar"]):
        findings.append(kb["accident"][l_code])
    if any(word in text for word in ["dent", "pichak", "body", "scrach", "scratch", "paint"]):
        findings.append(kb["body"][l_code])
    if any(word in text for word in ["battery", "start", "current"]):
        findings.append(kb["battery"][l_code])
    
    default_msg = "Analysis complete based on your inputs." if lang == "English" else "आपके द्वारा दी गई जानकारी का विश्लेषण पूरा हुआ।"
    return " <br> ".join(findings) if findings else default_msg

# --- Multi-Language Logic for Main Report ---
def generate_report_text(data, lang, type="normal"):
    if lang == "English":
        if type == "normal":
            return f"""
### 👨‍🔧 Senior Expert's Diagnostic Report
**Vehicle Model:** {data['model']} | **Overall Health:** {data['health']}%

**1. Engine & Mechanical Status:**
The system is showing {data['sound']} sounds and {data['smoke']} smoke levels. Engine condition is {data['status']}.

**2. Body & Safety Analysis:**
Accident History: {data['accident']}. Body integrity is at {data['body_score']}%. 

**3. Expert Findings on Your Issues:**
{data['custom_advice']}

**4. Maintenance Roadmap:**
- **Next Service:** Within **{data['days']} days**.
- **Priority:** Focus on {data['parts']}.
- **Verdict:** {data['verdict']}
"""
        else: # Expert Fleet
            return f"""
### 📉 Fleet Analytics & Intelligence Report
**Fleet Overview:** Average health score is **{data['avg_h']}%**.
**Critical Risks:** {data['risk_pc']}% of the fleet shows degradation in {data['common_part']}.
**Market Pattern:** Models older than {data['age_limit']} months are reporting higher risk levels.
**Solution:** Bulk inspection of {data['solution']} is recommended.
"""
    else: # Hindi
        if type == "normal":
            return f"""
### 👨‍🔧 सीनियर एक्सपर्ट मैकेनिक की रिपोर्ट
**मॉडल:** {data['model']} | **हेल्थ स्कोर:** {data['health']}%

**1. इंजन की स्थिति:**
इंजन से {data['sound']} आवाज़ और {data['smoke']} धुआं देखा गया है। इंजन अभी {data['status']} स्थिति में है।

**2. बॉडी और सुरक्षा:**
एक्सीडेंट इतिहास: {data['accident']}। बॉडी कंडीशन {data['body_score']}% है।

**3. आपकी बताई समस्याओं पर रिपोर्ट:**
{data['custom_advice']}

**4. सर्विस सलाह:**
- **अगली सर्विस:** **{data['days']} दिनों** के भीतर।
- **मुख्य कार्य:** {data['parts']} पर ध्यान दें।
- **निष्कर्ष:** {data['verdict']}
"""
        else: # Expert Fleet
            return f"""
### 📉 फ्लीट इंटेलिजेंस और एनालिसिस रिपोर्ट
**सारांश:** पूरी फ्लीट का औसत स्वास्थ्य स्कोर **{data['avg_h']}%** है।
**जोखिम:** {data['risk_pc']}% वाहनों में {data['common_part']} की समस्या है।
**समाधान:** हम {data['solution']} के सामूहिक निरीक्षण की सलाह देते हैं।
"""

# --- Sidebar ---
st.sidebar.title("Car Health AI Expert")
lang = st.sidebar.selectbox("Select Report Language", ["English", "Hindi"])
mode = st.sidebar.radio("Analysis Mode", ["Normal User (Deep Check)", "Expert Mode (Fleet Upload)"])

# --- 1. NORMAL USER MODE ---
if mode == "Normal User (Deep Check)":
    st.title("🚗 Personal AI Mechanic & Body Expert")
    
    with st.expander("Step 1: Core Details", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            car_model = st.text_input("Car Model Name", "Honda City 2022")
            last_service = st.number_input("Months Since Last Service", 0, 48, 6)
        with c2:
            sound = st.selectbox("Engine Sound", ["Smooth", "Grinding", "Ticking", "Knocking"])
            smoke = st.selectbox("Exhaust Smoke", ["None", "White", "Black", "Blue"])
        with c3:
            accident_hist = st.selectbox("Accident History", ["No Accidents", "Minor Dents", "Major Collision"])
            body_cond = st.slider("Body/Paint Condition %", 0, 100, 80)

    st.subheader("📝 Step 2: Describe Dents, Accidents or Any Issues (Optional)")
    custom_input = st.text_area("Write here (Hindi/English/Hinglish)", placeholder="e.g. Brakes are noisy or had a small accident...")

    if st.button("Generate Professional Mechanic Report"):
        health = max(0, 100 - (last_service * 5) - (30 if sound != "Smooth" else 0) - (40 if accident_hist == "Major Collision" else 0))
        
        data = {
            "model": car_model, "health": health, "sound": sound, "smoke": smoke,
            "status": "Perfect" if health > 75 else "Under Stress" if health > 50 else "Damaged",
            "accident": accident_hist, "body_score": body_cond,
            "custom_advice": analyze_text_input(custom_input, lang),
            "days": int(health * 1.5), "parts": "Engine Oil & Brake Pads" if health < 60 else "Routine Filters",
            "verdict": "Your car is safe." if health > 70 else "Visit mechanic soon!"
        }
        
        st.markdown(generate_report_text(data, lang), unsafe_allow_html=True)
        
        st.write("---")
        st.progress(health/100)
        st.caption(f"Overall Mechanical Health: {health}%")

# --- 2. EXPERT MODE ---
else:
    st.title("🔬 Universal Fleet Analytics")
    uploaded_file = st.file_uploader("Upload CSV/TXT", type=['csv', 'txt'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        st.success(f"Data loaded: {len(df)} records.")
        
        cols = df.columns.tolist()
        id_col = next((c for c in cols if any(x in c.lower() for x in ['id', 'unit', 'car'])), cols[0])
        health_col = next((c for c in cols if any(x in c.lower() for x in ['health', 'score', 'condition', 'rul'])), None)
        
        if health_col:
            avg_h = int(df[health_col].mean())
            risk_pc = round((len(df[df[health_col] < 60]) / len(df)) * 100, 1)
            
            fleet_data = {
                "avg_h": avg_h, "risk_pc": risk_pc, "common_part": "Suspension & Fuel Sensors",
                "age_limit": 24, "solution": "Full Fleet Inspection"
            }
            
            st.markdown(generate_report_text(fleet_data, lang, type="fleet"))
            
            fig, ax = plt.subplots()
            ax.pie([risk_pc, 100-risk_pc], labels=['Risk', 'Healthy'], autopct='%1.1f%%', colors=['#ff4b4b', '#00cc96'])
            st.pyplot(fig)