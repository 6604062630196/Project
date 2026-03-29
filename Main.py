import streamlit as st

st.set_page_config(page_title="Health AI", page_icon="🏥", layout="wide")

st.title(" ระบบ AI ทำนายความเสี่ยงโรคเบาหวาน")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.info("**เกี่ยวกับโมเดล**\n\nเรียนรู้วิธีการทำงานของ ML และ Neural Network")
with col2:
    st.success(" **ทดสอบโมเดล**\n\nกรอกข้อมูลสุขภาพเพื่อทำนายผล")

st.markdown("### เลือกหน้าจากแถบด้านซ้าย")

import streamlit as st

st.title("📘 เกี่ยวกับโมเดล Machine Learning")
st.markdown("---")

st.header("1. Dataset ที่ใช้")
st.write("ใช้ **Pima Indians Diabetes Dataset** จาก Kaggle มีข้อมูล 768 ตัวอย่าง 8 features")

st.header("2. การเตรียมข้อมูล")
st.markdown("""
- แทนค่า 0 ที่เป็นไปไม่ได้ด้วย **Median**
- Normalize ด้วย **StandardScaler**
- แบ่ง Train 80% / Test 20%
""")

st.header("3. อัลกอริทึม — Ensemble (Voting Classifier)")
st.markdown("""
รวม 3 โมเดลเข้าด้วยกัน:
| โมเดล | หลักการ |
|-------|---------|
| Random Forest | รวมหลาย Decision Tree |
| SVM | หาเส้นแบ่งที่ดีที่สุด |
| KNN | ดูจาก K เพื่อนบ้านที่ใกล้ที่สุด |
""")

st.header("4. ผลลัพธ์")
st.success("Accuracy ประมาณ 77-80%")

import streamlit as st

st.title("🧠 เกี่ยวกับโมเดล Neural Network")
st.markdown("---")

st.header("1. โครงสร้างโมเดล")
st.markdown("""
""")

st.header("2. การเทรน")
st.markdown("""
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Epochs:** 50
- **Batch Size:** 32
""")

st.header("3. ผลลัพธ์")
st.success("Accuracy ประมาณ 75-79%")

import streamlit as st
import pickle
import numpy as np

st.title("🧪 ทดสอบโมเดล Machine Learning")
st.markdown("---")

@st.cache_resource
def load_model():
    with open('models/ensemble.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

st.subheader("กรอกข้อมูลสุขภาพ")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("จำนวนครั้งที่ตั้งครรภ์", 0, 20, 1)
    glucose     = st.slider("ระดับน้ำตาลในเลือด", 50, 200, 100)
    bp          = st.slider("ความดันโลหิต", 40, 130, 70)
    skin        = st.slider("ความหนาผิวหนัง", 0, 100, 20)

with col2:
    insulin = st.slider("ระดับอินซูลิน", 0, 500, 80)
    bmi     = st.number_input("BMI", 10.0, 70.0, 25.0)
    dpf     = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age     = st.slider("อายุ", 1, 100, 30)

if st.button("🔍 ทำนายผล", type="primary"):
    data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0]

    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ มีความเสี่ยงเป็นเบาหวาน — ความน่าจะเป็น {prob[1]*100:.1f}%")
    else:
        st.success(f"✅ ความเสี่ยงต่ำ — ความน่าจะเป็นปกติ {prob[0]*100:.1f}%")


import streamlit as st
import numpy as np
import pickle
from tensorflow import keras

st.title("🧠 ทดสอบโมเดล Neural Network")
st.markdown("---")

@st.cache_resource
def load_model():
    model = keras.models.load_model('models/nn_model.h5')
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

st.subheader("กรอกข้อมูลสุขภาพ")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("จำนวนครั้งที่ตั้งครรภ์", 0, 20, 1)
    glucose     = st.slider("ระดับน้ำตาลในเลือด", 50, 200, 100)
    bp          = st.slider("ความดันโลหิต", 40, 130, 70)
    skin        = st.slider("ความหนาผิวหนัง", 0, 100, 20)

with col2:
    insulin = st.slider("ระดับอินซูลิน", 0, 500, 80)
    bmi     = st.number_input("BMI", 10.0, 70.0, 25.0)
    dpf     = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age     = st.slider("อายุ", 1, 100, 30)

if st.button("🔍 ทำนายผล", type="primary"):
    data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    data_scaled = scaler.transform(data)
    prob = model.predict(data_scaled)[0][0]

    st.markdown("---")
    if prob >= 0.5:
        st.error(f"⚠️ มีความเสี่ยงเป็นเบาหวาน — ความน่าจะเป็น {prob*100:.1f}%")
    else:
        st.success(f"✅ ความเสี่ยงต่ำ — ความน่าจะเป็นปกติ {(1-prob)*100:.1f}%")
