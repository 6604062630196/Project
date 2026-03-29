import streamlit as st
import numpy as np
import pickle
from tensorflow import keras

st.title("ทดสอบโมเดล Neural Network")
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
        st.error(f"มีความเสี่ยงเป็นเบาหวาน — ความน่าจะเป็น {prob*100:.1f}%")
    else:
        st.success(f"ความเสี่ยงต่ำ — ความน่าจะเป็นปกติ {(1-prob)*100:.1f}%")
