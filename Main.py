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
