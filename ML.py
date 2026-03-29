import streamlit as st

st.title("เกี่ยวกับโมเดล Machine Learning")
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
