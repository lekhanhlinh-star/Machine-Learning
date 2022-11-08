import dill
import pipelineDataNew
import streamlit as st
import encoder_data
import pandas as pd
filename='rfc_reg_model.dill'
with open(filename, "rb") as f:
    randomForest_models = dill.load(f)
page_bg_img="""
<style>

[data-testid="stAppViewContainer"]{
    background-image:url(""D:\MACHINE LEARING AT UNIVERSITY\pixabay_brain-stroke_1200.jpg"");
    
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)

    
feature=['age',
 'hypertension',
 'heart_disease',
 'ever_married',
 'avg_glucose_level',
 'bmi']


with st.sidebar:
    st.title("Stroke detection app")
    st.image("./pixabay_brain-stroke_1200.jpg")
    df = pd.DataFrame.from_dict({
        'age':[float(st.slider("age",0.080000, 100.000000, 25.0))],
    'hypertension':[st.radio(
        "hypertension",
        ('Yes',"No"))],
    'heart_disease':[st.radio(
        "Heart_disease",
        ('Yes',"No"))],
    'ever_married':[st.radio(
        "ever_married",
        ('Yes',"No"))],
    'avg_glucose_level':[st.slider("avg_glucose_level",0.0, 271.740000	, 60.0)],
    'bmi':[st.slider("BMI",14.000000, 48.900000, 20.0)],


})
st.title("Stroke Detection App")
# print(df)
# st.dataframe(df)
new_data=encoder_data.new_data_num(df)
# st.write(new_data)
with open("KNN_model.dill", "rb") as f:
    model = dill.load(f)

# st.write(model.predict(new_data)[0])
st.write("Probability stroke: ",model.predict_proba(new_data)[0][1]*100,"%")
# st.write("Probability stroke: ",model.predict_proba(new_data)[0])\\\\\\\\\\\\\
st.image("./00.-Machine-Learing.png")




