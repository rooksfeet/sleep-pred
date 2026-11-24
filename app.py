import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# read the data
df = pd.read_csv("student_sleep.csv")
st.title("Sleep Quality Predictor")
st.write("CBSE Class 12 AI Project. Linear Regression trained on 500 students (Kaggle).")
# inputs and output
X = df[["Study_Hours", "Screen_Time", "Caffeine_Intake", "Physical_Activity"]]
y = df["Sleep_Quality"]

# train the model
model = LinearRegression()
model.fit(X, y)

st.title("Sleep Quality Predictor")

study = st.slider("Study hours", 0.0, 12.0, 6.0, 0.5)
screen = st.slider("Screen time (hours)", 1.0, 4.0, 2.5, 0.5)
caffeine = st.slider("Caffeine (cups)", 0, 5, 2)
activity = st.slider("Physical activity (minutes)", 0, 120, 60, 10)

if st.button("Predict"):
    new = pd.DataFrame([[study, screen, caffeine, activity]], columns=X.columns)
    result = model.predict(new)[0]

    # keep result between 1 and 10
    if result > 10:
        result = 10
    if result < 1:
        result = 1

    st.subheader("Predicted sleep quality: " + str(round(result, 1)) + " / 10")
