import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# read the data
df = pd.read_csv("student_sleep.csv")

# inputs and output
X = df[["Study_Hours", "Screen_Time", "Caffeine_Intake", "Physical_Activity"]]
y = df["Sleep_Quality"]

# train the model
model = LinearRegression()
model.fit(X, y)

st.title("Sleep Quality Predictor")

study = st.number_input("Study hours", 0.0, 12.0, 6.0)
screen = st.number_input("Screen time (hours)", 1.0, 4.0, 2.5)
caffeine = st.number_input("Caffeine (cups)", 0, 5, 2)
activity = st.number_input("Physical activity (minutes)", 0, 120, 60)

if st.button("Predict"):
    new = pd.DataFrame([[study, screen, caffeine, activity]], columns=X.columns)
    result = model.predict(new)[0]

    # keep result between 1 and 10
    if result > 10:
        result = 10
    if result < 1:
        result = 1

    st.write("Predicted sleep quality:", round(result, 1), "/ 10")
