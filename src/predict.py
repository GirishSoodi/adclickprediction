import joblib
import pandas as pd

model = joblib.load("models/ctr_model.pkl")

sample = pd.DataFrame({
    "Daily Time Spent on Site":[68],
    "Age":[35],
    "Area Income":[60000],
    "Daily Internet Usage":[200],
    "Male":[1]
})

prediction = model.predict_proba(sample)[0][1]

print("Predicted CTR:", prediction)