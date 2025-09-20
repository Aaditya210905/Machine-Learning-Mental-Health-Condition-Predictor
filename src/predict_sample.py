import pandas as pd
import joblib
from preprocessing import build_preprocessor

# Load model & encoder
model = joblib.load("models/naive_bayes_model.pkl")
ordinal_encoder = joblib.load('models/ordinal_encoder.pkl')
ohe = joblib.load('models/ohe_encoder.pkl')
lda = joblib.load('models/lda_transformer.pkl')
le_target = joblib.load('models/label_encoder.pkl')

sample_dict = {
    "Gender": "Male",
    "Occupation": "Student",
    "self_employed": "NO",
    "family_history": "No",
    "Days_Indoors": "Go out Every day",
    "Growing_Stress": "No",
    "Changes_Habits": "No",
    "Mental_Health_History": "No",
    "Mood_Swings": "Low",
    "Coping_Struggles": "Yes",
    "Work_Interest": "Yes",
    "Social_Weakness": "Yes",
    "mental_health_interview": "No",
    "care_options": "Not sure"
}
# Convert to DataFrame
sample_df = pd.DataFrame([sample_dict])

# Preprocess
sample_df = build_preprocessor(sample_df=sample_df, is_sample=True,ordinal_encoder=ordinal_encoder, ohe=ohe, lda=lda)

# Predict
pred = model.predict(sample_df)
pred_label = le_target.inverse_transform(pred)
print("Prediction:", pred_label[0])
