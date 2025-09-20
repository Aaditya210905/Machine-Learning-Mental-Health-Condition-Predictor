import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,precision_score, recall_score, roc_auc_score
import joblib
import os
from sklearn.model_selection import train_test_split
from preprocessing import build_preprocessor

# 1. Load dataset
df = pd.read_csv("data/mental_health_india.csv")

# 2. Prepare features and target
X = df.drop('treatment', axis=1)
y = df['treatment']

# 3. Encode target labels
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# 4. Remove duplicate rows
df.drop_duplicates(inplace=True)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=41)

# 6. Preprocess data
X_train, X_test, y_train, y_test, ordinal_encoder, ohe,lda = build_preprocessor(X_train, X_test, y_train, y_test)

# 7. Train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# 8. Create a Function to Evaluate Model
def evaluate_model(test, predicted):
    accuracy = accuracy_score(test, predicted) # Calculate Accuracy
    cm = confusion_matrix(test, predicted) # Confusion Matrix
    cr = classification_report(test, predicted) # Classification Report
    precision = precision_score(test, predicted) # Calculate Precision
    recall = recall_score(test, predicted) # Calculate Recall
    roc_auc = roc_auc_score(test, predicted)
    return accuracy, cm, cr, precision, recall, roc_auc

y_pred = model.predict(X_test)
accuracy, cm, cr, precision, recall, roc_auc = evaluate_model(y_test, y_pred)
print("Accuracy", accuracy)
print("Confusion Matrix\n", cm)
print("Classification Report\n", cr)
print("Precision", precision)
print("Recall", recall)
print("ROC AUC", roc_auc)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save trained model and encoders
joblib.dump(model, "models/naive_bayes_model.pkl")
joblib.dump(ordinal_encoder, 'models/ordinal_encoder.pkl')
joblib.dump(ohe, 'models/ohe_encoder.pkl')
joblib.dump(lda, 'models/lda_transformer.pkl')
joblib.dump(le_target, 'models/label_encoder.pkl')

