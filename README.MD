# Mental Health Prediction Project

A machine learning project that predicts mental health treatment needs using Gaussian Naive Bayes classifier with comprehensive data preprocessing and feature engineering.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Data](#data)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project uses machine learning to predict whether an individual might need mental health treatment based on various personal and lifestyle factors. The model employs Gaussian Naive Bayes classifier with feature engineering techniques including ordinal encoding, one-hot encoding, and Linear Discriminant Analysis (LDA) for dimensionality reduction.

## 📁 Project Structure

```
mental-health-prediction/
├── data/
│   └── mental_health_india.csv          # Dataset
├── models/
│   ├── naive_bayes_model.pkl            # Trained model
│   ├── label_encoder.pkl                # Label encoder
│   ├── lda_transformer.pkl              # LDA transformer
│   ├── ohe_encoder.pkl                  # One-hot encoder
│   └── ordinal_encoder.pkl              # Ordinal encoder
├── notebook/
│   └── catboost_info/                   # CatBoost model info
│   └── code6.ipynb                      # Jupyter notebook
├── src/
│   ├── __init__.py                      # Package initializer
│   ├── predict_sample.py                # Sample prediction script
│   ├── preprocessing.py                 # Data preprocessing utilities
│   └── train_model.py                   # Model training script
└── requirements.txt                     # Project dependencies
```

## ✨ Features

- **Comprehensive Data Preprocessing**: Handles categorical variables with ordinal and one-hot encoding
- **Feature Engineering**: Implements LDA for dimensionality reduction
- **Model Evaluation**: Provides detailed performance metrics including:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
  - Precision, Recall, and ROC-AUC scores
- **Sample Prediction**: Easy-to-use prediction interface for new data
- **Model Persistence**: Saves trained models and encoders for reuse

## 🔧 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
catboost
joblib
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mental-health-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model

Run the training script to train the model on your dataset:

```bash
python src/train_model.py
```

This will:
- Load and preprocess the data
- Split into training and testing sets
- Apply feature encoding and dimensionality reduction
- Train the Gaussian Naive Bayes model
- Evaluate model performance
- Save the trained model and encoders

### Making Predictions

Use the sample prediction script to predict mental health treatment needs:

```bash
python src/predict_sample.py
```

The script includes a sample data point with features like:
- Gender, Occupation, Employment status
- Family history, Days indoors
- Stress levels, Habit changes
- Mental health history, Mood swings
- Coping struggles, Work interest
- Social weakness, Care options

### Custom Predictions

You can modify the `sample_dict` in `predict_sample.py` with your own data:

```python
sample_dict = {
    "Gender": "Male",
    "Occupation": "Student",
    "self_employed": "No",
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
```

## 📊 Model Performance

The model provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score for positive predictions
- **Recall**: Sensitivity of the model
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed breakdown of predictions
- **Classification Report**: Per-class performance metrics

## 📈 Data

The project uses the `mental_health_india.csv` dataset which contains various features related to:

- **Demographics**: Gender, Occupation, Employment status
- **Lifestyle**: Days spent indoors, Social interactions
- **Mental Health Indicators**: Stress levels, Mood swings, Coping mechanisms
- **History**: Family history, Previous mental health experiences
- **Support**: Care options, Professional interview experiences

**Target Variable**: Treatment requirement (binary classification)

## 🔄 Preprocessing Pipeline

The preprocessing pipeline includes:

1. **Data Cleaning**: Removes duplicate entries
2. **Feature Encoding**:
   - Ordinal Encoding for ordered categorical variables
   - One-Hot Encoding for nominal categorical variables
   - Label Encoding for target variable
3. **Dimensionality Reduction**: Linear Discriminant Analysis (LDA)
4. **Data Splitting**: Stratified train-test split (80-20)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- Thanks to the contributors of the mental health dataset
- Scikit-learn community for the machine learning tools
- Open source community for various utilities and libraries

---

**Note**: This model is for educational and research purposes only. It should not be used as a substitute for professional mental health diagnosis or treatment. Always consult with qualified healthcare professionals for mental health concerns.