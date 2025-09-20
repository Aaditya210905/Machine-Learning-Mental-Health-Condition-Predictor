import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def build_preprocessor(X_train=None,X_test=None,y_train=None, y_test=None,sample_df=None,is_sample=False,ordinal_encoder=None, ohe=None, lda=None):

    # Define ordinal columns + mappings
    ordinal_features = ["Days_Indoors", "Mood_Swings"]
    ordinal_mappings = [
        ["Go out Every day", "1-14 days", "15-30 days", "31-60 days", "More than 2 months"],  # Days_Indoors
        ["Low", "Medium", "High"]  # Mood_Swings
    ]

    if is_sample:
        sample_df[ordinal_features] = ordinal_encoder.transform(sample_df[ordinal_features])
        sample_df = ohe.transform(sample_df[['Growing_Stress', 'Changes_Habits', 'Mental_Health_History', 
               'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options','family_history','Coping_Struggles']])

        sample_df = lda.transform(sample_df)

        return sample_df

    else:
        X_train.drop(columns=['Timestamp'], inplace=True)
        X_test.drop(columns=['Timestamp'], inplace=True)

        # Ordinal Encoding
        ordinal_encoder = OrdinalEncoder(categories=ordinal_mappings)
        X_train[ordinal_features] = ordinal_encoder.fit_transform(X_train[ordinal_features])
        X_test[ordinal_features] = ordinal_encoder.transform(X_test[ordinal_features])

        ohe= OneHotEncoder(drop='first', sparse_output=False)
        X_train = ohe.fit_transform(X_train[['Growing_Stress', 'Changes_Habits', 'Mental_Health_History', 'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options','family_history','Coping_Struggles']])
        X_test = ohe.transform(X_test[['Growing_Stress', 'Changes_Habits', 'Mental_Health_History', 'Work_Interest', 'Social_Weakness', 'mental_health_interview', 'care_options','family_history','Coping_Struggles']])
        
        # Apply LDA for dimensionality reduction
        lda = LinearDiscriminantAnalysis(n_components=1)  # Since we have binary classification, max components is 1
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)

        return X_train, X_test, y_train, y_test, ordinal_encoder, ohe,lda