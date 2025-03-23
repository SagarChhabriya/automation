import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import streamlit as st

# Step 1: Data Preprocessing
def preprocess_data(df, target_column):
    """
    Preprocess the dataset: handle missing values, encode categorical variables, and scale numerical features.
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle missing values
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor, X, y

# Step 2: Exploratory Data Analysis (EDA)
def perform_eda(df):
    """
    Perform exploratory data analysis: visualize data, detect outliers, and analyze relationships.
    """
    st.write("### Exploratory Data Analysis (EDA)")

    # Display basic statistics
    st.write("#### Summary Statistics")
    st.write(df.describe())

    # Visualize distributions
    st.write("#### Feature Distributions")
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

    # Correlation matrix
    st.write("#### Correlation Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Step 3: Model Building
def build_model(X, y, preprocessor):
    """
    Build and evaluate machine learning models using cross-validation and hyperparameter tuning.
    """
    # Define models and hyperparameters
    models = {
        'Logistic Regression': (LogisticRegression(), {'classifier__C': [0.1, 1, 10]}),
        'Random Forest': (RandomForestClassifier(), {'classifier__n_estimators': [50, 100, 200]}),
        'XGBoost': (XGBClassifier(), {'classifier__learning_rate': [0.01, 0.1, 0.2]})
    }

    best_model = None
    best_score = 0

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model_name, (model, params) in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Hyperparameter tuning
        grid_search = GridSearchCV(pipeline, params, cv=5, scoring='roc_auc')
        grid_search.fit(X_train, y_train)

        # Evaluate model
        y_pred = grid_search.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred)

        st.write(f"#### {model_name}")
        st.write(f"Best Parameters: {grid_search.best_params_}")
        st.write(f"ROC-AUC Score: {roc_auc:.4f}")

        if roc_auc > best_score:
            best_score = roc_auc
            best_model = grid_search.best_estimator_

    return best_model

# Step 4: Save and Load Model
def save_model(model, filename):
    """
    Save the trained model and preprocessing steps to a file.
    """
    joblib.dump(model, filename)

def load_model(filename):
    """
    Load the trained model from a file.
    """
    return joblib.load(filename)

# Step 5: Streamlit App for Predictions
def streamlit_app():
    """
    Build a Streamlit app to load the model and make predictions.
    """
    st.title("Machine Learning Pipeline with Streamlit")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(df.head())

        # Select target column
        target_column = st.selectbox("Select the target column", df.columns)

        # Preprocess data
        preprocessor, X, y = preprocess_data(df, target_column)

        # Perform EDA
        perform_eda(df)

        # Build and evaluate model
        if st.button("Train Model"):
            model = build_model(X, y, preprocessor)
            save_model(model, 'best_model.pkl')
            st.success("Model trained and saved successfully!")

        # Load model and make predictions
        if st.button("Load Model and Predict"):
            model = load_model('best_model.pkl')
            st.write("### Make Predictions")
            input_data = {}
            for col in X.columns:
                input_data[col] = st.text_input(f"Enter value for {col}", "0")
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            st.write(f"Prediction: {prediction[0]}")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()