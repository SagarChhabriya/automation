# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib
import streamlit as st

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False
if 'input_data' not in st.session_state:
    st.session_state['input_data'] = {}

# Step 1: Data Preprocessing
def preprocess_data(df, target_column):
    """
    Preprocess the dataset: handle missing values, encode categorical variables, and scale numerical features.
    """
    try:
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
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None, None, None

# Step 2: Exploratory Data Analysis (EDA)
def perform_eda(df):
    """
    Perform exploratory data analysis: visualize data, detect outliers, and analyze relationships.
    """
    try:
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

        # Correlation matrix (only for numerical columns)
        st.write("#### Correlation Matrix")
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_columns) > 1:  # Ensure there are at least two numerical columns
            fig, ax = plt.subplots()
            sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numerical columns to compute correlation matrix.")
    except Exception as e:
        st.error(f"Error during EDA: {e}")

# Step 3: Model Building
def build_model(X, y, preprocessor, problem_type):
    """
    Build and evaluate machine learning models using cross-validation and hyperparameter tuning.
    """
    try:
        # Encode the target variable if it's categorical (for classification problems)
        if problem_type == "classification" and y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            st.write("#### Target Variable Encoded Successfully")

        # Define models and hyperparameters
        if problem_type == "classification":
            models = {
                'Logistic Regression': (LogisticRegression(), {'classifier__C': [0.1, 1, 10]}),
                'Random Forest': (RandomForestClassifier(), {'classifier__n_estimators': [50, 100, 200]}),
                'XGBoost': (XGBClassifier(), {'classifier__learning_rate': [0.01, 0.1, 0.2]})
            }
        else:
            models = {
                'Linear Regression': (LinearRegression(), {}),
                'Random Forest': (RandomForestRegressor(), {'classifier__n_estimators': [50, 100, 200]}),
                'XGBoost': (XGBRegressor(), {'classifier__learning_rate': [0.01, 0.1, 0.2]})
            }

        best_model = None
        best_score = -np.inf if problem_type == "regression" else 0

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for model_name, (model, params) in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            # Hyperparameter tuning
            grid_search = GridSearchCV(pipeline, params, cv=5, scoring='roc_auc' if problem_type == "classification" else 'r2')
            grid_search.fit(X_train, y_train)

            # Evaluate model
            y_pred = grid_search.predict(X_test)
            if problem_type == "classification":
                score = roc_auc_score(y_test, y_pred)
            else:
                score = r2_score(y_test, y_pred)

            st.write(f"#### {model_name}")
            st.write(f"Best Parameters: {grid_search.best_params_}")
            st.write(f"Score: {score:.4f}")

            if (problem_type == "classification" and score > best_score) or (problem_type == "regression" and score > best_score):
                best_score = score
                best_model = grid_search.best_estimator_

        return best_model
    except Exception as e:
        st.error(f"Error during model building: {e}")
        return None

# Step 4: Save and Load Model
def save_model(model, filename):
    """
    Save the trained model and preprocessing steps to a file.
    """
    try:
        joblib.dump(model, filename)
        st.success(f"Model saved successfully as {filename}")
    except Exception as e:
        st.error(f"Error saving model: {e}")

def load_model(filename):
    """
    Load the trained model from a file.
    """
    try:
        model = joblib.load(filename)
        st.success(f"Model loaded successfully from {filename}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Step 5: Streamlit App for Predictions
def streamlit_app():
    """
    Build a Streamlit app to load the model and make predictions.
    """
    st.title("Generalized Machine Learning Pipeline with Streamlit")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview")
            st.write(df.head())

            # Select target column
            target_column = st.selectbox("Select the target column", df.columns)

            # Select problem type
            problem_type = st.radio("Select the problem type", ["classification", "regression"])

            # Preprocess data
            preprocessor, X, y = preprocess_data(df, target_column)

            if preprocessor is not None:
                # Perform EDA
                perform_eda(df)

                # Build and evaluate model
                if st.button("Train Model"):
                    model = build_model(X, y, preprocessor, problem_type)
                    if model is not None:
                        save_model(model, 'best_model.pkl')
                        st.session_state['model_trained'] = True

                # Load model and make predictions
                if st.session_state['model_trained']:
                    st.write("### Make Predictions")
                    for col in X.columns:
                        st.session_state['input_data'][col] = st.text_input(f"Enter value for {col}", st.session_state['input_data'].get(col, "0"))

                    if st.button("Predict"):
                        model = load_model('best_model.pkl')
                        if model is not None:
                            input_df = pd.DataFrame([st.session_state['input_data']])
                            prediction = model.predict(input_df)
                            st.write(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"Error processing dataset: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()