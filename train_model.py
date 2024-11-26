import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib
import os

# Define model file paths
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calorie_predictor.joblib')
ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gender_encoder.joblib')

# Load and prepare data
def prepare_data():
    try:
        # Use a more efficient CSV reading method with only necessary columns
        calories = pd.read_csv('calories.csv', usecols=['User_ID', 'Calories'])
        exercise = pd.read_csv('exercise.csv', usecols=['User_ID', 'Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
        
        # Merge datasets
        df = exercise.merge(calories, on='User_ID')
        
        # Encode gender more efficiently
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'].values)
        
        return df, le
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        raise

def train_model(df, le):
    try:
        # Features and target
        X = df[['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']].values
        y = df['Calories'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = XGBRegressor(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            early_stopping_rounds=10,
            eval_metric='rmse'
        )
        
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=1
        )
        
        # Save model and encoder
        print(f"\nSaving model to: {MODEL_PATH}")
        joblib.dump(model, MODEL_PATH)
        print(f"Saving encoder to: {ENCODER_PATH}")
        joblib.dump(le, ENCODER_PATH)
        
        # Verify files were saved
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            print("\nModel files saved successfully!")
            print(f"Model file size: {os.path.getsize(MODEL_PATH) / 1024:.2f} KB")
            print(f"Encoder file size: {os.path.getsize(ENCODER_PATH) / 1024:.2f} KB")
        else:
            raise Exception("Failed to save model files!")
        
        # Calculate and print accuracy
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"Training R² Score: {train_score:.4f}")
        print(f"Testing R² Score: {test_score:.4f}")
        
    except Exception as e:
        print(f"Error in training model: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Starting model training process...")
        df, le = prepare_data()
        train_model(df, le)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
