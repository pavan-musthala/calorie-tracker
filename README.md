# Exercise Calorie Predictor

This application predicts the number of calories burned during exercise based on various physical and exercise parameters.

## Features

- Interactive web interface
- Real-time predictions
- Exercise intensity feedback
- Personalized tips based on prediction

## Setup Instructions

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Input Parameters

- Gender: male/female
- Age: 15-80 years
- Height: 130-230 cm
- Weight: 30-200 kg
- Exercise Duration: 1-60 minutes
- Heart Rate: 60-200 bpm
- Body Temperature: 35-42°C

## Model Details

The application uses an XGBoost regression model trained on exercise and calorie data. The model takes into account various physical attributes and exercise metrics to predict calorie burn.

## Files

- `app.py`: Streamlit dashboard application
- `train_model.py`: Script to train and save the model
- `requirements.txt`: Required Python packages
- `calorie_predictor.joblib`: Trained model (created after training)
- `gender_encoder.joblib`: Label encoder for gender (created after training)
