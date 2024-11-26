import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Exercise Calorie Predictor",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3rem;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    h1 {
        color: #FF4B4B;
        font-size: 2.5rem !important;
        font-weight: bold !important;
        margin-bottom: 2rem !important;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stNumberInput>div>div>input {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model and encoder
@st.cache_resource
def load_model():
    try:
        model = joblib.load('calorie_predictor.joblib')
        gender_encoder = joblib.load('gender_encoder.joblib')
        return model, gender_encoder
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def create_gauge_chart(value, intensity, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        number = {'suffix': " kcal", 'font': {'size': 24, 'color': color}},
        gauge = {
            'axis': {'range': [0, 400], 'tickwidth': 1, 'tickcolor': color},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': color,
            'steps': [
                {'range': [0, 100], 'color': '#E8F5E9'},
                {'range': [100, 200], 'color': '#FFF3E0'},
                {'range': [200, 400], 'color': '#FFEBEE'}
            ]
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': color, 'family': "Arial"}
    )
    return fig

def main():
    st.markdown("<h1>Exercise Calorie Predictor 🏃‍♂️</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Get accurate predictions for calories burned during your workout session</p>", unsafe_allow_html=True)
    
    model, gender_encoder = load_model()
    
    # Create input form with better layout
    with st.form("prediction_form"):
        st.markdown("### Enter Your Details")
        
        # Create three columns for better distribution
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Personal Info")
            gender = st.selectbox('Gender', ['male', 'female'])
            age = st.number_input('Age', min_value=15, max_value=80, value=30)
            height = st.number_input('Height (cm)', min_value=130, max_value=230, value=170)
        
        with col2:
            st.markdown("#### Body Metrics")
            weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)
            body_temp = st.number_input('Body Temperature (°C)', min_value=35.0, max_value=42.0, value=38.5, step=0.1)
        
        with col3:
            st.markdown("#### Exercise Data")
            duration = st.number_input('Exercise Duration (min)', min_value=1, max_value=60, value=15)
            heart_rate = st.number_input('Heart Rate (bpm)', min_value=60, max_value=200, value=110)
        
        st.markdown("")  # Add some spacing
        submit_button = st.form_submit_button(label='Calculate Calories Burned 🔥')
        
        if submit_button and model is not None and gender_encoder is not None:
            # Prepare input data
            gender_encoded = gender_encoder.transform([gender])[0]
            input_data = pd.DataFrame({
                'Gender': [gender_encoded],
                'Age': [age],
                'Height': [height],
                'Weight': [weight],
                'Duration': [duration],
                'Heart_Rate': [heart_rate],
                'Body_Temp': [body_temp]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Determine intensity and color
            if prediction < 100:
                intensity = "Light Exercise"
                color = "#4CAF50"  # Green
            elif prediction < 200:
                intensity = "Moderate Exercise"
                color = "#FF9800"  # Orange
            else:
                intensity = "Intense Exercise"
                color = "#F44336"  # Red
            
            # Display results in a nice box
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            
            # Create two columns for results
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown(f"### Estimated Calories Burned")
                st.markdown(f"<h2 style='color: {color}'>{prediction:.1f} kcal</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: {color}; font-size: 1.2rem;'><strong>Intensity:</strong> {intensity}</p>", unsafe_allow_html=True)
                
                # Add exercise recommendations based on intensity
                st.markdown("#### 💡 Recommendations")
                if prediction < 100:
                    st.markdown("- Perfect for warm-up or recovery")
                    st.markdown("- Good for beginners")
                    st.markdown("- Consider increasing duration for more benefits")
                elif prediction < 200:
                    st.markdown("- Great for fat burning")
                    st.markdown("- Sustainable workout intensity")
                    st.markdown("- Good for cardiovascular health")
                else:
                    st.markdown("- High-intensity workout")
                    st.markdown("- Great for athletic performance")
                    st.markdown("- Remember to stay hydrated")
            
            with res_col2:
                # Display gauge chart
                fig = create_gauge_chart(prediction, intensity, color)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
