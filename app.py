import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.markdown("""
<style>
h1 { 
    text-align: center; 
    background: linear-gradient(120deg, #667eea, #764ba2); 
    background-clip: text; 
    -webkit-background-clip: text; 
    color: transparent; 
}
.stButton > button { 
    background: linear-gradient(90deg, #667eea, #764ba2); 
    color: white; 
    border: none; 
    padding: 0.75rem; 
    border-radius: 50px; 
    width: 100%; 
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎓 Student Academic Performance Predictor</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('student_performance_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

model, preprocessor = load_models()

if model is not None:
    st.success("✅ Models loaded successfully!")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            hours_studied = st.slider("Hours Studied per Week", 1, 44, 20)
            attendance = st.slider("Attendance Rate (%)", 60, 100, 80)
            previous_scores = st.slider("Previous Exam Score (%)", 50, 100, 75)
            tutoring_sessions = st.slider("Tutoring Sessions (per month)", 0, 8, 2)
            sleep_hours = st.slider("Sleep Hours (per day)", 4, 10, 7)
            physical_activity = st.slider("Physical Activity (hours/week)", 0, 6, 3)
        
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            school_type = st.selectbox("School Type", ["Public", "Private"])
            parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
            access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
            motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
            teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
        
        submitted = st.form_submit_button("🚀 Predict Exam Score")
        
        if submitted:
            input_data = pd.DataFrame([{
                'Hours_Studied': hours_studied,
                'Attendance': attendance,
                'Parental_Involvement': parental_involvement,
                'Access_to_Resources': access_to_resources,
                'Extracurricular_Activities': 'No',
                'Sleep_Hours': sleep_hours,
                'Previous_Scores': previous_scores,
                'Motivation_Level': motivation_level,
                'Internet_Access': 'Yes',
                'Tutoring_Sessions': tutoring_sessions,
                'Family_Income': 'Medium',
                'Teacher_Quality': teacher_quality,
                'School_Type': school_type,
                'Peer_Influence': 'Neutral',
                'Physical_Activity': physical_activity,
                'Learning_Disabilities': 'No',
                'Parental_Education_Level': 'College',
                'Distance_from_Home': 'Near',
                'Gender': gender
            }])
            
            try:
                X_processed = preprocessor.transform(input_data)
                prediction = model.predict(X_processed)[0]
                
                st.balloons()
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea20, #764ba220); border-radius: 20px; padding: 2rem; text-align: center;'>
                    <h2>📈 Predicted Exam Score</h2>
                    <h1 style='font-size: 4rem; color: #667eea;'>{prediction:.1f}</h1>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
else:
    st.warning("⚠️ Model files not found. Please ensure .pkl files are in this directory.")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Student Performance Prediction System</p>", unsafe_allow_html=True)
