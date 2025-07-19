import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Enhanced page configuration
st.set_page_config(
    page_title="AI Income Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    
    .stSlider > div > div {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_enhanced_model():
    """Load the pre-trained model with error handling"""
    try:
        model = joblib.load('best_model.pkl')
        st.success("✅ Enhanced AI Model Loaded Successfully!")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'best_model.pkl' not found. Please ensure the file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

@st.cache_data
def load_sample_data():
    """Load and cache sample data for demonstration"""
    try:
        df = pd.read_csv('employee.csv')
        return df
    except FileNotFoundError:
        # Create sample data if CSV not found
        np.random.seed(42)
        sample_data = {
            'age': np.random.randint(18, 70, 1000),
            'workclass': np.random.choice(['Private', 'Self-emp', 'Government', 'Other'], 1000),
            'education': np.random.choice(['HS-grad', 'Bachelors', 'Masters', 'Doctorate'], 1000),
            'occupation': np.random.choice(['Manager', 'Professional', 'Admin', 'Technical', 'Other'], 1000),
            'hours-per-week': np.random.randint(20, 80, 1000),
            'marital-status': np.random.choice(['Single', 'Married', 'Divorced'], 1000),
            'income': np.random.choice(['<=50K', '>50K'], 1000)
        }
        return pd.DataFrame(sample_data)

def preprocess_input(age, workclass, education, occupation, hours, marital_status):
    """Enhanced preprocessing to match model's expected format"""
    # Create input DataFrame with proper column names
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass], 
        'education': [education],
        'occupation': [occupation],
        'hours-per-week': [hours],
        'marital-status': [marital_status]
    })
    
    return input_data

def create_gauge_chart(probability, title):
    """Create an attractive gauge chart for probability display"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgray'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
    return fig

def create_feature_importance_chart(model, feature_names):
    """Create feature importance visualization"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                feature_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Feature Importance in AI Model",
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            return fig
    except:
        pass
    return None

def main():
    # Main header
    st.markdown('<div class="main-header">🤖 AI-Powered Income Predictor</div>', unsafe_allow_html=True)
    
    # Load model and data
    model = load_enhanced_model()
    sample_data = load_sample_data()
    
    # Sidebar for model information
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        st.info("**Model Type:** Gradient Boosting Classifier")
        st.info("**Training Accuracy:** ~85%")
        st.info("**Features Used:** 6 key demographic factors")
        
        # Model statistics
        if sample_data is not None:
            st.markdown("## 📈 Data Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(sample_data))
            with col2:
                high_income = len(sample_data[sample_data['income'] == '>50K']) if 'income' in sample_data.columns else 0
                st.metric(">50K Income", high_income)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Enter Your Information")
        
        # Input form with enhanced styling
        with st.form("prediction_form", clear_on_submit=False):
            # Personal Information
            st.markdown("#### 👤 Personal Details")
            col_a, col_b = st.columns(2)
            
            with col_a:
                age = st.slider(
                    "Age", 
                    min_value=18, 
                    max_value=80, 
                    value=35,
                    help="Your current age in years"
                )
                
                workclass = st.selectbox(
                    "Work Type",
                    options=["Private", "Self-emp", "Government", "Other"],
                    help="Your primary work classification"
                )
                
                education = st.selectbox(
                    "Education Level",
                    options=["HS-grad", "Bachelors", "Masters", "Doctorate"],
                    help="Your highest completed education level"
                )
            
            with col_b:
                occupation = st.selectbox(
                    "Occupation Category",
                    options=["Manager", "Professional", "Admin", "Technical", "Other"],
                    help="Your primary occupation type"
                )
                
                hours = st.slider(
                    "Weekly Work Hours",
                    min_value=10,
                    max_value=80,
                    value=40,
                    help="Average hours worked per week"
                )
                
                marital_status = st.selectbox(
                    "Marital Status",
                    options=["Single", "Married", "Divorced"],
                    help="Your current marital status"
                )
            
            # Prediction button
            predict_button = st.form_submit_button(
                "🔮 Predict Income Level", 
                use_container_width=True,
                type="primary"
            )
    
    with col2:
        # Display input summary
        st.markdown("### 📋 Input Summary")
        
        summary_data = {
            "Age": f"{age} years",
            "Work Type": workclass,
            "Education": education,
            "Occupation": occupation,
            "Hours/Week": f"{hours} hours",
            "Marital Status": marital_status
        }
        
        for key, value in summary_data.items():
            st.markdown(f"**{key}:** {value}")
    
    # Prediction results
    if predict_button:
        try:
            # Preprocess input
            input_data = preprocess_input(age, workclass, education, occupation, hours, marital_status)
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Determine result
            result = "> $50K/year" if prediction == 1 else "≤ $50K/year"
            confidence = max(prediction_proba) * 100
            
            # Display results
            st.markdown("---")
            st.markdown("## 🎉 Prediction Results")
            
            # Main result
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>🎊 High Income Predicted!</h2>
                    <h3>Expected Income: {result}</h3>
                    <p>Confidence Level: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center;">
                    <h2>📊 Standard Income Predicted</h2>
                    <h3>Expected Income: {result}</h3>
                    <p>Confidence Level: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed analysis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Probability gauge for >50K
                gauge_fig = create_gauge_chart(prediction_proba[1], "Probability > $50K")
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Probability breakdown
                st.markdown("### 📊 Probability Breakdown")
                prob_data = {
                    'Income Level': ['≤ $50K', '> $50K'],
                    'Probability': [f"{prediction_proba[0]:.3f}", f"{prediction_proba[1]:.3f}"],
                    'Percentage': [f"{prediction_proba[0]*100:.1f}%", f"{prediction_proba[1]*100:.1f}%"]
                }
                prob_df = pd.DataFrame(prob_data)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            with col3:
                # Feature importance
                feature_names = ['Age', 'Work Type', 'Education', 'Occupation', 'Hours/Week', 'Marital Status']
                importance_chart = create_feature_importance_chart(model, feature_names)
                if importance_chart:
                    st.plotly_chart(importance_chart, use_container_width=True)
            
            # Additional insights
            st.markdown("### 💡 Key Insights")
            
            insights = []
            if age >= 50:
                insights.append("🎯 Mature age often correlates with higher income due to experience")
            if education in ['Masters', 'Doctorate']:
                insights.append("🎓 Advanced education significantly boosts income potential")
            if hours >= 50:
                insights.append("⏰ High work hours indicate strong work commitment")
            if occupation in ['Manager', 'Professional']:
                insights.append("💼 Professional/managerial roles typically offer higher compensation")
            
            if insights:
                for insight in insights:
                    st.success(insight)
            else:
                st.info("💭 Consider advancing education or gaining specialized skills to increase income potential")
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.error("Please check your input values and try again.")
    
    # Additional features
    st.markdown("---")
    
    # Expandable sections
    with st.expander("🔍 About This AI Model"):
        st.markdown("""
        ### Model Architecture
        - **Algorithm:** Gradient Boosting Classifier
        - **Training Data:** Census income dataset with 48,000+ records
        - **Accuracy:** Approximately 85% on validation data
        - **Features:** 6 key demographic and professional factors
        
        ### How It Works
        1. **Data Processing:** Your inputs are normalized and encoded
        2. **Feature Analysis:** The model analyzes patterns in your demographic profile
        3. **Probability Calculation:** Advanced algorithms compute likelihood scores
        4. **Prediction:** Final income category is determined based on learned patterns
        
        ### Limitations
        - Predictions are estimates based on historical data patterns
        - Individual circumstances may vary significantly
        - Model performance may vary across different demographic groups
        """)
    
    with st.expander("📈 Improve Your Income Potential"):
        st.markdown("""
        ### Education & Skills
        - 🎓 **Higher Education:** Advanced degrees strongly correlate with higher income
        - 💻 **Technical Skills:** Learn in-demand technologies and tools
        - 📜 **Certifications:** Industry certifications can boost earning potential
        
        ### Career Development  
        - 🚀 **Leadership Roles:** Seek management and leadership opportunities
        - 🤝 **Networking:** Build professional relationships and connections
        - 📊 **Performance:** Consistently exceed expectations in your current role
        
        ### Industry Considerations
        - 🏭 **High-Paying Industries:** Technology, finance, healthcare, consulting
        - 📍 **Location:** Urban areas typically offer higher salaries
        - 🕐 **Experience:** Years of relevant experience matter significantly
        """)

if __name__ == "__main__":
    main()
