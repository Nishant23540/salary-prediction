import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try importing optional libraries with fallbacks
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    st.warning("⚠️ joblib not found. Using pickle as fallback.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ plotly not found. Using basic charts.")

# If joblib not available, use pickle
if not JOBLIB_AVAILABLE:
    import pickle

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
    
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .standard-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained model with multiple fallback options"""
    model_files = ['best_model.pkl', 'model.pkl', 'best_model.joblib']
    
    for model_file in model_files:
        try:
            if JOBLIB_AVAILABLE:
                model = joblib.load(model_file)
                st.success(f"✅ Model loaded from {model_file}")
                return model
            else:
                # Try with pickle as fallback
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                st.success(f"✅ Model loaded from {model_file} (using pickle)")
                return model
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"⚠️ Could not load {model_file}: {str(e)}")
            continue
    
    # If no model found, create a simple fallback
    st.error("❌ No model file found. Creating demo model...")
    return create_demo_model()

def create_demo_model():
    """Create a simple demo model for demonstration"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Create dummy training data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'workclass': np.random.choice(['Private', 'Self-emp', 'Government', 'Other'], n_samples),
        'education': np.random.choice(['HS-grad', 'Bachelors', 'Masters', 'Doctorate'], n_samples),
        'occupation': np.random.choice(['Manager', 'Professional', 'Admin', 'Technical', 'Other'], n_samples),
        'hours-per-week': np.random.randint(20, 80, n_samples),
        'marital-status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'income': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['workclass', 'education', 'occupation', 'marital-status']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Prepare features
    feature_cols = ['age', 'hours-per-week'] + [col + '_encoded' for col in categorical_cols]
    X = df[feature_cols]
    y = df['income']
    
    # Train simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Store encoders with model
    model.label_encoders = label_encoders
    
    st.info("📝 Using demo model for demonstration purposes")
    return model

@st.cache_data
def load_sample_data():
    """Load sample data with fallback"""
    try:
        df = pd.read_csv('employee.csv')
        return df
    except FileNotFoundError:
        # Create sample data
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

def preprocess_input(age, workclass, education, occupation, hours, marital_status, model):
    """Preprocess input data for prediction"""
    input_data = {
        'age': age,
        'workclass': workclass,
        'education': education,
        'occupation': occupation,
        'hours-per-week': hours,
        'marital-status': marital_status
    }
    
    # If model has encoders (demo model), use them
    if hasattr(model, 'label_encoders'):
        categorical_cols = ['workclass', 'education', 'occupation', 'marital-status']
        processed_data = [age, hours]
        
        for col in categorical_cols:
            if col in model.label_encoders:
                # Handle unseen categories
                try:
                    encoded_val = model.label_encoders[col].transform([input_data[col]])[0]
                except ValueError:
                    # Use first class if category not seen
                    encoded_val = 0
                processed_data.append(encoded_val)
            else:
                processed_data.append(0)
        
        return np.array(processed_data).reshape(1, -1)
    else:
        # For original model, return DataFrame
        return pd.DataFrame([input_data])

def create_simple_gauge(probability, title):
    """Create a simple progress bar if plotly not available"""
    if PLOTLY_AVAILABLE:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 16}},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': 'lightgray'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'lightgreen'}
                ]
            }
        ))
        fig.update_layout(height=250)
        return fig
    else:
        # Fallback to progress bar
        return None

def main():
    # Header
    st.markdown('<div class="main-header">🤖 AI Income Predictor</div>', unsafe_allow_html=True)
    
    # Load model and data
    model = load_model()
    sample_data = load_sample_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        st.info("**Model Type:** Machine Learning Classifier")
        st.info("**Purpose:** Income Level Prediction")
        st.info("**Features:** Demographic & Work Data")
        
        if sample_data is not None:
            st.markdown("## 📈 Dataset Info")
            st.metric("Total Records", len(sample_data))
            if 'income' in sample_data.columns:
                high_income = len(sample_data[sample_data['income'] == '>50K'])
                st.metric("High Income (>50K)", high_income)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Enter Your Information")
        
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                age = st.slider("Age", 18, 80, 35)
                workclass = st.selectbox("Work Type", ["Private", "Self-emp", "Government", "Other"])
                education = st.selectbox("Education", ["HS-grad", "Bachelors", "Masters", "Doctorate"])
            
            with col_b:
                occupation = st.selectbox("Occupation", ["Manager", "Professional", "Admin", "Technical", "Other"])
                hours = st.slider("Hours per Week", 10, 80, 40)
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            
            predict_button = st.form_submit_button("🔮 Predict Income", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Your Profile")
        st.markdown(f"**Age:** {age} years")
        st.markdown(f"**Work:** {workclass}")
        st.markdown(f"**Education:** {education}")
        st.markdown(f"**Occupation:** {occupation}")
        st.markdown(f"**Hours/Week:** {hours}")
        st.markdown(f"**Status:** {marital_status}")
    
    # Prediction
    if predict_button:
        try:
            # Process input
            processed_input = preprocess_input(age, workclass, education, occupation, hours, marital_status, model)
            
            # Make prediction
            prediction = model.predict(processed_input)[0]
            prediction_proba = model.predict_proba(processed_input)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("## 🎉 Prediction Results")
            
            result = "> $50K/year" if prediction == 1 else "≤ $50K/year"
            confidence = max(prediction_proba) * 100
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>🎊 High Income Predicted!</h2>
                    <h3>{result}</h3>
                    <p>Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="standard-result">
                    <h2>📊 Standard Income Predicted</h2>
                    <h3>{result}</h3>
                    <p>Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed results
            col1, col2 = st.columns(2)
            
            with col1:
                if PLOTLY_AVAILABLE:
                    gauge_fig = create_simple_gauge(prediction_proba[1], "Probability > $50K")
                    if gauge_fig:
                        st.plotly_chart(gauge_fig, use_container_width=True)
                else:
                    st.markdown("### 📊 Income Probability")
                    prob_high = prediction_proba[1] * 100
                    st.progress(prediction_proba[1])
                    st.markdown(f"**> $50K:** {prob_high:.1f}%")
                    st.markdown(f"**≤ $50K:** {(100-prob_high):.1f}%")
            
            with col2:
                st.markdown("### 📈 Probability Details")
                prob_df = pd.DataFrame({
                    'Income Level': ['≤ $50K', '> $50K'],
                    'Probability': [f"{prediction_proba[0]:.3f}", f"{prediction_proba[1]:.3f}"],
                    'Percentage': [f"{prediction_proba[0]*100:.1f}%", f"{prediction_proba[1]*100:.1f}%"]
                })
                st.dataframe(prob_df, hide_index=True)
            
            # Insights
            st.markdown("### 💡 Insights")
            if age >= 45:
                st.success("🎯 Experience advantage: Mature age often correlates with higher income")
            if education in ['Masters', 'Doctorate']:
                st.success("🎓 Education boost: Advanced degrees increase income potential")
            if hours >= 45:
                st.success("⏰ Work commitment: High hours show dedication")
            if occupation in ['Manager', 'Professional']:
                st.success("💼 Career advantage: Professional roles offer better compensation")
            
            if prediction == 0:
                st.info("💭 Consider: Advanced education, skill development, or career growth opportunities")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please try again or check your inputs")
    
    # Additional info
    st.markdown("---")
    
    with st.expander("🔍 About This Model"):
        st.markdown("""
        ### How It Works
        - **Input Processing:** Your demographic data is analyzed
        - **Pattern Recognition:** Model finds income patterns in historical data  
        - **Probability Calculation:** Computes likelihood of income levels
        - **Prediction:** Provides income category with confidence score
        
        ### Limitations
        - Based on historical census data patterns
        - Individual results may vary
        - For educational/demonstration purposes
        """)
    
    with st.expander("📈 Tips to Increase Income"):
        st.markdown("""
        ### Education & Skills
        - 🎓 **Higher Education:** Advanced degrees correlate with higher income
        - 💻 **Technical Skills:** Learn in-demand technologies
        - 📜 **Certifications:** Industry certifications boost earning potential
        
        ### Career Growth
        - 🚀 **Leadership:** Seek management opportunities
        - 🤝 **Networking:** Build professional connections
        - 📊 **Performance:** Excel in your current role
        
        ### Strategic Moves
        - 🏭 **Industry Choice:** Tech, finance, healthcare pay well
        - 📍 **Location:** Urban areas offer higher salaries
        - 🕐 **Experience:** Gain relevant work experience
        """)

if __name__ == "__main__":
    main()
