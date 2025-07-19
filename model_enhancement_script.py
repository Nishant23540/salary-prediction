"""
Enhanced Model Training Script
This script improves upon your existing model with better preprocessing,
feature engineering, and hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedIncomePredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_and_preprocess_data(self, csv_path='employee.csv'):
        """Enhanced data loading and preprocessing"""
        try:
            # Load data
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(df)} records from {csv_path}")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Handle missing values marked as '?'
            df = df.replace('?', np.nan)
            
            # Select key features for prediction
            feature_columns = ['age', 'workclass', 'education', 'occupation', 
                             'hours-per-week', 'marital-status']
            target_column = 'income'
            
            # Check if all required columns exist
            missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return None, None
                
            # Filter data
            df_clean = df[feature_columns + [target_column]].copy()
            
            # Remove rows with missing values
            df_clean = df_clean.dropna()
            
            print(f"✅ Clean dataset: {len(df_clean)} records")
            
            X = df_clean[feature_columns]
            y = df_clean[target_column]
            
            return X, y
            
        except FileNotFoundError:
            print(f"❌ File {csv_path} not found")
            return None, None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None
    
    def engineer_features(self, X):
        """Advanced feature engineering"""
        X_enhanced = X.copy()
        
        # Create age groups
        X_enhanced['age_group'] = pd.cut(X_enhanced['age'], 
                                       bins=[0, 25, 35, 45, 55, 100], 
                                       labels=['Young', 'Early_Career', 'Mid_Career', 'Senior', 'Elder'])
        
        # Create hours categories
        X_enhanced['hours_category'] = pd.cut(X_enhanced['hours-per-week'],
                                            bins=[0, 20, 40, 50, 100],
                                            labels=['Part_Time', 'Full_Time', 'Overtime', 'Heavy_Overtime'])
        
        # Education ranking
        education_rank = {
            'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5,
            '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10,
            'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14,
            'Prof-school': 15, 'Doctorate': 16
        }
        X_enhanced['education_rank'] = X_enhanced['education'].map(education_rank).fillna(9)
        
        return X_enhanced
    
    def encode_features(self, X, is_training=True):
        """Enhanced feature encoding"""
        X_encoded = X.copy()
        
        # Categorical columns to encode
        categorical_cols = ['workclass', 'education', 'occupation', 'marital-status', 
                          'age_group', 'hours_category']
        
        for col in categorical_cols:
            if col in X_encoded.columns:
                if is_training:
                    # Create and fit encoder during training
                    self.encoders[col] = LabelEncoder()
                    X_encoded[col] = self.encoders[col].fit_transform(X_encoded[col].astype(str))
                else:
                    # Use existing encoder during prediction
                    if col in self.encoders:
                        # Handle unseen categories
                        unique_values = set(self.encoders[col].classes_)
                        X_encoded[col] = X_encoded[col].astype(str)
                        X_encoded[col] = X_encoded[col].apply(
                            lambda x: x if x in unique_values else self.encoders[col].classes_[0]
                        )
                        X_encoded[col] = self.encoders[col].transform(X_encoded[col])
                    else:
                        # Default encoding if encoder not found
                        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
        
        return X_encoded
    
    def train_enhanced_model(self, X, y):
        """Train model with hyperparameter optimization"""
        print("🚀 Starting enhanced model training...")
        
        # Feature engineering
        X_enhanced = self.engineer_features(X)
        
        # Encode features
        X_encoded = self.encode_features(X_enhanced, is_training=True)
        
        # Encode target
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        self.target_encoder = label_encoder
        
        # Store feature names
        self.feature_names = X_encoded.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define parameter grid for optimization
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [6, 8, 10],
            'min_samples_split': [100, 200],
            'subsample': [0.8, 0.9]
        }
        
        # Initialize model
        base_model = GradientBoostingClassifier(random_state=42)
        
        print("🔍 Performing hyperparameter optimization...")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV score: {grid_search.best_score_:.3f}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        test_accuracy = self.model.score(X_test_scaled, y_test)
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"✅ Test Accuracy: {test_accuracy:.3f}")
        print(f"✅ Test AUC: {test_auc:.3f}")
        
        # Print classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.target_encoder.classes_))
        
        return self.model
    
    def predict(self, X):
        """Enhanced prediction method"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Feature engineering
        X_enhanced = self.engineer_features(X)
        
        # Encode features
        X_encoded = self.encode_features(X_enhanced, is_training=False)
        
        # Ensure same feature order
        X_encoded = X_encoded.reindex(columns=self.feature_names, fill_value=0)
        
        # Scale features
        X_scaled = self.scaler.transform(X_encoded)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Decode predictions
        predictions_decoded = self.target_encoder.inverse_transform(predictions)
        
        return predictions_decoded, probabilities
    
    def save_model(self, filepath='enhanced_income_model.pkl'):
        """Save the complete model pipeline"""
        model_data = {
            'model': self.model,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'target_encoder': self.target_encoder,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='enhanced_income_model.pkl'):
        """Load the complete model pipeline"""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.encoders = model_data['encoders']
            self.scaler = model_data['scaler']
            self.target_encoder = model_data['target_encoder']
            self.feature_names = model_data['feature_names']
            
            print(f"✅ Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """Main training function"""
    print("🤖 Enhanced Income Prediction Model Training")
    print("=" * 50)
    
    # Initialize predictor
    predictor = EnhancedIncomePredictor()
    
    # Load and preprocess data
    X, y = predictor.load_and_preprocess_data('employee.csv')
    
    if X is not None and y is not None:
        # Train the enhanced model
        model = predictor.train_enhanced_model(X, y)
        
        # Save the model
        predictor.save_model('best_model.pkl')
        
        print("\n🎉 Model training completed successfully!")
        print("📁 Model saved as 'best_model.pkl'")
        print("🚀 You can now use the enhanced Streamlit app!")
        
        # Test prediction
        print("\n🧪 Testing prediction with sample data...")
        sample_data = pd.DataFrame({
            'age': [39],
            'workclass': ['Private'],
            'education': ['Bachelors'],
            'occupation': ['Professional'],
            'hours-per-week': [40],
            'marital-status': ['Married']
        })
        
        pred, prob = predictor.predict(sample_data)
        print(f"Sample prediction: {pred[0]} (confidence: {max(prob[0]):.3f})")
        
    else:
        print("❌ Failed to load data. Please check your CSV file.")

if __name__ == "__main__":
    main()
