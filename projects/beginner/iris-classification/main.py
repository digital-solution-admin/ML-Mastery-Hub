"""
🌸 Iris Classification - Your First ML Project
==============================================

This project demonstrates basic machine learning concepts using the famous Iris dataset.
Perfect for beginners to understand the ML workflow.

Author: ML Mastery Hub Team
Difficulty: ⭐ Beginner
Time: ~30 minutes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class IrisClassifier:
    """
    A comprehensive Iris classification pipeline that demonstrates
    the complete ML workflow from data loading to model evaluation.
    """
    
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and explore the Iris dataset"""
        print("🌸 Loading Iris Dataset...")
        
        # Load the dataset
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        
        # Create a DataFrame for easier manipulation
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        self.data = pd.DataFrame(self.X, columns=feature_names)
        self.data['species'] = [target_names[i] for i in self.y]
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Shape: {self.data.shape}")
        print(f"🏷️  Classes: {list(target_names)}")
        print(f"📝 Features: {list(feature_names)}")
        
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n🔍 Exploratory Data Analysis")
        print("=" * 40)
        
        # Basic statistics
        print("\n📈 Dataset Overview:")
        print(self.data.head())
        
        print("\n📊 Statistical Summary:")
        print(self.data.describe())
        
        print("\n🏷️  Class Distribution:")
        print(self.data['species'].value_counts())
        
        # Visualizations
        self._create_visualizations()
        
    def _create_visualizations(self):
        """Create informative visualizations"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🌸 Iris Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Pairplot
        sns.pairplot(self.data, hue='species', diag_kind='hist')
        plt.suptitle('Feature Relationships by Species', y=1.02)
        plt.tight_layout()
        plt.show()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('🔥 Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
        # 3. Box plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        features = self.data.columns[:-1]
        
        for i, feature in enumerate(features):
            row, col = i // 2, i % 2
            sns.boxplot(data=self.data, x='species', y=feature, ax=axes[row, col])
            axes[row, col].set_title(f'📦 {feature} by Species')
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for training"""
        print("\n🔧 Preparing Data for Training...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data split completed!")
        print(f"📊 Training set: {self.X_train.shape[0]} samples")
        print(f"📊 Test set: {self.X_test.shape[0]} samples")
        
    def train_models(self):
        """Train multiple models and compare performance"""
        print("\n🤖 Training Multiple Models...")
        print("=" * 40)
        
        # Define models
        self.models = {
            '🌳 Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            '🎯 SVM': SVC(kernel='rbf', random_state=42),
            '👥 K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
            '📈 Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            # Use scaled data for SVM and Logistic Regression
            if 'SVM' in name or 'Logistic' in name:
                X_train, X_test = self.X_train_scaled, self.X_test_scaled
            else:
                X_train, X_test = self.X_train, self.X_test
            
            # Train the model
            model.fit(X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, self.y_train, cv=5)
            
            self.results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"✅ {name} - Accuracy: {accuracy:.3f}")
            print(f"   📊 Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    def evaluate_models(self):
        """Detailed evaluation of all models"""
        print("\n📊 Model Evaluation Results")
        print("=" * 50)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test Accuracy': [self.results[model]['accuracy'] for model in self.results.keys()],
            'CV Mean': [self.results[model]['cv_mean'] for model in self.results.keys()],
            'CV Std': [self.results[model]['cv_std'] for model in self.results.keys()]
        })
        
        results_df = results_df.sort_values('Test Accuracy', ascending=False)
        print(results_df.to_string(index=False))
        
        # Find best model
        best_model_name = results_df.iloc[0]['Model']
        best_accuracy = results_df.iloc[0]['Test Accuracy']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"🎯 Best Accuracy: {best_accuracy:.3f}")
        
        # Detailed evaluation of best model
        self._detailed_evaluation(best_model_name)
        
        # Visualize results
        self._visualize_results(results_df)
    
    def _detailed_evaluation(self, model_name):
        """Detailed evaluation of the best model"""
        print(f"\n🔍 Detailed Analysis of {model_name}")
        print("=" * 50)
        
        y_pred = self.results[model_name]['predictions']
        
        # Classification report
        print("📋 Classification Report:")
        target_names = ['Setosa', 'Versicolor', 'Virginica']
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'🎯 Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def _visualize_results(self, results_df):
        """Visualize model comparison"""
        plt.figure(figsize=(12, 6))
        
        # Accuracy comparison
        plt.subplot(1, 2, 1)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = plt.bar(results_df['Model'], results_df['Test Accuracy'], color=colors)
        plt.title('🏆 Model Accuracy Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0.8, 1.0)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, results_df['Test Accuracy']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{accuracy:.3f}', ha='center', va='bottom')
        
        # Cross-validation scores
        plt.subplot(1, 2, 2)
        plt.errorbar(results_df['Model'], results_df['CV Mean'], 
                    yerr=results_df['CV Std'], fmt='o-', capsize=5, capthick=2)
        plt.title('📊 Cross-Validation Scores')
        plt.xlabel('Models')
        plt.ylabel('CV Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0.8, 1.0)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("🚀 Starting Complete Iris Classification Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Prepare data
        self.prepare_data()
        
        # Step 4: Train models
        self.train_models()
        
        # Step 5: Evaluate models
        self.evaluate_models()
        
        print("\n🎉 Pipeline completed successfully!")
        print("🌟 You've just completed your first ML project!")
        
        return self.results

def main():
    """Main function to run the Iris classification project"""
    print("🌸 Welcome to your first Machine Learning project!")
    print("🎯 Today we'll classify Iris flowers using various ML algorithms")
    print("📚 This is a perfect starting point for your ML journey!\n")
    
    # Create and run the classifier
    classifier = IrisClassifier()
    results = classifier.run_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("🎓 Congratulations! You've successfully:")
    print("✅ Loaded and explored a real dataset")
    print("✅ Visualized data patterns")
    print("✅ Trained multiple ML models")
    print("✅ Evaluated model performance")
    print("✅ Identified the best performing model")
    print("\n🚀 Ready for your next ML adventure?")
    print("👉 Check out intermediate projects in the projects/intermediate/ folder!")

if __name__ == "__main__":
    main()
