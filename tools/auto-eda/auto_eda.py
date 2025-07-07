"""
🔍 AutoEDA - Automated Exploratory Data Analysis Tool
===================================================

This tool automatically generates comprehensive exploratory data analysis reports
for any dataset. Perfect for quickly understanding your data before modeling.

Author: ML Mastery Hub Team
Features:
- Automatic data type detection
- Statistical summaries
- Missing value analysis
- Correlation analysis
- Distribution plots
- Outlier detection
- Interactive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import os

warnings.filterwarnings('ignore')

class AutoEDA:
    """
    Automated Exploratory Data Analysis tool that generates comprehensive
    reports for any dataset with minimal code.
    """
    
    def __init__(self, data: pd.DataFrame, target_column: Optional[str] = None):
        """
        Initialize AutoEDA with a dataset.
        
        Args:
            data: pandas DataFrame to analyze
            target_column: Optional target column for supervised learning tasks
        """
        self.data = data.copy()
        self.target_column = target_column
        self.numeric_columns = self._get_numeric_columns()
        self.categorical_columns = self._get_categorical_columns()
        self.datetime_columns = self._get_datetime_columns()
        self.report = {}
        
        print(f"🔍 AutoEDA initialized successfully!")
        print(f"📊 Dataset shape: {self.data.shape}")
        print(f"🔢 Numeric columns: {len(self.numeric_columns)}")
        print(f"🏷️  Categorical columns: {len(self.categorical_columns)}")
        print(f"📅 Datetime columns: {len(self.datetime_columns)}")
        
    def _get_numeric_columns(self) -> List[str]:
        """Get numeric columns from the dataset."""
        return self.data.select_dtypes(include=[np.number]).columns.tolist()
    
    def _get_categorical_columns(self) -> List[str]:
        """Get categorical columns from the dataset."""
        return self.data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def _get_datetime_columns(self) -> List[str]:
        """Get datetime columns from the dataset."""
        return self.data.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    
    def generate_basic_info(self) -> Dict:
        """Generate basic information about the dataset."""
        print("\n📋 Generating Basic Dataset Information...")
        
        info = {
            'shape': self.data.shape,
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'dtypes': self.data.dtypes.to_dict(),
            'null_counts': self.data.isnull().sum().to_dict(),
            'null_percentages': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'unique_counts': self.data.nunique().to_dict(),
            'duplicate_rows': self.data.duplicated().sum()
        }
        
        self.report['basic_info'] = info
        
        # Display summary
        print(f"✅ Dataset contains {info['shape'][0]:,} rows and {info['shape'][1]} columns")
        print(f"💾 Memory usage: {info['memory_usage'] / 1024**2:.2f} MB")
        print(f"🔄 Duplicate rows: {info['duplicate_rows']:,}")
        
        # Missing values summary
        missing_cols = [col for col, pct in info['null_percentages'].items() if pct > 0]
        if missing_cols:
            print(f"❌ Columns with missing values: {len(missing_cols)}")
        else:
            print("✅ No missing values found!")
            
        return info
    
    def analyze_numeric_columns(self) -> Dict:
        """Comprehensive analysis of numeric columns."""
        if not self.numeric_columns:
            print("⚠️  No numeric columns found!")
            return {}
            
        print(f"\n🔢 Analyzing {len(self.numeric_columns)} Numeric Columns...")
        
        numeric_analysis = {}
        
        for col in self.numeric_columns:
            series = self.data[col].dropna()
            
            if len(series) == 0:
                continue
                
            analysis = {
                'count': len(series),
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'q1': series.quantile(0.25),
                'q3': series.quantile(0.75),
                'iqr': series.quantile(0.75) - series.quantile(0.25),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'outliers_iqr': self._detect_outliers_iqr(series),
                'outliers_zscore': self._detect_outliers_zscore(series)
            }
            
            numeric_analysis[col] = analysis
        
        self.report['numeric_analysis'] = numeric_analysis
        
        # Create visualizations
        self._plot_numeric_distributions()
        self._plot_correlation_matrix()
        
        return numeric_analysis
    
    def analyze_categorical_columns(self) -> Dict:
        """Comprehensive analysis of categorical columns."""
        if not self.categorical_columns:
            print("⚠️  No categorical columns found!")
            return {}
            
        print(f"\n🏷️  Analyzing {len(self.categorical_columns)} Categorical Columns...")
        
        categorical_analysis = {}
        
        for col in self.categorical_columns:
            series = self.data[col].dropna()
            
            if len(series) == 0:
                continue
                
            value_counts = series.value_counts()
            
            analysis = {
                'unique_count': series.nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                'cardinality_ratio': series.nunique() / len(series),
                'top_10_values': value_counts.head(10).to_dict()
            }
            
            categorical_analysis[col] = analysis
        
        self.report['categorical_analysis'] = categorical_analysis
        
        # Create visualizations
        self._plot_categorical_distributions()
        
        return categorical_analysis
    
    def analyze_target_variable(self) -> Dict:
        """Analyze the target variable if specified."""
        if not self.target_column or self.target_column not in self.data.columns:
            print("⚠️  No valid target column specified!")
            return {}
            
        print(f"\n🎯 Analyzing Target Variable: {self.target_column}")
        
        target_series = self.data[self.target_column].dropna()
        
        if self.target_column in self.numeric_columns:
            # Regression target
            analysis = {
                'type': 'numeric',
                'count': len(target_series),
                'mean': target_series.mean(),
                'median': target_series.median(),
                'std': target_series.std(),
                'min': target_series.min(),
                'max': target_series.max(),
                'skewness': target_series.skew(),
                'kurtosis': target_series.kurtosis()
            }
        else:
            # Classification target
            value_counts = target_series.value_counts()
            analysis = {
                'type': 'categorical',
                'unique_count': target_series.nunique(),
                'class_distribution': value_counts.to_dict(),
                'class_percentages': (value_counts / len(target_series) * 100).to_dict(),
                'is_balanced': (value_counts.max() / value_counts.min()) < 2 if len(value_counts) > 1 else True
            }
        
        self.report['target_analysis'] = analysis
        
        # Create target-specific visualizations
        self._plot_target_analysis()
        
        return analysis
    
    def detect_outliers(self) -> Dict:
        """Detect outliers in numeric columns using multiple methods."""
        if not self.numeric_columns:
            return {}
            
        print("\n🚨 Detecting Outliers...")
        
        outlier_summary = {}
        
        for col in self.numeric_columns:
            series = self.data[col].dropna()
            
            if len(series) == 0:
                continue
            
            # IQR method
            iqr_outliers = self._detect_outliers_iqr(series)
            
            # Z-score method
            zscore_outliers = self._detect_outliers_zscore(series)
            
            # Modified Z-score method
            modified_zscore_outliers = self._detect_outliers_modified_zscore(series)
            
            outlier_summary[col] = {
                'iqr_outliers': len(iqr_outliers),
                'zscore_outliers': len(zscore_outliers),
                'modified_zscore_outliers': len(modified_zscore_outliers),
                'iqr_percentage': len(iqr_outliers) / len(series) * 100,
                'zscore_percentage': len(zscore_outliers) / len(series) * 100,
                'modified_zscore_percentage': len(modified_zscore_outliers) / len(series) * 100
            }
        
        self.report['outlier_analysis'] = outlier_summary
        
        # Create outlier visualizations
        self._plot_outlier_analysis()
        
        return outlier_summary
    
    def _detect_outliers_iqr(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using the IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series < lower_bound) | (series > upper_bound)].index.values
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3) -> np.ndarray:
        """Detect outliers using the Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return series[z_scores > threshold].index.values
    
    def _detect_outliers_modified_zscore(self, series: pd.Series, threshold: float = 3.5) -> np.ndarray:
        """Detect outliers using the Modified Z-score method."""
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        return series[np.abs(modified_z_scores) > threshold].index.values
    
    def _plot_numeric_distributions(self):
        """Create distribution plots for numeric columns."""
        if not self.numeric_columns:
            return
            
        n_cols = min(3, len(self.numeric_columns))
        n_rows = (len(self.numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(self.numeric_columns):
            if i < len(axes):
                self.data[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(self.numeric_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('📊 Numeric Column Distributions', y=1.02, fontsize=16)
        plt.show()
    
    def _plot_correlation_matrix(self):
        """Create correlation heatmap for numeric columns."""
        if len(self.numeric_columns) < 2:
            return
            
        correlation_matrix = self.data[self.numeric_columns].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('🔥 Correlation Matrix Heatmap')
        plt.tight_layout()
        plt.show()
    
    def _plot_categorical_distributions(self):
        """Create distribution plots for categorical columns."""
        if not self.categorical_columns:
            return
            
        n_cols = min(2, len(self.categorical_columns))
        n_rows = (len(self.categorical_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(self.categorical_columns):
            if i < len(axes) and self.data[col].nunique() <= 20:  # Only plot if not too many categories
                value_counts = self.data[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Top Values in {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(self.categorical_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('🏷️  Categorical Column Distributions', y=1.02, fontsize=16)
        plt.show()
    
    def _plot_target_analysis(self):
        """Create target variable analysis plots."""
        if not self.target_column:
            return
            
        if self.target_column in self.numeric_columns:
            # Numeric target
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Distribution
            self.data[self.target_column].hist(bins=30, ax=axes[0], alpha=0.7)
            axes[0].set_title(f'Distribution of {self.target_column}')
            axes[0].set_xlabel(self.target_column)
            axes[0].set_ylabel('Frequency')
            
            # Box plot
            self.data[self.target_column].plot(kind='box', ax=axes[1])
            axes[1].set_title(f'Box Plot of {self.target_column}')
            
        else:
            # Categorical target
            value_counts = self.data[self.target_column].value_counts()
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Bar plot
            value_counts.plot(kind='bar', ax=axes[0])
            axes[0].set_title(f'Class Distribution - {self.target_column}')
            axes[0].set_xlabel(self.target_column)
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Pie chart
            value_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
            axes[1].set_title(f'Class Proportions - {self.target_column}')
            axes[1].set_ylabel('')
        
        plt.tight_layout()
        plt.suptitle(f'🎯 Target Variable Analysis: {self.target_column}', y=1.02, fontsize=16)
        plt.show()
    
    def _plot_outlier_analysis(self):
        """Create outlier analysis plots."""
        if not self.numeric_columns:
            return
            
        n_cols = min(3, len(self.numeric_columns))
        n_rows = (len(self.numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(self.numeric_columns):
            if i < len(axes):
                self.data[col].plot(kind='box', ax=axes[i])
                axes[i].set_title(f'Outliers in {col}')
                axes[i].set_ylabel(col)
        
        # Hide unused subplots
        for i in range(len(self.numeric_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('🚨 Outlier Analysis', y=1.02, fontsize=16)
        plt.show()
    
    def generate_full_report(self, save_report: bool = False, output_dir: str = "eda_reports") -> Dict:
        """Generate a comprehensive EDA report."""
        print("🚀 Generating Comprehensive EDA Report...")
        print("=" * 60)
        
        # Run all analyses
        self.generate_basic_info()
        self.analyze_numeric_columns()
        self.analyze_categorical_columns()
        self.analyze_target_variable()
        self.detect_outliers()
        
        # Generate summary
        summary = self._generate_summary()
        self.report['summary'] = summary
        
        if save_report:
            self._save_report(output_dir)
        
        print("\n✅ EDA Report Generated Successfully!")
        return self.report
    
    def _generate_summary(self) -> Dict:
        """Generate a summary of key findings."""
        summary = {
            'dataset_quality': 'Good',
            'recommendations': [],
            'key_insights': []
        }
        
        # Check data quality
        basic_info = self.report.get('basic_info', {})
        missing_pct = max(basic_info.get('null_percentages', {}).values()) if basic_info.get('null_percentages') else 0
        
        if missing_pct > 50:
            summary['dataset_quality'] = 'Poor'
            summary['recommendations'].append("High missing values detected. Consider data imputation or collection.")
        elif missing_pct > 20:
            summary['dataset_quality'] = 'Fair'
            summary['recommendations'].append("Moderate missing values. Consider imputation strategies.")
        
        # Check for outliers
        outlier_analysis = self.report.get('outlier_analysis', {})
        if outlier_analysis:
            high_outlier_cols = [col for col, analysis in outlier_analysis.items() 
                               if analysis.get('iqr_percentage', 0) > 10]
            if high_outlier_cols:
                summary['recommendations'].append(f"High outlier percentage in: {', '.join(high_outlier_cols)}")
        
        # Target variable insights
        target_analysis = self.report.get('target_analysis', {})
        if target_analysis:
            if target_analysis.get('type') == 'categorical' and not target_analysis.get('is_balanced', True):
                summary['recommendations'].append("Imbalanced target classes detected. Consider resampling techniques.")
        
        return summary
    
    def _save_report(self, output_dir: str):
        """Save the report to files."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save report as JSON
        import json
        report_path = Path(output_dir) / "eda_report.json"
        
        # Convert numpy types to native Python types for JSON serialization
        json_report = self._convert_numpy_types(self.report)
        
        with open(report_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        print(f"📁 Report saved to: {report_path}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def quick_eda(data: pd.DataFrame, target_column: Optional[str] = None, 
              save_report: bool = False) -> Dict:
    """
    Quick EDA function for immediate analysis.
    
    Args:
        data: pandas DataFrame to analyze
        target_column: Optional target column name
        save_report: Whether to save the report to files
    
    Returns:
        Dictionary containing the complete EDA report
    """
    eda = AutoEDA(data, target_column)
    return eda.generate_full_report(save_report=save_report)

# Example usage
if __name__ == "__main__":
    # Demo with built-in dataset
    from sklearn.datasets import load_boston, load_iris
    import pandas as pd
    
    print("🌸 Demo: Iris Dataset Analysis")
    print("=" * 40)
    
    # Load iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    
    # Run AutoEDA
    report = quick_eda(iris_df, target_column='species', save_report=True)
    
    print("\n🎉 AutoEDA Demo Completed!")
    print("📁 Check the 'eda_reports' folder for saved reports.")
