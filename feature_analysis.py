#author: Lukshan Sharvaswaran
#date: 30/04/2025
#purpose: to analyse the features of the dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import os
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.multicomp import MultiComparison
import logging

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_analysis.log'),
        logging.StreamHandler()
    ]
)

# Create output directory
OUTPUT_DIR = 'feature_analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the dataset for analysis."""
    logging.info("Loading dataset...")
    df = pd.read_csv('dft-road-casualty-statistics-collision-2023.csv')
    return df

def basic_data_exploration(df):
    """Perform comprehensive basic data exploration."""
    logging.info("Performing basic data exploration...")
    
    # Basic information
    info = {
        "Dataset Shape": df.shape,
        "Columns": df.columns.tolist(),
        "Data Types": df.dtypes,
        "Summary Statistics": df.describe(),
        "Missing Values": df.isnull().sum(),
        "Missing Values Percentage": (df.isnull().sum() / len(df)) * 100
    }
    
    # Save to text file
    with open(os.path.join(OUTPUT_DIR, 'data_exploration_summary.txt'), 'w') as f:
        for key, value in info.items():
            f.write(f"\n=== {key} ===\n")
            f.write(str(value))
            f.write("\n")
    
    return info

def analyze_distributions(df):
    """Analyze distributions of numerical features with statistical tests."""
    logging.info("Analyzing feature distributions...")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    distribution_stats = {}
    
    for col in numerical_cols:
        # Create distribution plot
        plt.figure(figsize=(12, 6))
        
        # Histogram with KDE
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'distribution_{col}.png'))
        plt.close()
        
        # Calculate distribution statistics
        stats_dict = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis(),
            'shapiro_test': stats.shapiro(df[col].dropna())
        }
        
        distribution_stats[col] = stats_dict
    
    # Save distribution statistics
    pd.DataFrame(distribution_stats).to_csv(
        os.path.join(OUTPUT_DIR, 'distribution_statistics.csv')
    )
    
    return distribution_stats

def analyze_categorical_features(df):
    """Analyze categorical features with statistical tests."""
    logging.info("Analyzing categorical features...")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_stats = {}
    
    for col in categorical_cols:
        # Create value counts plot
        plt.figure(figsize=(12, 6))
        value_counts = df[col].value_counts()
        
        # Plot top 10 categories if there are many unique values
        if len(value_counts) > 10:
            value_counts = value_counts.head(10)
        
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'categorical_{col}.png'))
        plt.close()
        
        # Calculate categorical statistics
        stats_dict = {
            'unique_values': df[col].nunique(),
            'most_common': df[col].mode()[0],
            'most_common_count': df[col].value_counts().iloc[0],
            'entropy': stats.entropy(df[col].value_counts(normalize=True))
        }
        
        categorical_stats[col] = stats_dict
    
    # Save categorical statistics
    pd.DataFrame(categorical_stats).to_csv(
        os.path.join(OUTPUT_DIR, 'categorical_statistics.csv')
    )
    
    return categorical_stats

def analyze_correlations(df):
    """Perform comprehensive correlation analysis."""
    logging.info("Analyzing correlations...")
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate different correlation metrics
    correlation_matrices = {
        'pearson': df[numerical_cols].corr(method='pearson'),
        'spearman': df[numerical_cols].corr(method='spearman'),
        'kendall': df[numerical_cols].corr(method='kendall')
    }
    
    # Plot correlation matrices
    for method, corr_matrix in correlation_matrices.items():
        plt.figure(figsize=(15, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'correlation_matrix_{method}.png'))
        plt.close()
        
        # Save correlation matrix
        corr_matrix.to_csv(
            os.path.join(OUTPUT_DIR, f'correlation_matrix_{method}.csv')
        )
    
    return correlation_matrices

def analyze_severity_relationships(df):
    """Analyze relationships between features and accident severity."""
    logging.info("Analyzing severity relationships...")
    
    severity_analysis = {}
    
    # Analyze numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != 'accident_severity':
            # ANOVA test
            groups = [group for _, group in df.groupby('accident_severity')[col]]
            f_stat, p_value = f_oneway(*groups)
            
            severity_analysis[col] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'correlation_with_severity': df[col].corr(df['accident_severity'])
            }
    
    # Analyze categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Chi-square test
        contingency = pd.crosstab(df[col], df['accident_severity'])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        severity_analysis[col] = {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof
        }
        
        # Create severity distribution plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='accident_severity', y=col, data=df)
        plt.title(f'Severity Distribution by {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'severity_by_{col}.png'))
        plt.close()
    
    # Save severity analysis results
    pd.DataFrame(severity_analysis).to_csv(
        os.path.join(OUTPUT_DIR, 'severity_analysis.csv')
    )
    
    return severity_analysis

def perform_pca_analysis(df):
    """Perform Principal Component Analysis."""
    logging.info("Performing PCA analysis...")
    
    # Select numerical columns and handle missing values
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df_numerical = df[numerical_cols].fillna(df[numerical_cols].mean())
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numerical)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_explained_variance.png'))
    plt.close()
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=numerical_cols
    )
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(OUTPUT_DIR, 'pca_feature_importance.csv'))
    
    return feature_importance

def main():
    """Main function to run all analyses."""
    logging.info("Starting feature analysis...")
    
    # Load data
    df = load_and_prepare_data()
    
    # Perform analyses
    basic_info = basic_data_exploration(df)
    distribution_stats = analyze_distributions(df)
    categorical_stats = analyze_categorical_features(df)
    correlation_matrices = analyze_correlations(df)
    severity_analysis = analyze_severity_relationships(df)
    pca_results = perform_pca_analysis(df)
    
    logging.info("Feature analysis completed. Results saved in feature_analysis_results directory.")

if __name__ == "__main__":
    main()

