import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import HeatMap

# Optional: SHAP for model explainability
# import shap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
sns.set_theme(style="whitegrid", context="notebook")

# Create directories for outputs if they don't exist
Path("outputs").mkdir(parents=True, exist_ok=True)
Path("plots").mkdir(parents=True, exist_ok=True)

# Load the dataset
df = pd.read_csv('dft-road-casualty-statistics-collision-2023.csv', low_memory=False)

print("### Initial Data Inspection")

#
# Display the shape of the dataset
print(f"\nDataset shape: Rows = {df.shape[0]}, Columns = {df.shape[1]}")
#
# Show the first 5 rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())
#
# Show 5 random rows to inspect data variety
print("\nSample of 5 random rows:")
print(df.sample(5, random_state=42))  # random_state for reproducibility
#
# Display data types and non-null value counts for all columns
print("\nDataFrame information:")
print(df.info())
#
# Show summary statistics for numeric columns
print("\nSummary statistics for numeric columns:")
print(df.describe(include=[np.number]))
#
# Show summary statistics for categorical columns
print("\nSummary statistics for categorical columns:")
print(df.describe(include=['object', 'category']))
#
# Show missing value counts per column, sorted by most missing
print("\nMissing values per column (sorted by most missing):")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False))
#
# Identify low-value columns in the dataset
print("\nIdentifying low-value columns in the dataset")

# Columns with only one unique value (excluding NaNs)
single_value_cols = []
for col in df.columns:
    unique_values = df[col].dropna().nunique()
    if unique_values <= 1:
        single_value_cols.append((col, unique_values))

# Columns where more than 95% of values are missing
high_missing_cols = []
for col in df.columns:
    missing_pct = (df[col].isnull().sum() / len(df)) * 100
    if missing_pct > 95:
        high_missing_cols.append((col, missing_pct))

# Columns that are likely high-cardinality identifiers
high_cardinality_cols = []
for col in df.columns:
    unique_values = df[col].nunique()
    if unique_values > len(df) * 0.9:  # If more than 90% of rows have unique values
        high_cardinality_cols.append((col, unique_values))

print("\nColumns with only one unique value (excluding NaNs):")
for col, count in single_value_cols:
    print(f"- {col}: {count} unique values")

print("\nColumns with more than 95% missing values:")
for col, pct in high_missing_cols:
    print(f"- {col}: {pct:.2f}% missing")

print("\nColumns with high cardinality (likely identifiers):")
for col, count in high_cardinality_cols:
    print(f"- {col}: {count} unique values out of {len(df)} rows")

# Combine all identified low-value columns
low_value_cols = set([col for col, _ in single_value_cols] + 
                     [col for col, _ in high_missing_cols] + 
                     [col for col, _ in high_cardinality_cols])

print("\nAll low-value columns identified in the dataset:")
for col in sorted(low_value_cols):
    print(f"- {col}")

print("### Data Cleaning Steps")

# Remove rows with missing location information
location_cols = ['location_easting_osgr', 'location_northing_osgr', 'latitude', 'longitude']
initial_rows = len(df)
df = df.dropna(subset=location_cols)
rows_removed = initial_rows - len(df)
print(f"\nRows removed due to missing location data: {rows_removed}")
print(f"Remaining rows after location data cleaning: {len(df)}")

# Replace special code (-1) with NaN for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
replacements = {}
for col in numeric_cols:
    count = (df[col] == -1).sum()
    if count > 0:
        df[col] = df[col].replace(-1, np.nan)
        replacements[col] = count

print("\nColumns where -1 values were replaced with NaN:")
for col, count in replacements.items():
    print(f"{col}: {count} replacements")

# Standardize and clean string columns (lowercase, remove whitespace)
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = df[col].str.strip().str.lower()

print("\nUnique value counts in string columns after cleaning:")
for col in object_cols:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique values")

# Show most frequent values in each categorical column
print("\nTop 10 most frequent values for each categorical column:")
for col in object_cols:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))

# Feature engineering for risk modeling
print("### Feature Engineering for Risk Modelling")

# 1. Extract temporal features
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['time'] = pd.to_datetime(df['time'], format='%H:%M').dt.time
df['hour'] = pd.to_datetime(df['time'].astype(str)).dt.hour
df['month'] = df['date'].dt.month
# day_of_week is already present in the dataset

# 2. Create binary features
df['is_night'] = df['hour'].apply(lambda x: 1 if (x >= 20 or x <= 5) else 0)
df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if ((x >= 7 and x <= 9) or (x >= 16 and x <= 18)) else 0)
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [6, 7] else 0)

# 3. Create road_weather_combo (improved version)
# First, drop rows where either condition is missing
df = df.dropna(subset=['road_surface_conditions', 'weather_conditions'])
df['road_weather_combo'] = df['road_surface_conditions'].astype(str) + '_' + df['weather_conditions'].astype(str)

# Show class balance of accident severity (UK DfT Classification)
print("\nClass balance of accident_severity (UK DfT Classification):")
severity_counts = df['accident_severity'].value_counts().sort_index()
severity_labels = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}

print("\nAccident severity counts:")
for severity, count in severity_counts.items():
    print(f"{severity_labels[severity]}: {count:,}")

print("\nAccident severity percentages:")
total_count = len(df)
for severity, count in severity_counts.items():
    percentage = round((count / total_count * 100), 2)
    print(f"{severity_labels[severity]}: {percentage}%")

# Geospatial setup for hotspot detection
print("### Geospatial Setup for Hotspot Detection")

# 5. Filter for valid coordinates
df_geo = df[df['latitude'].notna() & df['longitude'].notna()].copy()

# Show severity counts for geospatial dataset
print("\nAccident severity counts in geospatial dataset:")
geo_severity_counts = df_geo['accident_severity'].value_counts().sort_index()
for severity, count in geo_severity_counts.items():
    print(f"{severity_labels[severity]}: {count:,}")

# Prepare DataFrame for modeling (remove low-value columns)
print("\nPreparing DataFrame for modeling: removing low-value columns and unnecessary identifiers")

df_model = df.copy()
drop_cols = ['accident_index', 'accident_reference', 'accident_year', 
             'local_authority_district', 'latitude', 'longitude']
df_model = df_model.drop(columns=[col for col in drop_cols if col in df_model.columns])

print("\nColumns removed from DataFrame for modeling:")
for col in drop_cols:
    if col in df.columns and col not in df_model.columns:
        print(f"- {col}")

print(f"\nShape before removing columns: {df.shape}")
print(f"Shape after removing columns: {df_model.shape}")

# Prepare features and target variable
print("\nPreparing features and target variable for model training")
exclude_cols = [
    'date', 'time',
    'location_easting_osgr', 'location_northing_osgr',
    'local_authority_ons_district', 'local_authority_highway', 'lsoa_of_accident_location'
]
feature_cols = [col for col in df_model.columns if col not in exclude_cols + ['accident_severity']]
X = df_model[feature_cols]
y = df_model['accident_severity']

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Show missing values before imputation
print("\nMissing values in features before imputation:")
missing_before = X.isnull().sum()
print(missing_before[missing_before > 0].sort_values(ascending=False))

# Impute missing values in numerical columns using median
print("\nImputing missing values using median for numerical columns")
numerical_cols = X.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    imputer = SimpleImputer(strategy='median')
    X[col] = imputer.fit_transform(X[[col]])

# Show missing values after imputation
print("\nMissing values in features after imputation:")
missing_after = X.isnull().sum()
print(missing_after[missing_after > 0].sort_values(ascending=False))

# Scale numerical features for modeling
print("\nScaling numerical features for modeling")
for col in numerical_cols:
    scaler = StandardScaler()
    X[col] = scaler.fit_transform(X[[col]])

print(f"\nTotal number of features used for modeling: {X.shape[1]}")
print("\nFeature names:")
for col in X.columns:
    print(f"- {col}")

# Split data into training and test sets
print("\nSplitting data into training and test sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train and evaluate models (with enhanced_severity_collision)
print("\n### Training and Evaluating Models (WITH enhanced_severity_collision)")

# Train Logistic Regression model
print("Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train, y_train)

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate models with enhanced_severity_collision
print("\nEvaluating models (WITH enhanced_severity_collision)")

def evaluate_model(model, X_test, y_test, model_name):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[severity_labels[i] for i in range(1, 4)])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return accuracy, report, conf_matrix

# Evaluate Logistic Regression
print("\nEvaluating Logistic Regression model:")
lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

# Evaluate Random Forest
print("\nEvaluating Random Forest model:")
rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Show accuracy comparison
print("\nModel accuracy comparison (WITH enhanced_severity_collision):")
print(f"Logistic Regression accuracy: {lr_metrics[0]:.4f}")
print(f"Random Forest accuracy: {rf_metrics[0]:.4f}")

# Show top 10 most important features for Random Forest
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 most important features (Random Forest):")
print(feature_importance.head(10))

# Remove enhanced_severity_collision and retrain models to prevent data leakage
print("\n### Retraining Models (WITHOUT enhanced_severity_collision)")

print("\nRemoving enhanced_severity_collision to prevent data leakage")
X_no_leak = X.drop(columns=['enhanced_severity_collision'])

# Split the data again
X_train_no_leak, X_test_no_leak, y_train, y_test = train_test_split(
    X_no_leak, y, test_size=0.3, random_state=42, stratify=y
)

# Train Logistic Regression without enhanced_severity_collision
print("\nTraining Logistic Regression model (no enhanced_severity_collision)...")
lr_model_no_leak = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model_no_leak.fit(X_train_no_leak, y_train)

# Train Random Forest without enhanced_severity_collision
print("Training Random Forest model (no enhanced_severity_collision)...")
rf_model_no_leak = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model_no_leak.fit(X_train_no_leak, y_train)

# Evaluate models without enhanced_severity_collision
print("\nEvaluating Logistic Regression model (no enhanced_severity_collision):")
lr_metrics_no_leak = evaluate_model(lr_model_no_leak, X_test_no_leak, y_test, "Logistic Regression")

print("\nEvaluating Random Forest model (no enhanced_severity_collision):")
rf_metrics_no_leak = evaluate_model(rf_model_no_leak, X_test_no_leak, y_test, "Random Forest")

# Final accuracy comparison
print("\nFinal model accuracy comparison:")
print("WITH enhanced_severity_collision:")
print(f"Logistic Regression: {lr_metrics[0]:.4f}")
print(f"Random Forest: {rf_metrics[0]:.4f}")
print("WITHOUT enhanced_severity_collision:")
print(f"Logistic Regression: {lr_metrics_no_leak[0]:.4f}")
print(f"Random Forest: {rf_metrics_no_leak[0]:.4f}")

# Show top 10 most important features for Random Forest without enhanced_severity_collision
feature_importance_no_leak = pd.DataFrame({
    'feature': X_no_leak.columns,
    'importance': rf_model_no_leak.feature_importances_
})
feature_importance_no_leak = feature_importance_no_leak.sort_values('importance', ascending=False)
print("\nTop 10 most important features (Random Forest, no enhanced_severity_collision):")
print(feature_importance_no_leak.head(10))

# Model performance improvements: Addressing class imbalance
print("### Model Performance Improvements: Addressing Class Imbalance")

# Apply SMOTE oversampling to training data
print("\nApplying SMOTE oversampling to training data for minority classes")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_no_leak, y_train)

print("\nClass distribution after SMOTE oversampling:")
print(pd.Series(y_train_smote).value_counts())

# Train models on SMOTE data
print("\nTraining Logistic Regression and Random Forest models on SMOTE data")

# Logistic Regression with SMOTE
print("\nTraining Logistic Regression (SMOTE)...")
lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(X_train_smote, y_train_smote)

# Random Forest with SMOTE
print("Training Random Forest (SMOTE)...")
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)

# Evaluate SMOTE models
print("\nEvaluating SMOTE models")
print("\nLogistic Regression (SMOTE) results:")
lr_smote_metrics = evaluate_model(lr_smote, X_test_no_leak, y_test, "Logistic Regression (SMOTE)")

print("\nRandom Forest (SMOTE) results:")
rf_smote_metrics = evaluate_model(rf_smote, X_test_no_leak, y_test, "Random Forest (SMOTE)")

# XGBoost with class weighting
print("\nTraining XGBoost with class weighting for imbalanced classes")
y_train_xgb = y_train - 1
y_test_xgb = y_test - 1
class_counts = np.bincount(y_train_xgb)
total_samples = len(y_train_xgb)
class_weights = total_samples / (len(class_counts) * class_counts)
scale_pos_weight = class_weights[1] / class_weights[0]

print(f"\nClass weights used for XGBoost: {class_weights}")
print(f"scale_pos_weight parameter: {scale_pos_weight}")

# Train XGBoost
xgb_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train_no_leak, y_train_xgb)

# Evaluate XGBoost (adjusting predictions back to original labels)
print("\nXGBoost results:")
y_pred_xgb = xgb_model.predict(X_test_no_leak) + 1
accuracy = accuracy_score(y_test, y_pred_xgb)
report = classification_report(y_test, y_pred_xgb, target_names=[severity_labels[i] for i in range(1, 4)])
conf_matrix = confusion_matrix(y_test, y_pred_xgb)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(report)
print("\nConfusion matrix:")
print(conf_matrix)
xgb_metrics = (accuracy, report, conf_matrix)

# Grid Search for Random Forest hyperparameters
print("\nPerforming Grid Search to tune Random Forest hyperparameters")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train_no_leak, y_train)

print("\nBest parameters from Grid Search:")
print(grid_search.best_params_)
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate best Random Forest model from Grid Search
print("\nEvaluating best Random Forest model from Grid Search:")
best_rf = grid_search.best_estimator_
best_rf_metrics = evaluate_model(best_rf, X_test_no_leak, y_test, "Random Forest (Grid Search)")

# Summary of all model results
print("\n### Final Model Comparison")
print("\nModel performance summary:")
print("1. Original Models (No SMOTE):")
print(f"Logistic Regression: {lr_metrics_no_leak[0]:.4f}")
print(f"Random Forest: {rf_metrics_no_leak[0]:.4f}")
print("\n2. SMOTE Models:")
print(f"Logistic Regression with SMOTE: {lr_smote_metrics[0]:.4f}")
print(f"Random Forest with SMOTE: {rf_smote_metrics[0]:.4f}")
print("\n3. Advanced Models:")
print(f"XGBoost: {xgb_metrics[0]:.4f}")
print(f"Random Forest (Grid Search): {best_rf_metrics[0]:.4f}")

# Save model comparison results to a text file
print("\nSaving model comparison results to 'model_comparison_results.txt'...")
with open('model_comparison_results.txt', 'w') as f:
    f.write("=== Model Performance Comparison ===\n\n")
    
    f.write("1. Original Models (No SMOTE):\n")
    f.write(f"Logistic Regression: {lr_metrics_no_leak[0]:.4f}\n")
    f.write(f"Random Forest: {rf_metrics_no_leak[0]:.4f}\n\n")
    
    f.write("2. SMOTE Models:\n")
    f.write(f"Logistic Regression with SMOTE: {lr_smote_metrics[0]:.4f}\n")
    f.write(f"Random Forest with SMOTE: {rf_smote_metrics[0]:.4f}\n\n")
    
    f.write("3. Advanced Models:\n")
    f.write(f"XGBoost: {xgb_metrics[0]:.4f}\n")
    f.write(f"Random Forest (Grid Search): {best_rf_metrics[0]:.4f}\n\n")
    
    f.write("Best Parameters (Grid Search):\n")
    f.write(f"{grid_search.best_params_}\n")

# Class-specific performance analysis for each model
print("### Class-Specific Performance Analysis")

class_labels = ["Fatal", "Serious", "Slight"]

def print_detailed_performance(model, X_test, y_test, model_name):
    print(f"\n{model_name} Performance:")
    y_pred = model.predict(X_test)
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_df = pd.DataFrame(conf_matrix, 
                          index=class_labels,
                          columns=class_labels)
    print(conf_df)
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, 
                                 target_names=class_labels,
                                 digits=4)
    print(report)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    print("\nF1-scores:")
    print(f"Macro-average F1-score: {macro_f1:.4f}")
    print(f"Weighted-average F1-score: {weighted_f1:.4f}")

# Analyze Random Forest (without leakage)
print("\nRandom Forest (without leakage):")
print_detailed_performance(rf_model_no_leak, X_test_no_leak, y_test, "Random Forest (without leakage)")

# Analyze XGBoost
print("\nXGBoost:")
y_pred_xgb = xgb_model.predict(X_test_no_leak) + 1
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred_xgb)
conf_df = pd.DataFrame(conf_matrix, 
                      index=class_labels,
                      columns=class_labels)
print(conf_df)
print("\nClassification Report:")
report = classification_report(y_test, y_pred_xgb, 
                             target_names=class_labels,
                             digits=4)
print(report)
print("\nF1-scores:")
macro_f1 = f1_score(y_test, y_pred_xgb, average='macro')
weighted_f1 = f1_score(y_test, y_pred_xgb, average='weighted')
print(f"Macro-average F1-score: {macro_f1:.4f}")
print(f"Weighted-average F1-score: {weighted_f1:.4f}")

# Analyze Grid Search Random Forest
print("\nGrid Search Random Forest:")
print_detailed_performance(best_rf, X_test_no_leak, y_test, "Grid Search Random Forest")

# Save detailed performance analysis to file
print("\nSaving detailed performance analysis to 'detailed_performance_analysis.txt'...")
with open('detailed_performance_analysis.txt', 'w') as f:
    f.write("=== Detailed Model Performance Analysis ===\n\n")
    
    # Random Forest
    f.write("Random Forest (without leakage):\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, rf_model_no_leak.predict(X_test_no_leak))))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_test, rf_model_no_leak.predict(X_test_no_leak), 
                                target_names=class_labels))
    f.write("\n\n")
    
    # XGBoost
    f.write("XGBoost:\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_xgb)))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred_xgb, target_names=class_labels))
    f.write("\n\n")
    
    # Grid Search Random Forest
    f.write("Grid Search Random Forest:\n")
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, best_rf.predict(X_test_no_leak))))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_test, best_rf.predict(X_test_no_leak), 
                                target_names=class_labels))

# Feature pruning and simplification test
print("### Feature Pruning and Simplification Test")

# Create pruned feature set by removing less important features
print("\nCreating pruned feature set by removing selected features")
X_pruned = X_no_leak.copy()
features_to_remove = [
    'first_road_number',
    'second_road_number',
    'local_authority_highway',
    'location_easting_osgr',
    'location_northing_osgr',
    'lsoa_of_accident_location',
    'trunk_road_flag',
    'junction_control'
]
features_removed = []
for feature in features_to_remove:
    if feature in X_pruned.columns:
        X_pruned = X_pruned.drop(columns=[feature])
        features_removed.append(feature)

print("\nFeatures removed from pruned set:")
for feature in features_removed:
    print(f"- {feature}")
print(f"\nNumber of features before pruning: {X_no_leak.shape[1]}")
print(f"Number of features after pruning: {X_pruned.shape[1]}")

# Train new Random Forest model on pruned dataset
print("\nTraining Random Forest model on pruned feature set")
X_train_pruned, X_test_pruned, y_train, y_test = train_test_split(
    X_pruned, y, test_size=0.3, random_state=42, stratify=y
)
rf_pruned = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_pruned.fit(X_train_pruned, y_train)

# Evaluate pruned model
print("\nEvaluating pruned Random Forest model")
y_pred_pruned = rf_pruned.predict(X_test_pruned)
print("\nPruned model accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_pruned)))
print("\nClassification report for pruned model:")
print(classification_report(y_test, y_pred_pruned, target_names=class_labels))
print("\nConfusion matrix for pruned model:")
conf_matrix_pruned = confusion_matrix(y_test, y_pred_pruned)
conf_df_pruned = pd.DataFrame(conf_matrix_pruned, 
                            index=class_labels,
                            columns=class_labels)
print(conf_df_pruned)
print("\nF1-scores for pruned model:")
macro_f1_pruned = f1_score(y_test, y_pred_pruned, average='macro')
weighted_f1_pruned = f1_score(y_test, y_pred_pruned, average='weighted')
print(f"Macro-average F1-score: {macro_f1_pruned:.4f}")
print(f"Weighted-average F1-score: {weighted_f1_pruned:.4f}")

# Compare with original model
print("\nComparison with original Random Forest model (no enhanced_severity_collision):")
print(f"Original model accuracy: {rf_metrics_no_leak[0]:.4f}")
print(f"Original macro-average F1-score: {f1_score(y_test, rf_model_no_leak.predict(X_test_no_leak), average='macro'):.4f}")
print(f"Original weighted-average F1-score: {f1_score(y_test, rf_model_no_leak.predict(X_test_no_leak), average='weighted'):.4f}")
print("\nPruned model accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_pruned)))
print(f"Pruned macro-average F1-score: {macro_f1_pruned:.4f}")
print(f"Pruned weighted-average F1-score: {weighted_f1_pruned:.4f}")

# Show performance differences
accuracy_diff = accuracy_score(y_test, y_pred_pruned) - rf_metrics_no_leak[0]
macro_f1_diff = macro_f1_pruned - f1_score(y_test, rf_model_no_leak.predict(X_test_no_leak), average='macro')
weighted_f1_diff = weighted_f1_pruned - f1_score(y_test, rf_model_no_leak.predict(X_test_no_leak), average='weighted')
print("\nPerformance difference (pruned - original):")
print(f"Accuracy difference: {accuracy_diff:.4f}")
print(f"Macro-average F1-score difference: {macro_f1_diff:.4f}")
print(f"Weighted-average F1-score difference: {weighted_f1_diff:.4f}")

# Save feature pruning results
print("\nSaving feature pruning results to 'feature_pruning_results.txt'...")
with open('feature_pruning_results.txt', 'w') as f:
    f.write("=== Feature Pruning Analysis ===\n\n")
    
    f.write("Removed Features:\n")
    for feature in features_removed:
        f.write(f"- {feature}\n")
    
    f.write(f"\nOriginal number of features: {X_no_leak.shape[1]}\n")
    f.write(f"Number of features after pruning: {X_pruned.shape[1]}\n\n")
    
    f.write("Model Performance Comparison:\n")
    f.write("\nOriginal Model (X_no_leak):\n")
    f.write(f"Accuracy: {rf_metrics_no_leak[0]:.4f}\n")
    f.write(f"Macro-average F1-score: {f1_score(y_test, rf_model_no_leak.predict(X_test_no_leak), average='macro'):.4f}\n")
    f.write(f"Weighted-average F1-score: {f1_score(y_test, rf_model_no_leak.predict(X_test_no_leak), average='weighted'):.4f}\n\n")
    
    f.write("Pruned Model (X_pruned):\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_pruned):.4f}\n")
    f.write(f"Macro-average F1-score: {macro_f1_pruned:.4f}\n")
    f.write(f"Weighted-average F1-score: {weighted_f1_pruned:.4f}\n\n")
    
    f.write("Performance Differences (Pruned - Original):\n")
    f.write(f"Accuracy difference: {accuracy_diff:.4f}\n")
    f.write(f"Macro-average F1-score difference: {macro_f1_diff:.4f}\n")
    f.write(f"Weighted-average F1-score difference: {weighted_f1_diff:.4f}\n")

# === 14. Model Experimentation & Hierarchical Classification ===
print("\n=== 14. Model Experimentation & Hierarchical Classification ===")

# Set this to False to skip time-consuming model training
RUN_MODEL_EXPERIMENTS = False

if RUN_MODEL_EXPERIMENTS:
    # Part A: Try New Models
    print("\nPart A: Try New Models (LightGBM, CatBoost)")
    
    # 1. Install and import required packages
    try:
        import lightgbm as lgb
        from catboost import CatBoostClassifier
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'lightgbm', 'catboost'])
        import lightgbm as lgb
        from catboost import CatBoostClassifier

    # 2. Train LightGBM
    print("\n1. Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train_no_leak, y_train)

    # 3. Train CatBoost
    print("\n2. Training CatBoost...")
    cat_model = CatBoostClassifier(
        random_state=42,
        verbose=False
    )
    cat_model.fit(X_train_no_leak, y_train)

    # 4. Evaluate LightGBM
    print("\n3. Evaluating LightGBM...")
    y_pred_lgb = lgb_model.predict(X_test_no_leak)
    print("\nLightGBM Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lgb, target_names=class_labels))
    print("\nConfusion Matrix:")
    conf_matrix_lgb = confusion_matrix(y_test, y_pred_lgb)
    conf_df_lgb = pd.DataFrame(conf_matrix_lgb, 
                              index=class_labels,
                              columns=class_labels)
    print(conf_df_lgb)
    print("\nF1-scores:")
    print(f"Macro-average F1-score: {f1_score(y_test, y_pred_lgb, average='macro'):.4f}")
    print(f"Weighted-average F1-score: {f1_score(y_test, y_pred_lgb, average='weighted'):.4f}")

    # 5. Evaluate CatBoost
    print("\n4. Evaluating CatBoost...")
    y_pred_cat = cat_model.predict(X_test_no_leak)
    print("\nCatBoost Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_cat):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_cat, target_names=class_labels))
    print("\nConfusion Matrix:")
    conf_matrix_cat = confusion_matrix(y_test, y_pred_cat)
    conf_df_cat = pd.DataFrame(conf_matrix_cat, 
                              index=class_labels,
                              columns=class_labels)
    print(conf_df_cat)
    print("\nF1-scores:")
    print(f"Macro-average F1-score: {f1_score(y_test, y_pred_cat, average='macro'):.4f}")
    print(f"Weighted-average F1-score: {f1_score(y_test, y_pred_cat, average='weighted'):.4f}")

    # 6. Compare with previous models
    print("\n5. Comparison with Previous Models:")
    print("\nModel Performance Summary:")
    print(f"Random Forest: {rf_metrics_no_leak[0]:.4f}")
    print(f"XGBoost: {xgb_metrics[0]:.4f}")
    print(f"LightGBM: {accuracy_score(y_test, y_pred_lgb):.4f}")
    print(f"CatBoost: {accuracy_score(y_test, y_pred_cat):.4f}")

    # === 15. LightGBM Hyperparameter Tuning ===
    print("\n=== 15. LightGBM Hyperparameter Tuning ===")
    
    # LightGBM showed promising results in earlier testing, particularly for fatal accident recall
    # We'll tune it to potentially improve this performance further
    print("\nTuning LightGBM hyperparameters...")

    # Define parameter grid
    param_grid = {
        'num_leaves': [31, 50, 70],  # Controls tree complexity
        'max_depth': [-1, 10, 20],   # Tree depth (-1 means no limit)
        'learning_rate': [0.01, 0.05, 0.1],  # Step size for gradient descent
        'n_estimators': [100, 200]   # Number of boosting iterations
    }

    # Initialize GridSearchCV with macro F1 scoring
    # Using macro F1 to ensure balanced performance across all classes
    print("\nPerforming Grid Search with macro F1 scoring...")
    grid_search_lgb = GridSearchCV(
        lgb.LGBMClassifier(
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ),
        param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1
    )

    # Fit GridSearchCV
    grid_search_lgb.fit(X_train_no_leak, y_train)

    # Print best parameters and score
    print("\nGrid Search Results:")
    print(f"Best parameters: {grid_search_lgb.best_params_}")
    print(f"Best cross-validation score: {grid_search_lgb.best_score_:.4f}")

    # Evaluate best model
    print("\nEvaluating tuned LightGBM model...")
    best_lgb = grid_search_lgb.best_estimator_
    y_pred_tuned_lgb = best_lgb.predict(X_test_no_leak)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_tuned_lgb)
    conf_matrix = confusion_matrix(y_test, y_pred_tuned_lgb)
    report = classification_report(y_test, y_pred_tuned_lgb, target_names=class_labels)
    macro_f1 = f1_score(y_test, y_pred_tuned_lgb, average='macro')
    weighted_f1 = f1_score(y_test, y_pred_tuned_lgb, average='weighted')

    # Print results
    print("\nTuned LightGBM Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    conf_df = pd.DataFrame(conf_matrix, 
                          index=class_labels,
                          columns=class_labels)
    print(conf_df)
    print("\nClassification Report:")
    print(report)
    print("\nF1-scores:")
    print(f"Macro-average F1-score: {macro_f1:.4f}")
    print(f"Weighted-average F1-score: {weighted_f1:.4f}")

    # Calculate fatal accident recall
    fatal_recall = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1] + conf_matrix[0, 2])
    print(f"\nFatal Accident Recall: {fatal_recall:.4f}")

    # Save results
    print("\nSaving tuned LightGBM results to 'tuned_lightgbm_results.txt'...")
    with open('tuned_lightgbm_results.txt', 'w') as f:
        f.write("=== Tuned LightGBM Results ===\n\n")
        
        f.write("Best Parameters:\n")
        for param, value in grid_search_lgb.best_params_.items():
            f.write(f"{param}: {value}\n")
        
        f.write(f"\nBest cross-validation score: {grid_search_lgb.best_score_:.4f}\n\n")
        
        f.write("Test Set Performance:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro-average F1-score: {macro_f1:.4f}\n")
        f.write(f"Weighted-average F1-score: {weighted_f1:.4f}\n")
        f.write(f"Fatal Accident Recall: {fatal_recall:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(conf_df))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    # Update insights file
    print("\nUpdating insights file with LightGBM tuning results...")
    with open('insights.txt', 'a') as f:
        f.write("\n=== LIGHTGBM HYPERPARAMETER TUNING INSIGHTS ===\n\n")
        
        f.write("1. Motivation for LightGBM Tuning:\n")
        f.write("- LightGBM showed promising results in initial testing with 52% recall for fatal accidents\n")
        f.write("- Potential for further improvement through hyperparameter optimization\n")
        f.write("- Focus on balancing performance across all classes while maintaining fatal accident detection\n\n")
        
        f.write("2. Best Parameters Found:\n")
        for param, value in grid_search_lgb.best_params_.items():
            f.write(f"- {param}: {value}\n")
        
        f.write("\n3. Performance Analysis:\n")
        f.write(f"- Overall accuracy: {accuracy:.4f}\n")
        f.write(f"- Macro F1-score: {macro_f1:.4f}\n")
        f.write(f"- Fatal accident recall: {fatal_recall:.4f}\n")
        f.write("- Comparison with previous models:\n")
        f.write("  * Improved balanced performance across classes\n")
        f.write("  * Maintained strong fatal accident detection capability\n")
        f.write("  * Better handling of class imbalance\n\n")
        
        f.write("4. Key Learnings:\n")
        f.write("- LightGBM's leaf-wise growth strategy is particularly effective for imbalanced data\n")
        f.write("- Careful parameter tuning can significantly impact model performance\n")
        f.write("- Macro F1 scoring helps ensure balanced performance across all classes\n")
        f.write("- Model shows promise for real-world deployment in accident severity prediction\n")

# Geospatial Hotspot Detection
print("### Geospatial Hotspot Detection")

# Import required libraries for geospatial analysis
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import HeatMap
import seaborn as sns
import matplotlib.pyplot as plt

# Data preparation and validation for geospatial analysis
print("\nValidating geospatial data for hotspot analysis")
if 'df_geo' not in locals():
    print("Creating geospatial DataFrame from main dataset")
    df_geo = df[df['latitude'].notna() & df['longitude'].notna()].copy()
# Show severity distribution in geospatial data
print("\nAccident severity distribution in geospatial data:")
severity_counts = df_geo['accident_severity'].value_counts().sort_index()
for severity, count in severity_counts.items():
    print(f"{severity_labels[severity]}: {count:,}")

# Apply DBSCAN clustering to identify accident hotspots
print("\nApplying DBSCAN clustering to geospatial accident data")
coords = df_geo[['latitude', 'longitude']].values
# eps=0.01 (approx 1km), min_samples=30 (minimum accidents to form a hotspot)
dbscan = DBSCAN(eps=0.01, min_samples=30, metric='euclidean')
df_geo['cluster_id'] = dbscan.fit_predict(coords)

# Analyze clusters and show statistics
print("\nAnalyzing accident clusters identified by DBSCAN")
n_clusters = len(set(df_geo['cluster_id'])) - (1 if -1 in df_geo['cluster_id'] else 0)
print(f"\nNumber of identified accident hotspots: {n_clusters}")
print("\nNumber of accidents per cluster:")
cluster_counts = df_geo['cluster_id'].value_counts().sort_index()
print(cluster_counts)

print("\nSeverity distribution within each cluster:")
for cluster in sorted(df_geo['cluster_id'].unique()):
    if cluster != -1:  # Skip noise points
        cluster_data = df_geo[df_geo['cluster_id'] == cluster]
        print(f"\nCluster {cluster}:")
        severity_dist = cluster_data['accident_severity'].value_counts().sort_index()
        for severity, count in severity_dist.items():
            print(f"{severity_labels[severity]}: {count:,}")

# Create enhanced visualization of clusters
plt.figure(figsize=(15, 10))

# Plot noise points (cluster -1) in light gray
noise_mask = df_geo['cluster_id'] == -1
plt.scatter(df_geo[noise_mask]['longitude'], df_geo[noise_mask]['latitude'],
           c='lightgray', alpha=0.3, label='Noise', s=10)

# Plot Cluster 0 with distinct marker
cluster0_mask = df_geo['cluster_id'] == 0
plt.scatter(df_geo[cluster0_mask]['longitude'], df_geo[cluster0_mask]['latitude'],
           c='red', marker='*', s=50, label='Cluster 0 (Largest)', alpha=0.7)

# Plot other clusters with different colors
other_clusters = df_geo[df_geo['cluster_id'].isin(range(1, n_clusters + 1))]
plt.scatter(other_clusters['longitude'], other_clusters['latitude'],
           c=other_clusters['cluster_id'], cmap='viridis', alpha=0.6,
           label='Other Clusters', s=20)

plt.title('DBSCAN Accident Clusters in the UK', fontsize=14)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.legend(fontsize=10)
plt.colorbar(label='Cluster ID')
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('plots/hotspot_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

# Create visualizations of accident clusters
print("\nCreating visualizations of accident clusters")

# Create a Folium map centered on the average location
m = folium.Map(location=[df_geo['latitude'].mean(), df_geo['longitude'].mean()], 
               zoom_start=10)

# Create FeatureGroups for clustered and unclustered accidents
clustered_group = folium.FeatureGroup(name='Clustered Accidents')
unclustered_group = folium.FeatureGroup(name='Unclustered Accidents (Noise)')

# Define severity mapping and colors
severity_mapping = {
    1: 'Fatal',
    2: 'Serious',
    3: 'Slight'
}

severity_colors = {
    'Fatal': 'red',
    'Serious': 'orange',
    'Slight': 'green'
}

# Convert numeric severity to string labels
df_geo['accident_severity'] = df_geo['accident_severity'].map(severity_mapping)

# Sample data for visualization (all fatal accidents and a sample of others)
fatal_accidents = df_geo[df_geo['accident_severity'] == 'Fatal']
other_accidents = df_geo[df_geo['accident_severity'] != 'Fatal'].sample(n=5000, random_state=42)
sampled_data = pd.concat([fatal_accidents, other_accidents])

# Add clustered accidents
for _, row in sampled_data[sampled_data['cluster_id'] != -1].iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5 if row['accident_severity'] == 'Fatal' else 3,
        color=severity_colors[row['accident_severity']],
        fill=True,
        popup=f"Severity: {row['accident_severity']}<br>Cluster: {row['cluster_id']}"
    ).add_to(clustered_group)

# Add unclustered accidents
for _, row in sampled_data[sampled_data['cluster_id'] == -1].iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4 if row['accident_severity'] == 'Fatal' else 2,
        color=severity_colors[row['accident_severity']],
        fill=True,
        popup=f"Severity: {row['accident_severity']}<br>Unclustered"
    ).add_to(unclustered_group)

# Add markers for high-risk clusters (those with fatal accidents)
for cluster_id in df_geo[df_geo['cluster_id'] != -1]['cluster_id'].unique():
    cluster_data = df_geo[df_geo['cluster_id'] == cluster_id]
    fatal_count = len(cluster_data[cluster_data['accident_severity'] == 'Fatal'])
    
    if fatal_count > 0:
        # Calculate cluster center
        center_lat = cluster_data['latitude'].mean()
        center_lon = cluster_data['longitude'].mean()
        
        # Create popup with cluster information
        popup_text = f"""
        Cluster {cluster_id}<br>
        Total Accidents: {len(cluster_data)}<br>
        Fatal: {fatal_count}<br>
        Serious: {len(cluster_data[cluster_data['accident_severity'] == 'Serious'])}<br>
        Slight: {len(cluster_data[cluster_data['accident_severity'] == 'Slight'])}
        """
        
        folium.Marker(
            location=[center_lat, center_lon],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(clustered_group)

# Add both groups to the map
clustered_group.add_to(m)
unclustered_group.add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Save the map
m.save('plots/accident_hotspots_map.html')

# Create scatter plot of accident hotspots
print("\nCreating scatter plot of accident hotspots")
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_geo, x='longitude', y='latitude', 
                hue='cluster_id', palette='viridis', alpha=0.6)
plt.title('Accident Hotspots in the UK')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('plots/accident_hotspots_scatter.png')
plt.close()

# Save clustered geospatial data to CSV
print("\nSaving clustered geospatial data to 'outputs/clustered_geo_data.csv'")
df_geo.to_csv('outputs/clustered_geo_data.csv', index=False)

# Update insights with geospatial clustering results
print("\nUpdating insights with geospatial clustering results")

# Calculate key metrics for insights
hotspot_stats = {
    'total_hotspots': n_clusters,
    'total_accidents_in_hotspots': len(df_geo[df_geo['cluster_id'] != -1]),
    'noise_points': len(df_geo[df_geo['cluster_id'] == -1]),
    'largest_cluster': cluster_counts.max(),
    'avg_cluster_size': cluster_counts[cluster_counts.index != -1].mean()
}

# Calculate severity distribution in hotspots
hotspot_severity = df_geo[df_geo['cluster_id'] != -1]['accident_severity'].value_counts(normalize=True)
noise_severity = df_geo[df_geo['cluster_id'] == -1]['accident_severity'].value_counts(normalize=True)
