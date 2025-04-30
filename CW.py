# %%
# Import necessary Python libraries
# 导入必要的Python库
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Read the traffic accident dataset
# 读取交通事故数据集
df = pd.read_csv('dft-road-casualty-statistics-collision-2023.csv', low_memory=False)

# Display basic information about the dataset
# 显示数据集的基本信息
print("Dataset Information / 数据集信息:")
print(df.info())
df.head()

# %%
# Select important features for prediction
# 选取与预测相关的重要字段
selected_cols = [
    'accident_severity', 'number_of_vehicles', 'number_of_casualties',
    'date', 'time', 'day_of_week', 'road_type', 'speed_limit',
    'light_conditions', 'weather_conditions', 'road_surface_conditions',
    'urban_or_rural_area'
]

df_selected = df[selected_cols].copy()

print("Selected features / 选取后的字段：")
print(df_selected.head())

# %%
# Handle missing values
# 针对每一列进行缺失值处理
for col in df_selected.columns:
    if df_selected[col].isnull().sum() > 0:
        if df_selected[col].dtype in ['float64', 'int64']:
            df_selected[col].fillna(df_selected[col].median(), inplace=True)
        else:
            df_selected[col].fillna(df_selected[col].mode()[0], inplace=True)

print("Missing value handling completed / 缺失值处理完成。")

# %%
# Process date and time features
# 处理日期和时间特征
date_series = pd.to_datetime(df_selected['date'], format='%d/%m/%Y', errors='coerce')
time_series = pd.to_datetime(df_selected['time'], format='%H:%M', errors='coerce')

# Convert date to numerical values (days since epoch)
# 将日期转换为数值（自纪元以来的天数）
df_selected['date_numeric'] = (date_series - pd.Timestamp('1970-01-01')).dt.total_seconds() / (24 * 3600)

# Convert time to numerical values (hours since midnight)
# 将时间转换为数值（自午夜以来的小时数）
df_selected['time_numeric'] = time_series.dt.hour + time_series.dt.minute / 60

# Create time-based features
# 创建基于时间的特征
df_selected['month'] = date_series.dt.month
df_selected['hour'] = time_series.dt.hour

df_selected['season'] = df_selected['month'].apply(lambda x: 
    'winter' if x in [12, 1, 2] else
    'spring' if x in [3, 4, 5] else
    'summer' if x in [6, 7, 8] else
    'autumn'
)

# Create rush hour indicator
# 创建高峰时段指标
df_selected['is_rush_hour'] = df_selected['hour'].apply(lambda x: 
    1 if (x >= 7 and x <= 9) or (x >= 16 and x <= 18) else 0
)

# Create day/night indicator
# 创建白天/夜晚指标
df_selected['is_night'] = df_selected['hour'].apply(lambda x: 
    1 if x >= 20 or x <= 5 else 0
)

# Drop original date and time columns
# 删除原始日期和时间列
df_selected = df_selected.drop(['date', 'time'], axis=1)

print("Time-based features created / 基于时间的特征创建完成！")
print("\nColumns in dataset / 数据集中的列：")
print(df_selected.columns)

df_selected.head()

# %%
# Create interaction features
# 创建交互特征
df_selected['vehicles_casualties_ratio'] = df_selected['number_of_vehicles'] / df_selected['number_of_casualties']
df_selected['vehicles_casualties_ratio'] = df_selected['vehicles_casualties_ratio'].replace([np.inf, -np.inf], np.nan)
df_selected['vehicles_casualties_ratio'] = df_selected['vehicles_casualties_ratio'].fillna(0)

# Create road condition combinations
# 创建道路条件组合
df_selected['road_weather_condition'] = df_selected['road_surface_conditions'].astype(str) + '_' + df_selected['weather_conditions'].astype(str)
df_selected['road_light_condition'] = df_selected['road_surface_conditions'].astype(str) + '_' + df_selected['light_conditions'].astype(str)

# Create speed limit categories
# 创建速度限制类别
df_selected['speed_category'] = pd.cut(df_selected['speed_limit'], 
    bins=[0, 30, 50, 70, 100],
    labels=['low', 'medium', 'high', 'very_high']
)

# Create urban/rural with road type combination
# 创建城乡与道路类型组合
df_selected['area_road_type'] = df_selected['urban_or_rural_area'].astype(str) + '_' + df_selected['road_type'].astype(str)

print("Interaction features created / 交互特征创建完成！")

# %%
# Encode categorical features
# 对类别特征进行编码
categorical_features = [
    'road_type', 'light_conditions', 'weather_conditions', 
    'road_surface_conditions', 'urban_or_rural_area',
    'season', 'road_weather_condition', 'road_light_condition',
    'speed_category', 'area_road_type'
]
encoder = LabelEncoder()
for col in categorical_features:
    df_selected[col] = encoder.fit_transform(df_selected[col])

print("Categorical feature encoding completed / 类别特征编码完成！")

# %%
print(df_selected['road_type'])

# %%
# Import modeling libraries
# 导入建模相关库
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Split features and target
# 特征和目标分开
X = df_selected.drop('accident_severity', axis=1)  # Input features / 输入特征
y = df_selected['accident_severity']               # Target variable / 目标变量

# %%
# Split dataset (70% training, 30% testing)
# 切分数据集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training set size / 训练集大小：", X_train.shape)
print("Test set size / 测试集大小：", X_test.shape)

# %%
# Initialize and train decision tree
# 初始化并训练决策树
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print("Decision Tree model training completed / 决策树模型训练完成！")

# %%
# Make predictions on test set
# 在测试集上进行预测
y_pred = model.predict(X_test)

# Evaluate model
# 评估指标输出
print("Model Accuracy / 模型准确率:", accuracy_score(y_test, y_pred))
print("\nClassification Report / 分类详细报告:\n", classification_report(y_test, y_pred))

# %%
# Plot confusion matrix
# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1,2,3], yticklabels=[1,2,3])
plt.xlabel('Predicted Severity')
plt.ylabel('Actual Severity')
plt.title('Confusion Matrix - Decision Tree')
plt.show()

# %%
# Import Random Forest
# 导入随机森林
from sklearn.ensemble import RandomForestClassifier

# %%
# Initialize Random Forest
# 初始化随机森林
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

print("Random Forest model training completed / 随机森林模型训练完成！")

# %%
# Make predictions
# 在测试集上进行预测
y_pred_rf = rf_model.predict(X_test)

# Evaluate model
# 评估指标输出
print("Random Forest Model Accuracy / 随机森林模型准确率:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report / 分类详细报告:\n", classification_report(y_test, y_pred_rf))

# %%
# Plot confusion matrix for Random Forest
# 绘制随机森林混淆矩阵
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=[1,2,3], yticklabels=[1,2,3])
plt.xlabel('Predicted Severity')
plt.ylabel('Actual Severity')
plt.title('Confusion Matrix - Random Forest')
plt.show() 


