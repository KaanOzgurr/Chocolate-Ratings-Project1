import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import os

# --- Load Dataset ---
if not os.path.exists("chocolate_ratings.csv"):
    print("Error: chocolate_ratings.csv not found!")
    exit()

df = pd.read_csv("chocolate_ratings.csv")

# --- Clean column names ---
df.columns = df.columns.str.strip().str.replace(';', '')
df.rename(columns={'Company (Manufacturer)': 'Company'}, inplace=True)

print("Columns in dataset:")
print(df.columns)

# --- Preprocessing ---
# Clean & convert Cocoa Percent to numeric
df['Cocoa Percent'] = (
    df['Cocoa Percent']
    .astype(str)
    .str.strip()
    .str.replace('%', '', regex=False)
)
df['Cocoa Percent'] = pd.to_numeric(df['Cocoa Percent'], errors='coerce')
df = df.dropna(subset=['Cocoa Percent'])

# Clean & convert Rating to numeric
df['Rating'] = (
    df['Rating']
    .astype(str)
    .str.strip()
    .str.replace(';', '', regex=False)
)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df.dropna(subset=['Rating'])

# Encode categorical features
le_company = LabelEncoder()
df['Company Encoded'] = le_company.fit_transform(df['Company'])

le_origin = LabelEncoder()
df['Bean Origin Encoded'] = le_origin.fit_transform(df['Country of Bean Origin'].astype(str))

# --- Correlation ---
corr = df[['Rating', 'Cocoa Percent', 'Company Encoded', 'Bean Origin Encoded']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

print("\nCorrelation with Rating:")
print(corr['Rating'].sort_values(ascending=False))

# --- Feature Importance with Random Forest ---
X = df[['Company Encoded', 'Bean Origin Encoded', 'Cocoa Percent']]
y = df['Rating']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False, inplace=True)
print("\nFeature Importances:")
print(importances)

importances.plot(kind='bar', title='Feature Importance')
plt.show()

# --- Recursive Feature Elimination (RFE) ---
model = LinearRegression()
rfe = RFE(model, n_features_to_select=3)
rfe.fit(X, y)
print("\nSelected Features by RFE:", X.columns[rfe.support_])

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Visualizations ---
plt.figure(figsize=(8,5))
sns.histplot(df['Rating'], bins=20, kde=True, color='brown')
plt.title('Distribution of Chocolate Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='Cocoa Percent', y='Rating', data=df)
plt.title('Cocoa Percent vs Rating')
plt.xlabel('Cocoa Percent')
plt.ylabel('Rating')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x=pd.cut(df['Cocoa Percent'], bins=5), y='Rating', data=df)
plt.xticks(rotation=45)
plt.title('Rating vs Cocoa Percent Bins')
plt.xlabel('Cocoa Percent Bins')
plt.ylabel('Rating')
plt.show()

top_companies = df.groupby('Company')['Rating'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_companies.values, y=top_companies.index, palette='viridis')
plt.title('Top 10 Companies by Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Company')
plt.show()

# --- Model Training & Evaluation ---
# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("\nLinear Regression Results:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2:", r2_score(y_test, y_pred_lr))

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Results:")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2:", r2_score(y_test, y_pred_rf))

# --- Confirm columns after encoding ---
print("\nColumns in processed dataset:")
print(df.columns)
