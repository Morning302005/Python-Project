#STEP 1: Importing all necessary libraries and data-set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#STEP2: Loading the dataset,undertsanding the data and checking for any missing value (if present and then handling the missing value )
df = pd.read_csv("C:/Users/Prabhat Dey/OneDrive/Desktop/Housingdata.csv")
print("First 5 rows:\n", df.head())   #Data Understanding
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
df = df.dropna()

#STEP 3: Creating a scatter plot and a correlation heatmap to visually explore how different features relate to house prices and to identify patterns useful for model training.
#Data Visualization
sns.histplot(df['MEDV'], kde=True)   #Distribution of house prices
plt.title("Distribution of House Prices")
plt.xlabel("MEDV")
plt.ylabel("Frequency")
plt.show()

sns.scatterplot(x='CRIM', y='MEDV', data=df)    # Crime Rate vs Price
plt.title("Crime Rate vs House Price")
plt.xlabel("Crime Rate (CRIM)")
plt.ylabel("MEDV")
plt.show()

plt.figure(figsize=(10, 8))   # Correlation Matrix
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='RM', y='MEDV', data=df)  #RM vs MEDV
plt.title("RM vs MEDV")
plt.xlabel("RM")
plt.ylabel("MEDV")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))   #LSTAT vs MEDV
sns.scatterplot(x='LSTAT', y='MEDV', data=df)
plt.title("LSTAT vs MEDV)")
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))  #PTRATIO vs MEDV
sns.scatterplot(x='PTRATIO', y='MEDV', data=df)
plt.title("PTratio vs MEDV")
plt.xlabel("PTRATIO")
plt.ylabel("MEDV")
plt.grid(True)
plt.show()

selected_features = ['RM', 'LSTAT', 'PTRATIO']
target = 'MEDV'
df[selected_features].hist(bins=20, figsize=(10, 5))  #Histograms of selected features
plt.suptitle("Feature Distributions")
plt.show()

#STEP 4: Now we are going to be splitting the data into train and test data sets and putting it through a Linear Regression model and calculate the MSE and R2 score
#Feature and Target Definition
X = df.drop('MEDV', axis=1)  # Use all other columns as features
y = df['MEDV']                # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.grid(True)
plt.show()





