#step 1: Importing all necessary libraries and data-set

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# step 2: Loading the dataset,undertsanding the data
df = pd.read_csv("C:/Users/Prabhat Dey/OneDrive/Desktop/student.csv")   #Data Understanding
print("First 5 rows:\n", df.head())
print("\nData Info:")
print(df.info())


# step3: Preprocess data — create pass_fail target(with the help of G1,G2 AND G3), handle missing values, and encode categorical features
df['pass_fail'] = df[['g1','g2','g3']].mean(axis=1).apply(lambda x: 1 if x >= 61 else 0) # Create pass fail based on average of G1, G2, G3
print(df[['g1', 'g2', 'g3', 'pass_fail']].head())
df = df.dropna(subset=['pass_fail'])
df.fillna(method='ffill',inplace=True)   # Fill missing values (basic handling)

le = LabelEncoder()                        # Encode categorical features
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])


#Step4:Plotting the distribution of pass vs fail, explored how study time and absences relate to performance, and generated a correlation matrix to identify which features are most strongly associated with student success
sns.countplot(x='pass_fail', data=df)   # Distribution of pass vs fail
plt.title('Distribution of Pass vs Fail')
plt.show()

sns.boxplot(x='pass_fail', y='study_time', data=df)  # Study time vs pass rate
plt.title('Study Time vs Pass Rate')
plt.show()

sns.boxplot(x='pass_fail', y='absences', data=df)   # Absences vs pass rate
plt.title('Absences vs Pass Rate')
plt.show()

plt.figure(figsize=(10, 6))   # Correlation Matrix
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#Step 5: Splitting the data into train and test data sets and putting it through a Logistic Regression model , evaluate required metrics.
X = df.drop(['pass_fail'], axis=1)  #Model building
y = df['pass_fail']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Evaluation
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
