import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv("train.csv")  # Make sure 'train.csv' is in your working directory

# Step 2: Basic Info
print("Shape of dataset:", df.shape)
print("\nColumn info:")
print(df.info())
print("\nStatistical summary:")
print(df.describe())

# Step 3: Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Step 4: Clean data
df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)  # Drop irrelevant columns
df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill missing age with median
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Fill missing Embarked

# Step 5: Visualizations
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title("Age Distribution")
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()
