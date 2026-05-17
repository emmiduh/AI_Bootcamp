import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Titanic dataset from a URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Fill missing values before model training
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("Dataset Preview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDataset Summary:")
print(df.describe())

# Apply one-hot encoding
df_one_hot = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Displey the encoded dataset
print("\n One-Hot Encoded dataset:")
print(df_one_hot.head())

# Apply lable encoding
label_encoder = LabelEncoder()
df['Pclass_encoded'] = label_encoder.fit_transform(df['Pclass'])

# Display encoded dataset
print("\n Label Encoded Dataset:")
print(df[['Pclass', 'Pclass_encoded']].head())

# Apply Frequency ENcoding
df['Ticket_frequency'] = df['Ticket'].map(df['Ticket'].value_counts())

# Display frequency encoded features
print("\n Frequency Encoded Feature: ")
print(df[['Ticket', 'Ticket_frequency']].head())

X = df_one_hot.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
y = df['Survived']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logisitc regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy with One-Hot Encoding:", accuracy_score(y_test, y_pred))