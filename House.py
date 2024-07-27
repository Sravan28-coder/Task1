import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, Binarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'house.csv'
df = pd.read_csv(file_path)

# Check columns
print(df.columns)
print(df.head())

# Preprocess the data
# Encode categorical variables
label_encoders = {}
categorical_columns = ['Location']

for column in categorical_columns:
    if column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Convert 'Price' to categorical variable
price_bins = [0, 100000, 200000, np.inf]  # Example bins for categorization
price_labels = ['Low', 'Medium', 'High']
df['PriceCategory'] = pd.cut(df['Price'], bins=price_bins, labels=price_labels)

# Define features and target
columns_to_drop = [col for col in ['HouseID', 'Price', 'PriceCategory'] if col in df.columns]
X = df.drop(columns=columns_to_drop, axis=1)
y = df['PriceCategory']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multi-class problems

# Print evaluation metrics
print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate the average price for each location (if needed)
if 'Location' in df.columns and 'Price' in df.columns:
    average_prices = df.groupby('Location')['Price'].mean()

    # Decode the location labels
    average_prices.index = [label_encoders['Location'].inverse_transform([i])[0] for i in average_prices.index]

    # Plot the average prices by location
    plt.figure(figsize=(10, 6))
    average_prices.plot(kind='bar', color='skyblue')

    plt.title(f'Average House Prices by Location')
    plt.xlabel('Location')
    plt.ylabel('Average Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
