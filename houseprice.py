
# House Price Prediction - Simple Linear Regression

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Sample Data Generation (Replace this with your actual dataset)
# For demonstration, we use some dummy features: [Size (sq ft), Bedrooms, Age of House (years)]
X = np.array([
    [1500, 3, 5],
    [1800, 4, 10],
    [2400, 3, 2],
    [3000, 4, 15],
    [1200, 2, 8]
])

# Target variable: House Prices (in thousands of dollars)
y = np.array([300, 400, 500, 600, 250])

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions using the test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
print("--- Model Performance ---")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Coefficient of Determination (R^2 Score):", r2_score(y_test, y_pred))

# 6. Example prediction for a new house
new_house = np.array([[1600, 3, 4]]) # 1600 sq ft, 3 bedrooms, 4 years old
predicted_price = model.predict(new_house)
print(f"\nPredicted Price for a new house (1600 sq ft, 3 beds, 4 years old): ${predicted_price[0]*1000:,.2f}")
