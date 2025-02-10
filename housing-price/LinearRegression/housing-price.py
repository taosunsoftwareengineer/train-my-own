# prerequisite: Set up a virtual environment:

# Create a virtual environment (uncomment below line)
# python -m venv myenv

# Activate the virtual environment (uncomment below line)
# source myenv/bin/activate

# Install requried libraries
# pip install numpy pandas scikit-learn jupyter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Sample data
data = {
    'list_price': [950000, 975000, 935000, 951513, 949000],
    'sale_price': [950000, 975000, 935000, 951513, 949000]
}

df = pd.DataFrame(data)

# Features and target
X = df[['list_price']]
y = df['sale_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Predict sale price for a new list price
new_list_price = np.array([[960000]])
predicted_sale_price = model.predict(new_list_price)
print(f'Predicted Sale Price: {predicted_sale_price[0]}')
