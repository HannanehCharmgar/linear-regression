#1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# 1. Create dataset (10 houses: area and price)
# ============================================
data = {
    'area': [50, 70, 80, 90, 100, 120, 130, 150, 170, 200],
    'price': [320, 400, 480, 520, 590, 650, 750, 820, 900, 1100]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)
print("\n" + "="*50 + "\n")

# ============================================
# 2. Prepare data for regression
# ============================================
X = df[['area']]
y = df['price']

# ============================================
# 3. Create and train linear regression model
# ============================================
model = LinearRegression()
model.fit(X, y)

# ============================================
# 4. Display model information
# ============================================
print("Linear Regression Results:")
print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Equation: price = {model.intercept_:.2f} + {model.coef_[0]:.2f} * area")

# ============================================
# 5. Make predictions
# ============================================
y_pred = model.predict(X)

# Calculate evaluation metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"\nEvaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# ============================================
# 6. Predict for new areas
# ============================================
new_areas = [[60], [110], [180]]
new_predictions = model.predict(new_areas)

print(f"\nPredictions for new areas:")
for area, price in zip(new_areas, new_predictions):
    print(f"Area {area[0]} m² → Predicted price: {price:.2f}")

# ============================================
# 7. Plot the results
# ============================================
plt.figure(figsize=(12, 6))

# Plot 1: Scatter points and regression line
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', s=80, label='Actual Data', alpha=0.7)
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Area (m²)', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.title('Linear Regression: House Price Prediction', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted comparison
plt.subplot(1, 2, 2)
plt.scatter(y, y_pred, color='green', s=80, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal Line')
plt.xlabel('Actual Price', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.title('Actual vs Predicted Values', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 8. Display comparison table
# ============================================
df_comparison = df.copy()
df_comparison['predicted_price'] = y_pred.round(2)
df_comparison['difference'] = (df_comparison['price'] - df_comparison['predicted_price']).round(2)

print("\n" + "="*50)
print("Comparison Table:")
print("="*50)
print(df_comparison)