import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

data = pd.read_csv("Data/stock_re.csv")

X = data[[ 'Day', 'Open', 'High', 'Low', 'Volume', 'Chg%']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

y_train_pred = model.predict(X_train_scaled).flatten()
y_test_pred = model.predict(X_test_scaled).flatten()

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)

mape_train = tf.keras.losses.MeanAbsolutePercentageError()(y_train, y_train_pred).numpy()
mape_test = tf.keras.losses.MeanAbsolutePercentageError()(y_test, y_test_pred).numpy()

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Training Set:")
print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("MAPE:", mape_train)
print("R2 Score:", r2_train)

print("\nTest Set:")
print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("MAPE:", mape_test)
print("R2 Score:", r2_test)

print("\nAnalysis of Results:")
print("- MSE and RMSE measure the average squared error between actual and predicted values, with RMSE providing a more interpretable scale.")
print("- MAPE measures the average percentage error, providing insight into the accuracy of predictions relative to the actual values.")
print("- R2 score represents the proportion of the variance in the dependent variable that is predictable from the independent variables.")
print("- Higher R2 scores closer to 1 indicate a better fit of the model to the data.")
