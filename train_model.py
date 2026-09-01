import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

data = pd.read_csv("data/raw/wastewater.csv")


# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

features = [
    "influent_bod5",
    "influent_cod",
    "influent_tss",
    "flow_m3_day",
    "dissolved_oxygen",
    "temperature",
    "hrt_hours",
]

target = "effluent_bod5"


X = data[features]
y = data[target]


# --------------------------------------------------
# 3. Split data into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# --------------------------------------------------
# 4. Create and train model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 5. Evaluate model
# --------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


print("Model Evaluation")
print("----------------")
print(f"MAE:  {mae:.2f} mg/L")
print(f"RMSE: {rmse:.2f} mg/L")
print(f"R²:   {r2:.2f}")


# --------------------------------------------------
# 6. Save trained model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

model_path = "models/wastewater_bod5_model.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")


# --------------------------------------------------
# 7. Make an example prediction
# --------------------------------------------------

sample = pd.DataFrame([{
    "influent_bod5": 300,
    "influent_cod": 570,
    "influent_tss": 250,
    "flow_m3_day": 1050,
    "dissolved_oxygen": 2.1,
    "temperature": 27,
    "hrt_hours": 8,
}])

prediction = model.predict(sample)[0]

print(f"\nPredicted effluent BOD₅: {prediction:.2f} mg/L")