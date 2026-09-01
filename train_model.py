import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# Load wastewater dataset
data = pd.read_csv("data/raw/wastewater.csv")


# Input features
features = [
    "influent_bod5",
    "influent_cod",
    "influent_tss",
    "flow_m3_day",
    "dissolved_oxygen",
    "temperature",
    "hrt_hours",
]


# Target variable
target = "effluent_bod5"


X = data[features]
y = data[target]


# Create and train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# Example treatment conditions
sample = pd.DataFrame([{
    "influent_bod5": 300,
    "influent_cod": 570,
    "influent_tss": 250,
    "flow_m3_day": 1050,
    "dissolved_oxygen": 2.1,
    "temperature": 27,
    "hrt_hours": 8,
}])


# Make prediction
prediction = model.predict(sample)[0]


print(f"Predicted effluent BOD₅: {prediction:.2f} mg/L")