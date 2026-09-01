import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Small synthetic wastewater dataset
data = {
    "influent_bod": [250, 300, 280, 350, 220, 400, 320, 270, 380, 290],
    "influent_cod": [500, 600, 550, 700, 450, 800, 650, 520, 750, 580],
    "influent_tss": [250, 300, 280, 350, 200, 400, 330, 270, 380, 290],
    "ph": [7.1, 7.2, 7.0, 7.3, 7.1, 7.4, 7.2, 7.0, 7.3, 7.2],
    "hrt": [8, 8, 9, 7, 10, 6, 8, 9, 7, 8],
    "effluent_bod": [25, 30, 28, 35, 22, 42, 32, 27, 39, 29],
}

df = pd.DataFrame(data)

# Features
X = df[
    [
        "influent_bod",
        "influent_cod",
        "influent_tss",
        "ph",
        "hrt",
    ]
]

# Target
y = df["effluent_bod"]

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X, y)

# Example prediction
new_wastewater = pd.DataFrame(
    {
        "influent_bod": [310],
        "influent_cod": [620],
        "influent_tss": [300],
        "ph": [7.2],
        "hrt": [8],
    }
)

prediction = model.predict(new_wastewater)

print("Predicted effluent BOD₅:", prediction[0], "mg/L")