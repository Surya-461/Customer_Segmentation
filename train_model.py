import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("final_df.csv")

# Select correct features (MATCH EXACT COLUMN NAMES)
features = [
    "Total_Spending",
    "num_of_orders",
    "Average_Order_Value",
    "Recency",
    "Frequency"
]

X = df[features]

# Create Pipeline (Scaler + KMeans)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=3, random_state=42))
])

# Train model
pipeline.fit(X)

# Save pipeline
joblib.dump(pipeline, "customer_segmentation_pipeline.pkl")

print("Model trained and saved successfully!")
