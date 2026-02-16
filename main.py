from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained pipeline (Scaler + KMeans)
model = joblib.load("customer_segmentation_pipeline.pkl")

# Request body with validation (optional but recommended)
class CustomerData(BaseModel):
    total_spending: float = Field(..., ge=0)
    num_of_orders: float = Field(..., ge=0)
    average_order_value: float = Field(..., ge=0)
    recency: float = Field(..., ge=0)
    frequency: float = Field(..., ge=0)

@app.get("/")
def home():
    return {"message": "Customer Segmentation API Running"}

@app.post("/predict")
def predict_segment(data: CustomerData):

    input_df = pd.DataFrame([{
        "Total_Spending": data.total_spending,
        "num_of_orders": data.num_of_orders,
        "Average_Order_Value": data.average_order_value,
        "Recency": data.recency,
        "Frequency": data.frequency
    }])

    predicted_cluster = model.predict(input_df)[0]

    scaler = model.named_steps["scaler"]
    kmeans = model.named_steps["kmeans"]

    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # Recency column index = 3
    recency_values = centers[:, 3]

    # Lower recency = better customer
    sorted_clusters = recency_values.argsort()

    cluster_label_map = {
        sorted_clusters[0]: "High Value Customer",    # lowest recency
        sorted_clusters[1]: "Medium Value Customer",
        sorted_clusters[2]: "Low Value Customer"      # highest recency
    }

    return {
        "segment_number": int(predicted_cluster),
        "customer_type": cluster_label_map[predicted_cluster]
    }
