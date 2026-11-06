import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Load trained model and DictVectorizer
MODEL_PATH = Path(__file__).resolve().parent / "models" / "credit_risk_model.bin"

with open(MODEL_PATH, "rb") as f_in:
    dv, model = pickle.load(f_in)


app = FastAPI(
    title="Credit Card Default Risk Prediction API",
    description="Predict credit card payment default using Random Forest",
    version="1.0.0"
)


# Customer input schema
class Customer(BaseModel):
    limit_bal: float = Field(gt=0)
    sex: int
    education: int
    marriage: int
    age: int = Field(gt=0)

    pay_0: int
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int

    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float

    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float


@app.get("/")
def home():
    return {
        "message": "Credit Card Default Risk Prediction API",
        "status": "running"
    }


@app.post("/predict")
def predict(customer: Customer):

    # Convert Pydantic input to dictionary
    customer_dict = customer.model_dump()

    # Same categorical preprocessing used in training
    categorical = ["sex", "education", "marriage"]

    for col in categorical:
        customer_dict[col] = str(customer_dict[col])

    # Transform features
    X = dv.transform([customer_dict])

    # Predict probability
    default_probability = model.predict_proba(X)[0, 1]

    # Classification threshold
    default = default_probability >= 0.5

    return {
        "default_probability": float(default_probability),
        "default": bool(default)
    }
