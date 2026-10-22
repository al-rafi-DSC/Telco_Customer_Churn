"""FastAPI and Gradio share input validation and model inference."""
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
import gradio as gr

from src.serving.inference import BINARY_MAP, CATEGORIES, predict_details

logger = logging.getLogger(__name__)
app = FastAPI(title="Telco Customer Churn Prediction API", version="1.1.0")

YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]


class CustomerData(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    tenure: int = Field(ge=0, le=120)
    MonthlyCharges: float = Field(ge=0, le=200)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        return predict_details(data.model_dump())
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


FIELD_ORDER = list(CustomerData.model_fields)
DEFAULTS = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "tenure": 1,
    "MonthlyCharges": 85.0, "TotalCharges": 85.0,
}


def gradio_interface(*values):
    try:
        data = CustomerData(**dict(zip(FIELD_ORDER, values)))
    except ValidationError as exc:
        raise gr.Error("Please check the customer fields and numeric ranges.") from exc
    result = predict_details(data.model_dump())
    return f"{result['prediction']} ? estimated churn probability: {result['churn_probability']:.1%}"


inputs = []
for field in FIELD_ORDER:
    options = list(BINARY_MAP[field]) if field in BINARY_MAP else CATEGORIES.get(field)
    if field == "SeniorCitizen":
        options = [0, 1]
    if options is not None:
        inputs.append(gr.Dropdown(options, value=DEFAULTS[field], label=field))
    else:
        inputs.append(gr.Number(value=DEFAULTS[field], label=field, minimum=0))

demo = gr.Interface(
    fn=gradio_interface, inputs=inputs, outputs=gr.Textbox(label="Prediction"),
    title="Telco Customer Churn Predictor",
    description="Enter customer details to estimate churn risk. SeniorCitizen: 0 = No, 1 = Yes.",
    examples=[[DEFAULTS[field] for field in FIELD_ORDER]],
    cache_examples=False, analytics_enabled=False, theme=gr.themes.Soft(),
)
app = gr.mount_gradio_app(app, demo, path="/ui")
