"""FastAPI and Gradio share input validation and model inference."""
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
import gradio as gr

from src.serving.inference import BINARY_MAP, CATEGORIES, MANIFEST, THRESHOLD, predict_details

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


LOW_RISK_EXAMPLE = {
    **DEFAULTS, "Partner": "Yes", "Dependents": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)", "tenure": 60,
    "MonthlyCharges": 65.0, "TotalCharges": 3900.0,
}
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
LABELS = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Has partner",
    "Dependents": "Has dependents", "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
    "InternetService": "Internet service", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "Contract": "Contract", "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
    "tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges ($)",
    "TotalCharges": "Total charges ($)",
}


def profile_signals(data: dict) -> tuple[list[str], list[str]]:
    """Patterns associated with churn in the Telco dataset, shown alongside the score."""
    risk, safe = [], []
    if data["Contract"] == "Month-to-month":
        risk.append("Month-to-month contract")
    elif data["Contract"] == "Two year":
        safe.append("Two-year contract")
    if data["tenure"] < 12:
        risk.append("Customer for less than a year")
    elif data["tenure"] >= 48:
        safe.append("Long-standing customer (4+ years)")
    if data["InternetService"] == "Fiber optic":
        risk.append("Fiber optic internet")
    if data["PaymentMethod"] == "Electronic check":
        risk.append("Pays by electronic check")
    elif "automatic" in data["PaymentMethod"]:
        safe.append("Automatic payments")
    if data["InternetService"] != "No" and data["OnlineSecurity"] == "No" and data["TechSupport"] == "No":
        risk.append("No online security or tech support")
    if data["Partner"] == "Yes" and data["Dependents"] == "Yes":
        safe.append("Has partner and dependents")
    return risk, safe


def render_result(result: dict, data: dict) -> str:
    probability, threshold = result["churn_probability"], result["threshold"]
    tone = "risk" if result["prediction"] == "Likely to churn" else "safe"
    risk, safe = profile_signals(data)
    chips = "".join(f'<li class="chip chip-risk">{s}</li>' for s in risk)
    chips += "".join(f'<li class="chip chip-safe">{s}</li>' for s in safe)
    signals = (
        f'<div class="signals"><p class="eyebrow">Profile signals</p><ul>{chips}</ul>'
        '<p class="footnote">Patterns linked to churn in the training data, not a per-customer model explanation.</p></div>'
        if chips else ""
    )
    return f"""
<div class="result result-{tone}">
  <p class="eyebrow">Churn probability</p>
  <div class="score">{probability:.1%}</div>
  <span class="badge badge-{tone}">{result['prediction']}</span>
  <div class="gauge" role="img" aria-label="Churn probability {probability:.1%}, decision threshold {threshold:.0%}">
    <div class="gauge-fill" style="width:{probability * 100:.1f}%"></div>
    <div class="gauge-mark" style="left:{threshold * 100:.1f}%"><span>threshold {threshold:.0%}</span></div>
  </div>
  {signals}
</div>"""


EMPTY_RESULT = """
<div class="result result-empty">
  <p class="eyebrow">Churn probability</p>
  <div class="score">--</div>
  <p class="hint">Fill in the customer profile and press <b>Predict churn</b>, or load one of the examples.</p>
</div>"""


def gradio_interface(*values):
    try:
        data = CustomerData(**dict(zip(FIELD_ORDER, values)))
    except ValidationError as exc:
        raise gr.Error("Please check the customer fields and numeric ranges.") from exc
    return render_result(predict_details(data.model_dump()), data.model_dump())


def sync_internet(service, *addons):
    """Add-on services only exist for customers who have internet."""
    if service == "No":
        return [gr.update(value="No internet service", interactive=False) for _ in addons]
    return [gr.update(value="No" if v == "No internet service" else v, interactive=True) for v in addons]


def sync_phone(service, lines):
    if service == "No":
        return gr.update(value="No phone service", interactive=False)
    return gr.update(value="No" if lines == "No phone service" else lines, interactive=True)


def estimate_total(tenure, monthly):
    return round(float(tenure or 0) * float(monthly or 0), 2)


metrics = MANIFEST["metrics"]
HEADER = f"""
<header class="hero">
  <p class="eyebrow">XGBoost &middot; Telco customer churn</p>
  <h1>Will this customer churn?</h1>
  <p class="lede">Describe a subscriber and the model estimates how likely they are to cancel. Scores at or above the {THRESHOLD:.0%} threshold are flagged as likely to churn.</p>
  <ul class="stats">
    <li><b>{metrics['roc_auc']:.2f}</b><span>ROC AUC</span></li>
    <li><b>{metrics['recall']:.0%}</b><span>Recall</span></li>
    <li><b>{metrics['precision']:.0%}</b><span>Precision</span></li>
    <li><b>{THRESHOLD:.2f}</b><span>Threshold</span></li>
  </ul>
</header>"""

CSS = """
:root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --card:#ffffff; --soft:#f1f5f9;
  --risk:#e11d48; --risk-bg:#fff1f2; --safe:#059669; --safe-bg:#ecfdf5; }
.dark { --ink:#f1f5f9; --muted:#94a3b8; --line:#334155; --card:#0f172a; --soft:#1e293b;
  --risk:#fb7185; --risk-bg:#4c0519; --safe:#34d399; --safe-bg:#022c22; }
.gradio-container { max-width:1180px !important; margin:0 auto !important; }
footer { display:none !important; }
.hero { padding:36px 32px; border-radius:20px; color:#fff; margin-bottom:8px;
  background:radial-gradient(circle at 85% 0%, #a78bfa 0%, transparent 45%), linear-gradient(135deg,#312e81,#4f46e5 60%,#7c3aed); }
.hero h1 { font-size:clamp(28px,4vw,40px); font-weight:800; letter-spacing:-.02em; margin:6px 0 10px; color:#fff; }
.hero .lede { max-width:620px; opacity:.9; font-size:16px; line-height:1.55; margin:0; color:#fff; }
.hero .eyebrow { color:#c7d2fe; }
.stats { display:flex; flex-wrap:wrap; gap:12px; list-style:none; padding:0; margin:24px 0 0; }
.stats li { background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.2); border-radius:14px; padding:10px 16px; min-width:96px; }
.stats b { display:block; font-size:22px; font-variant-numeric:tabular-nums; color:#fff; }
.stats span { font-size:12px; opacity:.8; text-transform:uppercase; letter-spacing:.06em; color:#fff; }
.eyebrow { font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:0; }
.section-title h3 { font-size:15px; font-weight:700; margin:10px 12px 0; color:var(--ink); }
.section-title p { font-size:13px; color:var(--muted); margin:2px 12px 6px; }
.sticky { position:sticky; top:16px; align-self:flex-start; }
.result { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:24px; }
.result-risk { border-top:4px solid var(--risk); }
.result-safe { border-top:4px solid var(--safe); }
.result-empty { border-top:4px solid var(--line); }
.score { font-size:56px; font-weight:800; letter-spacing:-.03em; color:var(--ink); font-variant-numeric:tabular-nums; line-height:1.1; margin:6px 0 10px; }
.result-risk .score { color:var(--risk); } .result-safe .score { color:var(--safe); }
.badge { display:inline-block; padding:6px 12px; border-radius:999px; font-weight:600; font-size:14px; }
.badge-risk { background:var(--risk-bg); color:var(--risk); } .badge-safe { background:var(--safe-bg); color:var(--safe); }
.gauge { position:relative; height:12px; border-radius:999px; background:var(--soft); border:1px solid var(--line); margin:28px 0 34px; }
.gauge-fill { height:100%; border-radius:999px; transition:width .5s ease; }
.result-risk .gauge-fill { background:linear-gradient(90deg,#fb923c,var(--risk)); }
.result-safe .gauge-fill { background:linear-gradient(90deg,#6ee7b7,var(--safe)); }
.gauge-mark { position:absolute; top:-6px; width:2px; height:24px; background:var(--ink); }
.gauge-mark span { position:absolute; top:26px; left:50%; transform:translateX(-50%); white-space:nowrap; font-size:11px; color:var(--muted); }
.signals ul { list-style:none; padding:0; margin:10px 0 8px; display:flex; flex-wrap:wrap; gap:8px; }
.chip { font-size:13px; padding:5px 10px; border-radius:8px; margin:0; }
.chip-risk { background:var(--risk-bg); color:var(--risk); } .chip-safe { background:var(--safe-bg); color:var(--safe); }
.chip-risk::before { content:"\\25B2  "; font-size:10px; } .chip-safe::before { content:"\\25BC  "; font-size:10px; }
.footnote, .hint { font-size:12px; color:var(--muted); margin:6px 0 0; line-height:1.5; }
.hint { font-size:14px; }
@media (max-width: 768px) { .hero { padding:24px 18px; } .score { font-size:44px; } .sticky { position:static; } }
"""

THEME = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="violet", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    block_title_background_fill="transparent", block_title_background_fill_dark="transparent",
    block_title_text_color="*neutral_600", block_title_text_color_dark="*neutral_300",
    block_title_padding="0", block_title_text_weight="600",
)


def _field(name, kind="radio"):
    if name == "SeniorCitizen":
        return gr.Radio([("No", 0), ("Yes", 1)], value=DEFAULTS[name], label=LABELS[name])
    options = list(BINARY_MAP[name]) if name in BINARY_MAP else CATEGORIES[name]
    component = gr.Dropdown if kind == "dropdown" else gr.Radio
    return component(options, value=DEFAULTS[name], label=LABELS[name])


def _section(title, subtitle):
    gr.HTML(f'<div class="section-title"><h3>{title}</h3><p>{subtitle}</p></div>')


with gr.Blocks(theme=THEME, css=CSS, title="Telco Churn Predictor", analytics_enabled=False) as demo:
    gr.HTML(HEADER)
    fields = {}
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            with gr.Group():
                _section("Customer profile", "Who the subscriber is")
                with gr.Row():
                    for name in ("gender", "SeniorCitizen", "Partner", "Dependents"):
                        fields[name] = _field(name)
            with gr.Group():
                _section("Services", "What they subscribe to")
                with gr.Row():
                    fields["PhoneService"] = _field("PhoneService")
                    fields["MultipleLines"] = _field("MultipleLines")
                fields["InternetService"] = _field("InternetService")
                with gr.Row():
                    for name in INTERNET_ADDONS[:3]:
                        fields[name] = _field(name, "dropdown")
                with gr.Row():
                    for name in INTERNET_ADDONS[3:]:
                        fields[name] = _field(name, "dropdown")
            with gr.Group():
                _section("Account &amp; billing", "Contract, payment and charges")
                fields["Contract"] = _field("Contract")
                with gr.Row():
                    fields["PaymentMethod"] = _field("PaymentMethod", "dropdown")
                    fields["PaperlessBilling"] = _field("PaperlessBilling")
                fields["tenure"] = gr.Slider(0, 72, value=DEFAULTS["tenure"], step=1, label=LABELS["tenure"])
                fields["MonthlyCharges"] = gr.Slider(
                    0, 150, value=DEFAULTS["MonthlyCharges"], step=0.05, label=LABELS["MonthlyCharges"])
                fields["TotalCharges"] = gr.Number(
                    value=DEFAULTS["TotalCharges"], minimum=0, label=LABELS["TotalCharges"],
                    info="Filled in from tenure x monthly charges; you can edit it.",
                )
        with gr.Column(scale=2, elem_classes="sticky"):
            output = gr.HTML(EMPTY_RESULT)
            predict_btn = gr.Button("Predict churn", variant="primary", size="lg")
            gr.Markdown("**Try an example**")
            with gr.Row():
                high_btn = gr.Button("High-risk customer", size="sm")
                low_btn = gr.Button("Loyal customer", size="sm")

    inputs = [fields[name] for name in FIELD_ORDER]
    addons = [fields[name] for name in INTERNET_ADDONS]
    fields["InternetService"].change(sync_internet, [fields["InternetService"], *addons], addons)
    fields["PhoneService"].change(sync_phone, [fields["PhoneService"], fields["MultipleLines"]], fields["MultipleLines"])
    for slider in (fields["tenure"], fields["MonthlyCharges"]):
        slider.release(estimate_total, [fields["tenure"], fields["MonthlyCharges"]], fields["TotalCharges"])
    predict_btn.click(gradio_interface, inputs, output)
    for button, example in ((high_btn, DEFAULTS), (low_btn, LOW_RISK_EXAMPLE)):
        button.click(lambda e=example: [e[name] for name in FIELD_ORDER], None, inputs).then(
            gradio_interface, inputs, output)

app = gr.mount_gradio_app(app, demo, path="/ui")
