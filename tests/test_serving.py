import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.app.main import app, DEFAULTS, FIELD_ORDER, LOW_RISK_EXAMPLE, gradio_interface, sync_internet, sync_phone
from src.features.build_features import build_features
from src.serving import inference


class ServingTests(unittest.TestCase):
    def test_single_and_batch_encoding_match_training(self):
        # Cover every category, including all four payment options.
        rows = []
        for i in range(4):
            row = dict(DEFAULTS)
            row.update({c: list(v)[i % len(v)] for c, v in inference.BINARY_MAP.items()})
            row.update({c: v[i % len(v)] for c, v in inference.CATEGORIES.items()})
            row["SeniorCitizen"] = i % 2
            rows.append(row)
        frame = pd.DataFrame(rows)
        with contextlib.redirect_stdout(io.StringIO()):
            expected = build_features(frame).reindex(columns=inference.FEATURE_COLS)
        actual = inference._serve_transform(frame)
        np.testing.assert_array_equal(actual.values, expected.values)
        for i, row in enumerate(rows):
            np.testing.assert_array_equal(inference._serve_transform(pd.DataFrame([row])).values[0], expected.values[i])

    def test_threshold_uses_probability_not_default_class_prediction(self):
        for probability, label in [(0.34, "Not likely to churn"), (0.35, "Likely to churn"), (0.4, "Likely to churn")]:
            with patch.object(inference.model, "predict_proba", return_value=np.array([[1-probability, probability]])):
                self.assertEqual(inference.predict(dict(DEFAULTS)), label)

    def test_health_ui_and_prediction(self):
        with TestClient(app) as client:
            self.assertEqual(client.get("/").json(), {"status": "ok"})
            self.assertEqual(client.get("/ui/").status_code, 200)
            self.assertEqual(client.get("/ui/config").status_code, 200)
            response = client.post("/predict", json=DEFAULTS)
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertTrue(0 <= result["churn_probability"] <= 1)
            self.assertEqual(result["threshold"], 0.35)
            self.assertEqual(result["prediction"], inference.predict(dict(DEFAULTS)))

    def test_api_rejects_invalid_or_missing_inputs(self):
        with TestClient(app) as client:
            for changes in ({"Contract": "unknown"}, {"tenure": -1}, {"tenure": 1.5}, {"SeniorCitizen": 3}, {"MonthlyCharges": "nan"}):
                self.assertEqual(client.post("/predict", json={**DEFAULTS, **changes}).status_code, 422)
            payload = dict(DEFAULTS)
            del payload["SeniorCitizen"]
            self.assertEqual(client.post("/predict", json=payload).status_code, 422)

    def test_ui_uses_same_inputs_and_prediction(self):
        text = gradio_interface(*[DEFAULTS[field] for field in FIELD_ORDER])
        self.assertIn(inference.predict(dict(DEFAULTS)), text)

    def test_ui_examples_are_valid_and_render_both_outcomes(self):
        for example, tone in ((DEFAULTS, "result-risk"), (LOW_RISK_EXAMPLE, "result-safe")):
            html = gradio_interface(*[example[field] for field in FIELD_ORDER])
            self.assertIn(tone, html)

    def test_ui_locks_services_that_cannot_exist(self):
        locked = sync_internet("No", "Yes", "No")
        self.assertTrue(all(u["value"] == "No internet service" and not u["interactive"] for u in locked))
        restored = sync_internet("DSL", "No internet service", "Yes")
        self.assertEqual([u["value"] for u in restored], ["No", "Yes"])
        self.assertEqual(sync_phone("No", "Yes")["value"], "No phone service")
        self.assertEqual(sync_phone("Yes", "No phone service")["value"], "No")

    def test_prediction_failure_returns_server_error(self):
        with TestClient(app) as client, patch("src.app.main.predict_details", side_effect=RuntimeError("private details")):
            response = client.post("/predict", json=DEFAULTS)
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json(), {"detail": "Prediction failed"})
