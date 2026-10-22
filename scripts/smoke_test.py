"""Check a running deployment using only Python's standard library."""
import argparse
import json
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    deadline = time.monotonic() + 120
    while True:
        try:
            with urllib.request.urlopen(base + "/", timeout=5) as response:
                assert json.load(response)["status"] == "ok"
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)
    for path in ("/ui/", "/ui/config", "/docs"):
        with urllib.request.urlopen(base + path, timeout=20) as response:
            assert response.status == 200
    payload = {
        "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Two year", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "tenure": 12, "MonthlyCharges": 85.0, "TotalCharges": 1020.0,
    }
    request = urllib.request.Request(base + "/predict", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.load(response)
    probability = result["churn_probability"]
    assert 0 <= probability <= 1
    expected = "Likely to churn" if probability >= result["threshold"] else "Not likely to churn"
    assert result["prediction"] == expected
    print("PASS: health, UI, docs and real model prediction", result)


if __name__ == "__main__":
    main()
