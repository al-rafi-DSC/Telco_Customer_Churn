import contextlib
import io
import unittest

import pandas as pd

from src.utils.validate_data import validate_telco_data


class ValidateTelcoDataTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "customerID": ["a", "b"], "gender": ["Female", "Male"],
            "Partner": ["No", "Yes"], "Dependents": ["No", "Yes"],
            "PhoneService": ["Yes", "Yes"], "InternetService": ["DSL", "No"],
            "Contract": ["Month-to-month", "Two year"], "tenure": [0, 12],
            "MonthlyCharges": [20.0, 30.0], "TotalCharges": [" ", "360.0"],
        })

    def validate(self, frame):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return validate_telco_data(frame)

    def test_numeric_strings_and_blank_charges_without_mutating_input(self):
        original = self.frame.copy(deep=True)
        self.assertEqual(self.validate(self.frame), (True, []))
        pd.testing.assert_frame_equal(self.frame, original)

    def test_invalid_numeric_text_is_not_silently_ignored(self):
        self.frame.loc[1, "TotalCharges"] = "invalid"
        valid, failures = self.validate(self.frame)
        self.assertFalse(valid)
        self.assertIn("invalid_numeric:TotalCharges", failures)

    def test_invalid_categories_and_ranges_fail(self):
        self.frame.loc[1, "Contract"] = "invalid"
        self.frame.loc[1, "tenure"] = -1
        valid, failures = self.validate(self.frame)
        self.assertFalse(valid)
        self.assertIn("expect_column_values_to_be_in_set", failures)
        self.assertIn("expect_column_values_to_be_between", failures)

    def test_missing_column_returns_actionable_failure(self):
        self.assertEqual(self.validate(self.frame.drop(columns="customerID")),
                         (False, ["missing_column:customerID"]))
