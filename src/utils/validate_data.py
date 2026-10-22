import great_expectations as ge
import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("🔍 Starting data validation with Great Expectations...")
    
    # Check schema before constructing a GX 1.x in-memory batch.
    required = ["customerID", "gender", "Partner", "Dependents", "PhoneService",
                "InternetService", "Contract", "tenure", "MonthlyCharges", "TotalCharges"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, [f"missing_column:{c}" for c in missing]

    # Numeric comparisons need numbers, not CSV strings. Preserve the input.
    data = df.copy()
    conversion_failures = []
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        original = data[column]
        blank = original.isna() | original.astype(str).str.strip().eq("")
        numeric = pd.to_numeric(original, errors="coerce")
        if (numeric.isna() & ~blank).any():
            conversion_failures.append(f"invalid_numeric:{column}")
        data[column] = numeric

    context = ge.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("telco")
    asset = source.add_dataframe_asset("customers")
    definition = asset.add_batch_definition_whole_dataframe("all_customers")
    batch = definition.get_batch(batch_parameters={"dataframe": data})
    suite = ge.ExpectationSuite(name="telco_data_quality")
    
    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")
    
    # Customer identifier must exist (required for business operations)  
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column='customerID'))
    suite.add_expectation(ge.expectations.ExpectColumnValuesToNotBeNull(column='customerID'))
    
    # Core demographic features
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column='gender'))
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column='Partner'))
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column='Dependents'))
    
    # Service features (critical for churn analysis)
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column='PhoneService'))
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column='InternetService'))
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column='Contract'))
    
    # Financial features (key churn predictors)
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column='tenure'))
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column='MonthlyCharges'))
    suite.add_expectation(ge.expectations.ExpectColumnToExist(column='TotalCharges'))
    
    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")
    
    # Gender must be one of expected values (data integrity)
    suite.add_expectation(ge.expectations.ExpectColumnValuesToBeInSet(column='gender', value_set=['Male', 'Female']))
    
    # Yes/No fields must have valid values
    suite.add_expectation(ge.expectations.ExpectColumnValuesToBeInSet(column='Partner', value_set=['Yes', 'No']))
    suite.add_expectation(ge.expectations.ExpectColumnValuesToBeInSet(column='Dependents', value_set=['Yes', 'No']))
    suite.add_expectation(ge.expectations.ExpectColumnValuesToBeInSet(column='PhoneService', value_set=['Yes', 'No']))
    
    # Contract types must be valid (business constraint)
    suite.add_expectation(ge.expectations.ExpectColumnValuesToBeInSet(column='Contract', value_set=['Month-to-month', 'One year', 'Two year']))
    
    # Internet service types (business constraint)
    suite.add_expectation(ge.expectations.ExpectColumnValuesToBeInSet(column='InternetService', value_set=['DSL', 'Fiber optic', 'No']))
    
    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")
    
    # Tenure must be non-negative (business logic - can't have negative tenure)
    suite.add_expectation(ge.expectations.ExpectColumnValuesToBeBetween(column='tenure', min_value=0))
    
    # Monthly charges must be positive (business logic - no free service)
    suite.add_expectation(ge.expectations.ExpectColumnValuesToBeBetween(column='MonthlyCharges', min_value=0))
    
    # Total charges should be non-negative (business logic)
    suite.add_expectation(ge.expectations.ExpectColumnValuesToBeBetween(column='TotalCharges', min_value=0))
    
    # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")
    
    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    suite.add_expectation(ge.expectations.ExpectColumnValuesToBeBetween(column='tenure', min_value=0, max_value=120))
    
    # Monthly charges should be within reasonable business range
    suite.add_expectation(ge.expectations.ExpectColumnValuesToBeBetween(column='MonthlyCharges', min_value=0, max_value=200))
    
    # No missing values in critical numeric features  
    suite.add_expectation(ge.expectations.ExpectColumnValuesToNotBeNull(column='tenure'))
    suite.add_expectation(ge.expectations.ExpectColumnValuesToNotBeNull(column='MonthlyCharges'))
    
    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")
    
    # Total charges should generally be >= Monthly charges (except for very new customers)
    # This is a business logic check to catch data entry errors
    suite.add_expectation(ge.expectations.ExpectColumnPairValuesAToBeGreaterThanB(column_A='TotalCharges', column_B='MonthlyCharges', or_equal=True, mostly=0.95, ignore_row_if='either_value_is_missing'))
    
    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    results = batch.validate(suite)
    
    # === PROCESS RESULTS ===
    # Extract failed expectations for detailed error reporting
    failed_expectations = list(conversion_failures)
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r.expectation_config.type
            failed_expectations.append(expectation_type)
    
    # Print validation summary
    total_checks = len(results["results"]) + 3  # Include numeric conversion checks.
    passed_checks = sum(1 for r in results["results"] if r["success"]) + 3 - len(conversion_failures)
    failed_checks = total_checks - passed_checks
    
    if results["success"] and not conversion_failures:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    
    return bool(results["success"]) and not conversion_failures, failed_expectations
