import pytest
from classification.rule_classifier import classify_by_rules, classify_by_rules_with_confidence
from classification.ml_classifier import MLClassifier

def test_rule_classifier_keywords():
    assert classify_by_rules("ZOMATO FOOD") == "Food & Dining"
    assert classify_by_rules("UBER TRIP") == "Transportation"
    assert classify_by_rules("ZEPTO GROCERY") == "Shopping"
    assert classify_by_rules("SALARY CREDIT") == "Income"
    assert classify_by_rules("UNKNOWN STUFF") == "Uncategorized"

def test_rule_classifier_word_boundary_edge_cases():
    """Verify word boundaries prevent substring false matches."""
    # 'COLA CENTRE' must not match 'OLA' (Transportation)
    assert classify_by_rules("COLA CENTRE") == "Uncategorized"
    
    # 'CURRENT ACCOUNT' must not match 'RENT' (Rent)
    assert classify_by_rules("CURRENT ACCOUNT") == "Uncategorized"
    
    # 'LABOUR CHARGES' must not match 'LAB' (Healthcare)
    assert classify_by_rules("LABOUR CHARGES") == "Uncategorized"
    
    # 'COFFEE' must match Food & Dining and not Bank Charges (FEE)
    assert classify_by_rules("COFFEE SHOP") == "Food & Dining"

def test_rule_classifier_direction_awareness():
    """Verify direction awareness: interest paid on loan is Loans & EMI, not Income."""
    cat_loan, _ = classify_by_rules_with_confidence("INTEREST PAID ON HOME LOAN EMI", debit=15000.0, credit=0.0)
    assert cat_loan == "Loans & EMI"

    cat_inc, _ = classify_by_rules_with_confidence("INTEREST CREDIT", debit=0.0, credit=125.0)
    assert cat_inc == "Income"

    cat_fuel, _ = classify_by_rules_with_confidence("SHELL PETROL PUMP FUEL", debit=500.0, credit=0.0)
    assert cat_fuel == "Fuel"

def test_rule_classifier_confidence():
    cat, conf = classify_by_rules_with_confidence("ZOMATO FOOD")
    assert cat == "Food & Dining"
    assert conf >= 0.85

    cat_unknown, conf_unknown = classify_by_rules_with_confidence("RANDOM UNKNOWN STRING")
    assert cat_unknown == "Uncategorized"
    assert conf_unknown == 0.0

def test_ml_classifier_prediction_and_confidence():
    ml = MLClassifier()
    assert ml.is_trained is True

    # High-confidence classification
    cat, conf = ml.classify_with_confidence("SALARY CREDIT")
    assert cat == "Income"
    assert conf >= 0.15

    # Low-confidence fallback to Uncategorized
    cat_unk, conf_unk = ml.classify_with_confidence("RANDOM GIBBERISH XYZ123", threshold=0.30)
    assert cat_unk == "Uncategorized"

def test_ml_evaluation_metrics():
    ml = MLClassifier()
    metrics = ml.get_evaluation_metrics()
    
    assert "accuracy" in metrics
    assert "precision_weighted" in metrics
    assert "recall_weighted" in metrics
    assert "f1_weighted" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] >= 0.70
    assert metrics["f1_weighted"] >= 0.70
    assert len(metrics["classes"]) == 16
    assert metrics["training_samples"] >= 200

def test_confusion_matrix_figure():
    ml = MLClassifier()
    fig = ml.create_confusion_matrix_figure()
    assert fig is not None
    assert fig.data[0].type == "heatmap"
