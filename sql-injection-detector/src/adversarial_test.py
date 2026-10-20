from pathlib import Path

import joblib


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "sql_injection_model_v4.pkl"


# ============================================================
# Load model
# ============================================================

print("Loading model...")

loaded = joblib.load(MODEL_PATH)

# V4 was saved as:
# {
#     "features": features,
#     "model": model
# }
#
# V1/V2/V3 were saved as the model directly.

if isinstance(loaded, dict):
    features = loaded["features"]
    model = loaded["model"]

    print("Detected V4 model format.")

else:
    features = None
    model = loaded

    print("Detected standard model format.")


# ============================================================
# Adversarial test cases
# ============================================================

test_cases = [

    # --------------------------------------------------------
    # Normal English containing SQL-like words
    # --------------------------------------------------------

    ("Please select a product", 0),

    ("Select your preferred option", 0),

    ("Union membership benefits", 0),

    ("Update my profile", 0),

    ("Delete my account", 0),

    ("Insert your address", 0),

    ("Drop-down menu", 0),

    ("Administrator account", 0),

    ("Select a country", 0),

    ("Update your contact details", 0),


    # --------------------------------------------------------
    # SQL Injection examples
    # --------------------------------------------------------

    ("admin' OR '1'='1", 1),

    ("' OR 1=1 --", 1),

    ("' AND 1=2 --", 1),

    ("admin'--", 1),

    ("' UNION SELECT NULL --", 1),

    ("' OR 'a'='a' --", 1),

    ("' AND SLEEP(5) --", 1),

    ("' OR EXISTS(SELECT 1) --", 1),
]


# ============================================================
# Helper function
# ============================================================

def predict_text(text):
    """
    Make a prediction for a single input.

    Handles both:
    1. V4 dictionary format
    2. Standard sklearn model format
    """

    if features is not None:

        # V4:
        # First convert raw text into TF-IDF features
        text_features = features.transform([text])

        prediction = model.predict(text_features)[0]

        probabilities = model.predict_proba(text_features)[0]

    else:

        # Standard model:
        prediction = model.predict([text])[0]

        probabilities = model.predict_proba([text])[0]

    confidence = probabilities[int(prediction)]

    return int(prediction), confidence


# ============================================================
# Run adversarial test
# ============================================================

print("\n")
print("=" * 70)
print("SQL Injection Adversarial Test")
print("=" * 70)


correct = 0
total = len(test_cases)


for text, expected in test_cases:

    prediction, confidence = predict_text(text)


    # Convert numeric labels to readable names

    if prediction == 1:
        predicted_label = "SQL Injection"
    else:
        predicted_label = "Normal"


    if expected == 1:
        expected_label = "SQL Injection"
    else:
        expected_label = "Normal"


    # Check result

    if prediction == expected:

        result = "CORRECT"
        correct += 1

    else:

        result = "WRONG"


    # Print result

    print()

    print(f"Input:      {text}")
    print(f"Expected:   {expected_label}")
    print(f"Predicted:  {predicted_label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Result:     {result}")


# ============================================================
# Final accuracy
# ============================================================

accuracy = correct / total


print("\n")
print("=" * 70)
print(f"Adversarial Test Accuracy: {accuracy:.2%}")
print("=" * 70)