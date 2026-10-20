from pathlib import Path

import joblib


# --------------------------------------------------
# 1. Find the trained model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sql_injection_model.pkl"


# --------------------------------------------------
# 2. Load model
# --------------------------------------------------

print("Loading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print()


# --------------------------------------------------
# 3. Prediction function
# --------------------------------------------------

def predict_sql_injection(user_input):
    prediction = model.predict([user_input])[0]
    probability = model.predict_proba([user_input])[0]

    if prediction == 1:
        result = "SQL Injection"
        confidence = probability[1]
    else:
        result = "Normal"
        confidence = probability[0]

    return result, confidence


# --------------------------------------------------
# 4. Interactive testing
# --------------------------------------------------

while True:

    user_input = input("Enter input (or type 'exit'): ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    result, confidence = predict_sql_injection(user_input)

    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2%}")
    print()