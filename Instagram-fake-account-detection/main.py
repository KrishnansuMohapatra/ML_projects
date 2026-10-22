import pandas as pd
import  numpy as np
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("our_instagram_fake_account_dataset_synthetic.csv")
print(df.head())
print(df.info)
print(df.columns)
# 2. Select features
features = [
    "followers",
    "following",
    "media_count",
    "bio_length",
    "username_length",
    "username_digit_count",
    "has_profile_picture",
    "is_private",
    "follower_following_ratio"
]

X = df[features]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Create Logistic Regression model
model = LogisticRegression(
    random_state=42,
    max_iter=1000
)


# 6. Train
model.fit(X_train_scaled, y_train)


# 7. Predict
y_pred = model.predict(X_test_scaled)


# 8. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy:.4f}")


# 9. Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 10. Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Non-Fake", "Fake"]
))
coefficients = pd.DataFrame({
    "feature": features,
    "coefficient": model.coef_[0]
})

coefficients = coefficients.sort_values(
    "coefficient",
    ascending=False
)

print(coefficients)
probabilities = model.predict_proba(X_test_scaled)

results = pd.DataFrame({
    "actual": y_test.values,
    "fake_probability": probabilities[:, 1],
    "prediction": y_pred
})

print("\nSample predictions:")
print(results.head(10))
