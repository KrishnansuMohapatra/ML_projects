# Instagram Fake Account Detection

## Project Summary

An end-to-end machine-learning classification project that estimates whether an Instagram account is **Fake** or **Non-Fake** from six profile-level features.

**Pipeline:** Data → EDA → Feature Selection → Train/Test Split → Scaling → Logistic Regression → Probability → Threshold Analysis → ROC-AUC → Precision-Recall → F1-based threshold → Model persistence → Streamlit.

## Dataset

Working dataset: **900 account records**.

Features used:
- `userFollowerCount`
- `userFollowingCount`
- `userBiographyLength`
- `userMediaCount`
- `usernameDigitCount`
- `usernameLength`

Target:
- `isFake = 1`: Fake
- `isFake = 0`: Non-Fake

Important: this working file is a 900-row subset, not the complete 1,194-account InstaFake dataset. Results therefore describe this dataset/test split and are not claims of real-world accuracy.

## Training

The data was split 80/20 using `random_state=42` and `stratify=y`:
- Training: 720
- Testing: 180

`StandardScaler` was fitted only on training data, then applied to test data.

The classifier is Logistic Regression with `max_iter=1000`.

## Results at Threshold 0.50

Accuracy: **90.00%**

Confusion matrix:
```text
[[135, 5],
 [ 13, 27]]
```

Fake class:
- Precision: 84%
- Recall: 68%
- F1: 75%

The default threshold misses 13 of 40 fake accounts.

## Threshold Analysis

| Threshold | Fake Precision | Fake Recall |
|---:|---:|---:|
| 0.50 | 84.38% | 67.50% |
| 0.40 | 79.41% | 67.50% |
| 0.30 | 73.17% | 75.00% |
| 0.20 | 67.92% | 90.00% |

Lowering the threshold catches more fake accounts but also flags more non-fake accounts.

## ROC-AUC

**ROC-AUC = 0.9507**

This indicates strong class-ranking/discrimination ability on the test split.

## Precision-Recall

**Average Precision = 0.8550**

This evaluates the precision-recall behavior across thresholds.

## Selected Threshold

F1-score optimization selected:

**Threshold = 0.1935**

At this threshold:
- Precision = **68.52%**
- Recall = **92.50%**
- F1 = **78.72%**
- TN = 123
- FP = 17
- FN = 3
- TP = 37

Thus, 37/40 fake accounts were detected in this test split.

This is a high-recall operating point. It is not universally optimal; the appropriate threshold depends on the cost of false positives versus false negatives.

## Streamlit

The app loads:
- `models/instagram_fake_model.pkl`
- `models/instagram_fake_scaler.pkl`
- `models/model_metadata.json`

Run from the project root:

```powershell
uv run streamlit run app.py
```

The application collects the six features, scales them with the saved scaler, obtains the fake probability, and applies the saved threshold.

## Project Structure

```text
Instagram-fake-account-detection/
├── app.py
├── train_real.py
├── main.py
├── instagram_fake_account_enhanced_dataset.csv
├── our_instagram_fake_account_dataset_synthetic.csv
├── models/
│   ├── instagram_fake_model.pkl
│   ├── instagram_fake_scaler.pkl
│   ├── model_metadata.json
│   ├── roc_curve.png
│   └── precision_recall_curve.png
└── README.md
```

## Limitations

This is an educational ML system, not an Instagram enforcement system. It uses only six profile-level features and does not analyze images, captions, comments, network behavior, engagement authenticity, account history, or coordinated activity.

The dataset is relatively small and the working data is a 900-row subset. Threshold selection was performed on the available test split; a production workflow should use a separate validation set for threshold selection and reserve an untouched test set for final evaluation.

A prediction means **model-estimated likelihood**, not proof that an account or person is fake.

## Technologies

Python, Pandas, Scikit-learn, Matplotlib, Joblib, Streamlit, uv.

## Learning Outcomes

This project demonstrates that practical ML involves much more than model training. It covers:
- EDA
- Feature selection
- Data splitting
- Scaling
- Logistic Regression
- Confusion matrices
- Precision, recall and F1
- Probability prediction
- Threshold tuning
- ROC-AUC
- Precision-Recall analysis
- Model persistence
- Basic deployment with Streamlit
