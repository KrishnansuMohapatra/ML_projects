"""Project-wide paths, constants, and the canonical feature list."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
BATCH_DIR = PROJECT_ROOT / "batch"

DATASET_PATH = DATA_DIR / "phishing_data.csv"
MODEL_PATH = MODEL_DIR / "phishing_model.pkl"

TARGET_COLUMN = "result"

PHISHING_LABEL = 1
LEGITIMATE_LABEL = -1

LABEL_NAMES = {
    PHISHING_LABEL: "Phishing",
    LEGITIMATE_LABEL: "Legitimate",
}

FEATURES = [
    "having_ip_address",
    "url_length",
    "shortining_service",
    "having_at_symbol",
    "double_slash_redirecting",
    "prefix_suffix",
    "having_sub_domain",
    "sslfinal_state",
    "domain_registration_length",
    "favicon",
    "port",
    "https_token",
    "request_url",
    "url_of_anchor",
    "links_in_tags",
    "sfh",
    "submitting_to_email",
    "abnormal_url",
    "redirect",
    "on_mouseover",
    "rightclick",
    "popupwindow",
    "iframe",
    "age_of_domain",
    "dnsrecord",
    "web_traffic",
    "page_rank",
    "google_index",
    "links_pointing_to_page",
    "statistical_report",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
