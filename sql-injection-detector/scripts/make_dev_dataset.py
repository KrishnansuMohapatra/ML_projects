"""Generate a small synthetic SQLi dataset for local development and tests.

The real project uses the SQLiV2 dataset from Kaggle (``data/sqliv2.csv``,
UTF-16 encoded). That file is git-ignored, so this script produces a
clearly-labelled synthetic stand-in with the same column names
(``Sentence``/``Label``) so the API can be exercised without the download.

Usage::

    uv run python scripts/make_dev_dataset.py --rows 4000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BASE_DIR / "data" / "sqliv2_dev.csv"

INJECTION_TEMPLATES = [
    "' OR '{n}'='{n}",
    "' OR {n}={n} --",
    "' AND {n}={n} --",
    "admin' --",
    "admin' #",
    "admin'/*",
    "' UNION SELECT {cols} --",
    "' UNION ALL SELECT {cols} FROM users --",
    "'; DROP TABLE {table}; --",
    "DROP TABLE {table}",
    "TRUNCATE TABLE {table}",
    "'; INSERT INTO users VALUES ('admin','pass') --",
    "'; UPDATE users SET role='admin' --",
    "'; DELETE FROM {table} WHERE 1=1 --",
    "' AND SLEEP({n}) --",
    "' OR pg_sleep({n}) --",
    "' AND BENCHMARK(1000000,MD5('a')) --",
    "' OR EXISTS(SELECT 1) --",
    "' UNION SELECT NULL,NULL,NULL --",
    "' OR 1=1 LIMIT 1 --",
    "1' AND extractvalue(1,concat(0x7e,version())) --",
    "' OR 'a'='a",
    "%27%20OR%201%3D1--",
    "') OR ('1'='1",
    "' OR 1=1 ORDER BY 1 --",
    "' UNION SELECT username, password FROM users --",
    "' OR username IS NOT NULL --",
    "'; WAITFOR DELAY '0:0:5' --",
    "' OR 1 LIKE 1 --",
    "admin' OR '1'='1' /*",
    "' OR SLEEP(5) AND '1'='1",
]

NORMAL_TEMPLATES = [
    "Please select a product",
    "Select a country",
    "Select your preferred language",
    "Update my profile",
    "Delete my account",
    "Contact the administrator",
    "Insert your phone number",
    "Create a new user account",
    "Union membership benefits",
    "Grant permission to the user",
    "Remove the old account",
    "Drop-down menu options",
    "How do I update my password?",
    "I want to delete my saved addresses",
    "Can you show my order history?",
    "Where can I insert my coupon code?",
    "SELECT * FROM users WHERE id = 1",
    "SELECT name FROM products",
    "SELECT COUNT(*) FROM orders",
    "SELECT id, name FROM customers WHERE active = 1",
    "The SQL SELECT statement retrieves records",
    "UPDATE users SET last_login = NOW() WHERE id = 42",
    "Please check the logs table for errors",
    "Search for books about database administration",
    "Show me users created this week",
    "What is the union of two sets?",
    "Ordinary sentence about selecting a plan",
    "Our administrator reviews every request",
    "Insert the card and enter your PIN",
    "Table booking for two people",
]

COLUMNS = ["username, password", "NULL", "1,2,3", "id, name", "'a','b'"]
TABLES = ["users", "customers", "orders", "accounts", "payments"]

# Extra fragments keep the synthetic corpus from collapsing to a handful of
# unique strings (dedupe in the trainer would otherwise shrink it badly).
PREFIXES = ["search=", "id=", "user=", "q=", "login=", "filter=", "", "", ""]
SUFFIXES = [
    "",
    " AND active=1",
    " ORDER BY {n}",
    " LIMIT {n}",
    " /* note */",
    " -- trailing",
    " # end",
]
FILLERS = [
    "on the {thing} page",
    "for account {n}",
    "in the {thing} table",
    "please",
    "thanks",
    "as requested",
]
THINGS = ["checkout", "profile", "billing", "dashboard", "settings", "reports"]


def build_rows(count: int, seed: int = 42) -> list[tuple[str, int]]:
    random.seed(seed)
    rows: list[tuple[str, int]] = []
    per_class = count // 2

    for _ in range(per_class):
        template = random.choice(INJECTION_TEMPLATES)
        text = template.format(
            n=random.randint(1, 99),
            cols=random.choice(COLUMNS),
            table=random.choice(TABLES),
        )
        text = (
            random.choice(PREFIXES)
            + text
            + random.choice(SUFFIXES).format(n=random.randint(1, 20))
        )
        rows.append((text.strip(), 1))

    for _ in range(per_class):
        text = random.choice(NORMAL_TEMPLATES)
        if random.random() < 0.6:
            text = f"{text} {random.choice(FILLERS).format(n=random.randint(1, 99), thing=random.choice(THINGS))}"
        if random.random() < 0.15:
            text = text.lower()
        if random.random() < 0.1:
            text = text.upper()
        rows.append((text.strip(), 0))

    random.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4000, help="total rows to generate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_rows(args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # UTF-16 without BOM handling differences: keep it simple and UTF-8 here;
    # src/train_v4.py reads the real dataset as UTF-16.
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        handle.write("Sentence,Label\n")
        for text, label in rows:
            escaped = text.replace('"', '""')
            handle.write(f'"{escaped}",{label}\n')

    print(f"Wrote {len(rows):,} rows to {args.output}")


if __name__ == "__main__":
    main()
