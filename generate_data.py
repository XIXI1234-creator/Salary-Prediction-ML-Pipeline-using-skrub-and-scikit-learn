"""
Generate a realistic synthetic salary dataset.

This script produces `data/employee_salaries.csv`, a tabular dataset
that mimics the structure of public-sector salary records (inspired by
the Montgomery County, MD open salary data and the skrub
`fetch_employee_salaries` schema).

The dataset is intentionally messy enough to make skrub useful:
- mixed numeric, categorical and date columns
- high-cardinality categories (job titles)
- a few missing values
- realistic noise around an underlying salary model

Run once before opening the notebook:
    python src/generate_data.py
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)

# --- Reference vocabularies (kept small but realistic) ----------------------

DEPARTMENTS = {
    "POL": ("Department of Police", 1.10),
    "FRS": ("Fire and Rescue Services", 1.08),
    "HHS": ("Department of Health and Human Services", 0.98),
    "DOT": ("Department of Transportation", 1.00),
    "LIB": ("Department of Public Libraries", 0.88),
    "FIN": ("Department of Finance", 1.05),
    "GEN": ("Department of General Services", 0.95),
    "COR": ("Correction and Rehabilitation", 1.02),
    "LIQ": ("Department of Liquor Control", 0.92),
    "PER": ("Department of Permitting Services", 0.97),
}

# (job title, base salary, std)
JOB_TITLES = [
    ("Police Officer III", 78_000, 9_000),
    ("Police Sergeant", 95_000, 10_000),
    ("Firefighter / Rescuer III", 72_000, 8_000),
    ("Fire Captain", 105_000, 11_000),
    ("Social Worker IV", 68_000, 7_500),
    ("Public Health Nurse", 76_000, 9_000),
    ("Bus Operator", 52_000, 6_000),
    ("Transit Manager", 92_000, 10_000),
    ("Librarian II", 58_000, 6_500),
    ("Library Assistant", 42_000, 5_000),
    ("Accountant", 71_000, 8_000),
    ("Senior Financial Analyst", 98_000, 11_000),
    ("Office Services Coordinator", 55_000, 6_000),
    ("Correctional Officer III", 66_000, 7_500),
    ("Liquor Store Clerk II", 41_000, 4_500),
    ("Permit Technician", 58_000, 6_500),
    ("IT Specialist III", 95_000, 12_000),
    ("Senior Engineer", 112_000, 13_000),
    ("Administrative Specialist II", 60_000, 6_500),
    ("Planning Specialist III", 82_000, 9_000),
]

ASSIGNMENT_CATEGORIES = ["Fulltime-Regular", "Parttime-Regular"]
GENDERS = ["F", "M"]


def generate_salaries(n: int = 9_500) -> pd.DataFrame:
    """Generate `n` synthetic employee salary records."""
    rows = []

    dept_codes = list(DEPARTMENTS.keys())
    for _ in range(n):
        # Categorical features
        dept_code = RNG.choice(dept_codes)
        dept_name, dept_factor = DEPARTMENTS[dept_code]

        title, base, std = JOB_TITLES[RNG.integers(len(JOB_TITLES))]
        gender = RNG.choice(GENDERS)
        assignment = RNG.choice(ASSIGNMENT_CATEGORIES, p=[0.92, 0.08])
        division = f"{dept_code} Division {RNG.integers(1, 9)}"

        # Hire date — anywhere from 1985 to 2022
        year_hired = int(RNG.integers(1985, 2023))
        month = int(RNG.integers(1, 13))
        day = int(RNG.integers(1, 28))
        date_first_hired = f"{month:02d}/{day:02d}/{year_hired}"

        # Salary model: base * department factor * tenure bonus + noise
        tenure = 2024 - year_hired
        tenure_bonus = 1 + 0.012 * min(tenure, 30)  # cap experience effect
        part_time_factor = 0.55 if assignment == "Parttime-Regular" else 1.0
        noise = RNG.normal(0, std)

        salary = base * dept_factor * tenure_bonus * part_time_factor + noise
        salary = float(np.clip(salary, 25_000, 250_000))

        rows.append({
            "gender": gender,
            "department": dept_code,
            "department_name": dept_name,
            "division": division,
            "assignment_category": assignment,
            "employee_position_title": title,
            "date_first_hired": date_first_hired,
            "year_first_hired": year_hired,
            "current_annual_salary": round(salary, 2),
        })

    df = pd.DataFrame(rows)

    # Inject a small amount of missingness so preprocessing has work to do
    missing_mask = RNG.random(len(df)) < 0.015
    df.loc[missing_mask, "gender"] = np.nan
    missing_mask2 = RNG.random(len(df)) < 0.005
    df.loc[missing_mask2, "division"] = np.nan

    return df


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, "employee_salaries.csv"))

    df = generate_salaries()
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} rows -> {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
