from __future__ import annotations
import argparse, hashlib, random, string
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

FIRST = ["Asha","Kavin","Meena","Ravi","Nila","Arun","Divya","Vikram","Latha","Surya"]
LAST = ["Kumar","Rao","Iyer","Singh","Nair","Das","Patel","Shah","Menon","Bose"]
STREETS = ["Lake Road","Temple Street","Market Road","River Avenue","Hill View","Station Road"]

def _token(rng: random.Random, n=10):
    chars = string.ascii_uppercase + string.digits
    return "".join(rng.choice(chars) for _ in range(n))

def generate_synthetic_pii(n: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    base = datetime(2025, 1, 1)
    rows = []
    for i in range(n):
        full_name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        phone = f"+91-{rng.randint(60000,99999)}-{rng.randint(10000,99999)}"
        dob = datetime(rng.randint(1940, 2002), rng.randint(1,12), rng.randint(1,28))
        address = f"{rng.randint(1,399)}, {rng.choice(STREETS)}"
        zipcode = str(rng.randint(100000, 999999))
        adm = base + timedelta(minutes=rng.randint(0, 365*24*60))
        mrn = f"MRN-{i:08d}-{_token(rng,4)}"
        national = hashlib.sha256(f"SYNTHETIC-{seed}-{i}".encode()).hexdigest()
        rows.append({
            "national_identity_hash": national,
            "full_name": full_name,
            "contact_number": phone,
            "street_address": address,
            "zip_code": zipcode,
            "date_of_birth": dob.date().isoformat(),
            "admission_timestamp": adm.isoformat(),
            "medical_record_number": mrn,
        })
    return pd.DataFrame(rows)

def overlay(clinical: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    pii = generate_synthetic_pii(len(clinical), seed=seed)
    clinical = clinical.reset_index(drop=True).copy()
    return pd.concat([clinical, pii], axis=1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="synthetic_pii.csv")
    a = p.parse_args()
    generate_synthetic_pii(a.n, a.seed).to_csv(a.output, index=False)
    print(a.output)
