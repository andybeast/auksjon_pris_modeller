import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
import joblib
import re
import matplotlib.pyplot as plt

# === CONFIG ===
INPUT_FILE = "MERGED_cleaned_for_bert.json"
CACHE_EMBEDDINGS_FILE = "bert_embeddings.npy"
MODEL_DIR = "trained_models"
BATCH_SIZE = 64

# === Helper Functions ===
def contains_currency_word(title):
    currency_terms = ["krone", "kroner", "kr", "øre", "öre", "skilling", "krona", "kronur", "marks", "skillings", "speciedaler", "spesidaler", "spesidal", "species"]
    return any(term in title.lower() for term in currency_terms)



def extract_normalized_grade(title):
    title = title.strip()
    title_lower = title.lower()
    title_upper = title.upper()

    # Grade hierarchy: worst (left) to best (right)
    grade_order = [
    "2", "1-", "1", "1+", "1++", "01", "0/01", "0", "proof"
    ]
    valid_grades = set(grade_order)

    foreign_to_norwegian = {
        "UNC": "0",
        "MS": "0",
        "AU": "0/01",
        "XF": "01",
        "EF": "01",
        "VF": "1+",
        "F": "1",
        "VG": "1-",
        "G": "2"
    }

    # Step 0: Handle "proof" coins early
    if "proof" in title_lower or "speilglans" in title_lower:
        return "proof"

    def extract_worst_grade_from_combo(combo: str):
        cleaned_combo = combo.strip().replace(" ", "")
        if cleaned_combo in valid_grades:
            return cleaned_combo

        parts = re.split(r"[/|\\]", cleaned_combo)
        worst = None
        for part in parts:
            cleaned = re.sub(r"[^\d\+\-]", "", part.strip())
            if cleaned in valid_grades:
                if not worst or grade_order.index(cleaned) < grade_order.index(worst):
                    worst = cleaned
        return worst

    # Step 1: Match Norwegian "kv"/"kvalitet" patterns
    grade_pattern = re.findall(
    r"\bkv(?:alitet)?\.?\s*:? ?([0-9+\-/]{1,8})\b",
    title, flags=re.IGNORECASE
)

    for raw in grade_pattern:
        worst = extract_worst_grade_from_combo(raw)
        if worst:
            return worst

    # Step 2: Match any valid-looking grade or composite grade at END of title
    match = re.search(r"([0-9+\-/]{1,8})\s*$", title_lower)
    if match:
        raw = match.group(1)
        worst = extract_worst_grade_from_combo(raw)
        if worst:
            return worst

    # Step 3: Handle foreign grades like "UNC", "MS64", "VF"
    match_foreign = re.search(r"\b(UNC|MS\d{2}|AU|XF|EF|VF|F|VG|G)\b", title_upper)
    if match_foreign:
        foreign = match_foreign.group(1)
        if foreign.startswith("MS"):
            return "0"
        return foreign_to_norwegian.get(foreign, "ukjent")

    return "ukjent"


def extract_king_name(title):
    kings = [
        "Olav V", "Harald V", "Haakon VII", "Oscar II", "Oscar I", "Karl XIV Johan", "Karl XIII", "Frederik VI",
        "Christian VII", "Frederik V", "Christian VI", "Frederik IV", "Christian V", "Frederik III", "Christian IV",
        "Frederik II", "Christian III", "Frederik I", "Christian II", "Christian I", "Olav IV Håkonsson",
        "Håkon VI", "Magnus VII", "Håkon V", "Eirik II", "Magnus VI", "Håkon IV Håkonsson", "Hans"
    ]
    for king in kings:
        if king.lower() in title.lower():
            return king
    return "unknown"

def is_coin_roll(title):
    return bool(re.search(r"\brull(er)?\b", title.lower()))

def extract_country(title):
    country_aliases = {
        "Norway": ["norge", "norway"],
        "Sweden": ["sweden", "sverige"],
        "Denmark": ["denmark", "danmark"],
        "USA": ["usa", "us", "united states"],
        "Germany": ["germany", "deutschland", "tyskland"],
        "India": ["india"],
        "UK": ["united kingdom", "uk", "england", "great britain"]
    }
    title_lower = title.lower()
    for country, aliases in country_aliases.items():
        if any(alias in title_lower for alias in aliases):
            return country

    currency_country_map = {
        "Norway": ["krone", "kroner", "kr", "skilling", "speciedaler", "spesiedaler"],
        "Sweden": ["öre", "krona"]
    }
    for country, terms in currency_country_map.items():
        if any(term in title_lower for term in terms):
            return country

    return "unknown"

# === Load & Parse Data ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    coin_data = json.load(f)

records = []
for item in coin_data:
    title = item.get("title")
    price = item.get("price")
    raw_date = item.get("date")

    year_matches = re.findall(r"(18|19|20)\d{2}", title)
    if len(year_matches) != 1:
        continue
    coin_year = int(year_matches[0])

    if not contains_currency_word(title):
        continue

    if title and price and raw_date:
        try:
            dt = datetime.strptime(raw_date.strip(), "%d/%m/%Y %H:%M:%S")
            material_match = re.search(r"\b(gold|silver|bronze|copper|nickel|gull|sølv|brons|bronse|kobber|nikkel)\b", title.lower())
            pcgs_match = re.search(r"PCGS\s?[A-Z0-9+/]{4,}", title.upper())

            records.append({
                "title": title,
                "price": price,
                "year": dt.year,
                "month": dt.month,
                "weekday": dt.weekday(),
                "coin_year": coin_year,
                "grade": extract_normalized_grade(title),
                "material": material_match.group().lower() if material_match else "unknown",
                "pcgs": pcgs_match.group() if pcgs_match else "none",
                "king": extract_king_name(title),
                "is_roll": is_coin_roll(title),
                "country": extract_country(title),
                "date": dt
            })
        except Exception as e:
            print(f"⚠️ Skipping bad date '{raw_date}': {e}")

# Continue as before with DataFrame creation, plotting, and model training
df = pd.DataFrame(records)
# Define grade quality from worst to best
grade_order = ["2", "1-", "1", "1+", "1++", "01", "0/01", "0", "proof"]
grade_to_score = {grade: i for i, grade in enumerate(grade_order)}

# Convert normalized grade to ordinal score
df["grade_score"] = df["grade"].map(grade_to_score)

# Optional: drop rows with unknown or unmapped grades

print(f"✅ Loaded {len(df)} currency-related coin rows")

# === Plot distributions ===
plt.figure(figsize=(12, 6))
for country in df["country"].unique():
    country_df = df[df["country"] == country]
    plt.scatter(country_df["date"], country_df["price"], label=country, alpha=0.6)
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Coin Price Distribution by Country")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

# Sort grades by count for better legend order
grade_counts = df["grade"].value_counts()
for grade in grade_counts.index:
    grade_df = df[df["grade"] == grade]
    count = len(grade_df)
    plt.scatter(grade_df["date"], grade_df["price"], label=f"{grade} (n={count})", alpha=0.6)

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Coin Price Distribution by Grade")
plt.legend(title="Grade", loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()


threshold = df["price"].quantile(0.95)
df = df[df["price"] <= threshold]
print(f"✅ Removed top 5% prices above {threshold:,.2f}. Remaining rows: {len(df)}")

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists(CACHE_EMBEDDINGS_FILE):
    print("📦 Loading cached BERT embeddings...")
    bert_embeddings = np.load(CACHE_EMBEDDINGS_FILE)
    if bert_embeddings.shape[0] != len(df):
        print("⚠️ Embedding cache size mismatch — regenerating embeddings...")
        bert_embeddings = bert_model.encode(df["title"].tolist(), batch_size=BATCH_SIZE, show_progress_bar=True)
        np.save(CACHE_EMBEDDINGS_FILE, bert_embeddings)
else:
    print("🧠 Generating BERT embeddings...")
    bert_embeddings = bert_model.encode(df["title"].tolist(), batch_size=BATCH_SIZE, show_progress_bar=True)
    np.save(CACHE_EMBEDDINGS_FILE, bert_embeddings)





# === Encode categorical features ===
df["grade_encoded"] = pd.factorize(df["grade"])[0]
df["material_encoded"] = pd.factorize(df["material"])[0]
df["pcgs_encoded"] = pd.factorize(df["pcgs"])[0]
df["king_encoded"] = pd.factorize(df["king"])[0]
df["roll_encoded"] = df["is_roll"].astype(int)
df["log_price"] = np.log1p(df["price"])
y = df["log_price"].to_numpy()

structured_features = df[[
    "year", "month", "weekday", "coin_year",
    "grade_score", "material_encoded", "pcgs_encoded",
    "king_encoded", "roll_encoded"
]].to_numpy()

X = np.hstack([bert_embeddings, structured_features])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📚 Training on {len(X_train)} samples, testing on {len(X_test)}")

xgb_params = {
    "learning_rate": [0.12],              # best value
    "max_depth": [8],                     # best value
    "n_estimators": [250],                # best value
    "subsample": [0.9, 1.0],              # try slight variation
    "colsample_bytree": [0.8],            # fixed to best
    "reg_alpha": [0, 0.1],                # L1: try none + small
    "reg_lambda": [1, 5]                  # L2: default + stronger
}
xgb_search = GridSearchCV(
    XGBRegressor(verbosity=0),
    xgb_params,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("🔍 Starting XGBoost grid search...")
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print("✅ Best XGBoost params:", xgb_search.best_params_)



def evaluate_and_plot(model, name):
    y_pred_log = model.predict(X_test)
    y_pred_price = np.expm1(y_pred_log)
    y_test_price = np.expm1(y_test)

    r2 = r2_score(y_test_price, y_pred_price)
    rmse = np.sqrt(mean_squared_error(y_test_price, y_pred_price))
    mae = mean_absolute_error(y_test_price, y_pred_price)
    mape = np.mean(np.abs((y_test_price - y_pred_price) / (y_test_price + 1e-8))) * 100

    print(f"\n📊 {name} results:")
    print(f"   R² Score:       {r2:.4f}")
    print(f"   RMSE (NOK):     {rmse:,.2f}")
    print(f"   MAE (NOK):      {mae:,.2f}")
    print(f"   MAPE (%):       {mape:.2f}%")

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_price, y_pred_price, alpha=0.4, label="Predicted")
    plt.plot([y_test_price.min(), y_test_price.max()], [y_test_price.min(), y_test_price.max()], 'r--', label="Perfect Prediction")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Actual Price (log scale)")
    plt.ylabel("Predicted Price (log scale)")
    plt.title(f"Predicted vs Actual Prices — {name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{name}_coin_model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Saved {name} model at {model_path}")

    with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
        f.write(f"{name}_coin_model.pkl")

evaluate_and_plot(best_xgb, "xgboost_gridsearch")

pd.DataFrame(xgb_search.cv_results_).to_csv(os.path.join(MODEL_DIR, "xgb_grid_results.csv"), index=False)

