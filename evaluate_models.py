import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the same dataset used in baseLine.py
df = pd.read_csv("../data/Filtered/filtered_ctu13_flows.csv")
df = df.dropna(subset=["Label"])
df["Label"] = df["Label"].apply(lambda x: 1 if "botnet" in x.lower() else 0)

features = ["Dur", "Proto", "Sport", "Dport", "TotPkts", "TotBytes", "SrcBytes"]
X = df[features].copy()
X = X.dropna()

# Convert hex ports if necessary
for col in ["Sport", "Dport"]:
    X[col] = X[col].apply(lambda x: int(str(x), 16) if isinstance(x, str) and str(x).startswith("0x") else (int(x) if pd.notnull(x) and str(x).isdigit() else 0))

# Encode protocol
X["Proto"] = LabelEncoder().fit_transform(X["Proto"])
X = X.fillna(0)
y = df["Label"].loc[X.index]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Take only 10 samples from the test set for fair comparison
X_test_sample = X_test.iloc[:10]
y_test_sample = y_test.iloc[:10]

# --------- Train Random Forest Model (baseline) ----------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test_sample)

# --------- Load LLM Model ----------
tokenizer = AutoTokenizer.from_pretrained("../models/llama2-finetuned-cyberagent/checkpoint-500")
model = AutoModelForCausalLM.from_pretrained("../models/llama2-finetuned-cyberagent/checkpoint-500").to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# --------- Generate predictions from LLM ----------
def format_flow(row):
    return f"StartTime: ..., Duration: {row['Dur']}, Proto: ..., Sport: {row['Sport']}, Dport: {row['Dport']}, TotPkts: {row['TotPkts']}, TotBytes: {row['TotBytes']}, SrcBytes: {row['SrcBytes']}"

llm_preds = []
for _, row in X_test_sample.iterrows():
    prompt = format_flow(row)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(output[0], skip_special_tokens=True).lower()
    label = 1 if "botnet" in response else 0
    llm_preds.append(label)

# --------- Evaluation ---------
print("Random Forest Accuracy:", accuracy_score(y_test_sample, rf_preds))
print("LLM Accuracy:", accuracy_score(y_test_sample, llm_preds))

print("\nRandom Forest Report:\n", classification_report(y_test_sample, rf_preds))
print("\nLLM Report:\n", classification_report(y_test_sample, llm_preds))
