import pandas as pd
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report


# Load model and tokenizer
model_path = "../models/llama2-finetuned-cyberagent/checkpoint-500"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
model.eval()

# Instruction used during fine-tuning
instruction = "Assess the traffic to detect any unusual behavior."

# Load the flows from CSV
data_path = "../data/Filtered/filtered_ctu13_flows.csv"
df = pd.read_csv(data_path)

# Randomly sample 20 rows
sampled = df.sample(n=20, random_state=42)

# Format flows for LLM input
flows = []
for _, row in sampled.iterrows():
    label = row['Label'].lower()  # expects 'Normal' or 'Botnet'
    flow_info = f"""Start Time: {row['StartTime']}
Duration: {row['Duration']} seconds
Protocol: {row['Proto']}
Source IP: {row['SrcAddr']}:{row['Sport']}
Destination IP: {row['DstAddr']}:{row['Dport']}
State: {row['State']}
Total Packets: {row['TotPkts']}
Total Bytes: {row['TotBytes']}
Source Bytes: {row['SrcBytes']}"""
    flows.append((label, flow_info))

# Collect all predictions and labels for detailed evaluation
all_preds = []
all_labels = []

for label, flow_info in flows:
    prompt = f"""### Instruction:
{instruction}

### Response:
{flow_info}"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, temperature=0.2, top_p=0.95, do_sample=True)

    response = tokenizer.decode(output[0], skip_special_tokens=True).lower()
    prediction = "botnet" if "botnet" in response else "normal"
    all_preds.append(prediction)
    all_labels.append(label)

# Summary
correct = sum([pred == label for pred, label in zip(all_preds, all_labels)])
total = len(flows)
accuracy = correct / total * 100
print(f"\n\n📊 Summary: {correct}/{total} correct")
print(f"✅ Accuracy: {accuracy:.2f}%")

# Convert to binary format
binary_map = {"botnet": 1, "normal": 0}
y_true = [binary_map[label] for label in all_labels]
y_pred = [binary_map[pred] for pred in all_preds]

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["NORMAL", "BOTNET"]))