"""
This script generates instruction-based training examples from the CTU-13 network flow dataset.
It samples labeled botnet and normal flows, formats them into natural language prompts with
explanatory outputs, and saves the results in JSONL format for fine-tuning LLMs.
"""

import pandas as pd
import random
import json
import os

# Path to the filtered CTU-13 network flow data
input_file = "../data/Filtered/filtered_ctu13_flows.csv"

# Load the dataset into a pandas DataFrame
df = pd.read_csv(input_file)

# Separate flows into botnet and normal based on their labels
botnet_df = df[df["Label"].str.contains("From-Botnet", case=False, na=False)]
normal_df = df[df["Label"].str.contains("From-Normal", case=False, na=False)]

# Randomly sample an equal number of botnet and normal flows for balanced training
botnet_sample = botnet_df.sample(n=500, random_state=42)
normal_sample = normal_df.sample(n=500, random_state=42)

# Combine and shuffle the sampled flows to avoid ordering bias
sampled_df = pd.concat([botnet_sample, normal_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

# Predefined instruction templates for each class to simulate prompt diversity
botnet_instruction_templates = [
    "Analyze the flow carefully. Is there any evidence of botnet activity?",
    "Review the flow for malicious behavior.",
    "Evaluate if this communication suggests infection or C2 activity."
]

normal_instruction_templates = [
    "Analyze the flow and classify it as benign or malicious.",
    "Review this network session and confirm if it seems normal.",
    "Assess the traffic to detect any unusual behavior."
]

# Prepare the training examples by pairing flow metadata with natural language prompts and labels
training_examples = []

for idx, row in sampled_df.iterrows():
    label = row['Label'].lower()

    # Select a random instruction template based on the flow's class label
    if 'botnet' in label:
        instruction = random.choice(botnet_instruction_templates)
    else:
        instruction = random.choice(normal_instruction_templates)

    # Format the flow's key metadata into a human-readable input format
    input_text = (
        f"[FLOW INFO]\n"
        f"Start Time: {row['StartTime']}\n"
        f"Duration: {row['Dur']} seconds\n"
        f"Protocol: {row['Proto']}\n"
        f"Source IP: {row['SrcAddr']}:{row['Sport']}\n"
        f"Destination IP: {row['DstAddr']}:{row['Dport']}\n"
        f"State: {row['State']}\n"
        f"Total Packets: {row['TotPkts']}\n"
        f"Total Bytes: {row['TotBytes']}\n"
        f"Source Bytes: {row['SrcBytes']}"
    )

    # Generate a professional explanation of the flow's behavior based on its characteristics
    if 'botnet' in label:
        if row['Dport'] in [22, 23]:
            explanation = (
                "This flow is suspicious because it attempts to connect to a remote access port (SSH/Telnet), "
                "which is often exploited by botnets for unauthorized access."
            )
        elif row['TotPkts'] < 5 and row['State'].startswith('S'):
            explanation = (
                "This flow is suspicious due to very few packets and an incomplete connection state, "
                "suggesting possible botnet scanning or failed connections."
            )
        else:
            explanation = (
                "This flow shows characteristics consistent with botnet activity, "
                "such as unusual communication patterns or port usage."
            )
    elif 'normal' in label:
        if row['Dport'] in [80, 443, 53]:
            explanation = (
                "This flow appears benign, communicating over standard web or DNS ports, "
                "with typical packet and byte counts observed."
            )
        else:
            explanation = (
                "This flow seems normal, showing no significant anomalies in communication behavior."
            )
    else:
        explanation = "Unable to determine the nature of this flow."

    # Construct a dictionary with instruction, input, and output fields
    training_example = {
        "instruction": instruction,
        "input": input_text.strip(),
        "output": explanation.strip()
    }

    training_examples.append(training_example)

# Create the output directory if it doesn't exist
output_dir = "../data/FineTuning"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "training_data.jsonl")

# Write each training example as a JSONL entry
with open(output_file, "w", encoding="utf-8") as f:
    for example in training_examples:
        json.dump(example, f)
        f.write("\n")

# Confirm how many examples were saved
print(f"Saved {len(training_examples)} training examples to {output_file}")