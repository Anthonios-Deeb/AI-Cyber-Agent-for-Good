import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import random

# Load the fine-tuned model
print("Loading fine-tuned model...")
model_path = "../models/llama2-finetuned-cyberagent"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

# Function to predict
def predict(flow_info):
    # Build prompt
    prompt = f"""### Instruction:
Analyze the following network flow and decide if it is normal or botnet activity. Explain your reasoning.

Flow Details:
{flow_info}

### Response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            use_cache=False,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the answer part
    answer_start = response.find("### Response:") + len("### Response:")
    print("\n🔎 Model Response:\n")
    print(response[answer_start:].strip())

REQUIRED_FIELDS = [
    "StartTime:", "Duration:", "Proto:", "SrcAddr:", "Sport:", "Dir:",
    "DstAddr:", "Dport:", "State:", "TotPkts:", "TotBytes:", "SrcBytes:"
]

def is_valid_flow(flow_str):
    return all(field in flow_str for field in REQUIRED_FIELDS)

if __name__ == "__main__":
    print("Choose mode:")
    print("1 - Enter flow(s) manually (up to 3)")
    print("2 - Random flows from dataset")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        print("\nPlease use the following format for each flow:")
        print("StartTime: <time>, Duration: <duration> seconds, Proto: <protocol>, SrcAddr: <source IP>, Sport: <source port>, Dir: <direction>, DstAddr: <destination IP>, Dport: <destination port>, State: <state>, TotPkts: <packet count>, TotBytes: <byte count>, SrcBytes: <source byte count>\n")
        print("Enter up to 3 flows (press Enter to stop):")
        count = 0
        while count < 3:
            flow_input = input(f"\nEnter flow #{count+1} (or press Enter to finish): ").strip()
            if not flow_input:
                break
            if is_valid_flow(flow_input):
                predict(flow_input)
                count += 1
            else:
                print("⚠️ Invalid flow format. Please include all required fields.")

    elif mode == "2":
        df = pd.read_csv("../data/Filtered/filtered_ctu13_flows.csv")
        random_rows = df.sample(n=10)

        for idx, random_row in random_rows.iterrows():
            flow_info = f"""StartTime: {random_row['StartTime']}, Duration: {random_row['Dur']} seconds, Proto: {random_row['Proto']}, SrcAddr: {random_row['SrcAddr']}, Sport: {random_row['Sport']}, Dir: {random_row['Dir']}, DstAddr: {random_row['DstAddr']}, Dport: {random_row['Dport']}, State: {random_row['State']}, TotPkts: {random_row['TotPkts']}, TotBytes: {random_row['TotBytes']}, SrcBytes: {random_row['SrcBytes']}"""
            print(f"\n🔹 Randomly selected flow #{idx+1}:")
            print(flow_info)
            predict(flow_info)
    else:
        print("Invalid option. Please enter 1 or 2.")