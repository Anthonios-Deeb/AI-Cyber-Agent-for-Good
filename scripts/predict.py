#Anthonios Deeb
#Wasim Shebalny
"""
CyberFlow AI Agent - predict.py

This script allows real-time interaction with a fine-tuned LLaMA 2 model for detecting and reasoning about botnet activity in network flows.
Users can either input flow data manually or analyze a random flow sampled from the CTU-13 dataset.
The model returns both a classification and a human-readable explanation.

"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import random
import os
import sys
import re # Import the regular expression module

# --- Configuration ---
MODEL_PATH = "../models/llama2-finetuned-cyberagent"
DATASET_PATH = "../data/Filtered/filtered_ctu13_flows.csv"
REQUIRED_FIELDS = [
    "StartTime:", "Duration:", "Proto:", "SrcAddr:", "Sport:", "Dir:",
    "DstAddr:", "Dport:", "State:", "TotPkts:", "TotBytes:", "SrcBytes:"
]

# Define a standard example flow for display and for testing extraction
EXAMPLE_FLOW = "StartTime: 1678886400, Duration: 10 seconds, Proto: TCP, SrcAddr: 192.168.1.10, Sport: 54321, Dir: ->, DstAddr: 10.0.0.5, Dport: 80, State: EST, TotPkts: 20, TotBytes: 2000, SrcBytes: 1000"

# --- Model Loading ---
# Loads the fine-tuned LLaMA 2 model and tokenizer for causal language modeling.
# Returns the tokenizer, model, and the selected device (CPU or GPU).
def load_model():
    """Loads the fine-tuned LLaMA 2 model and tokenizer."""
    print("\n🚀 Initializing CyberFlow AI Agent...")
    print("⏳ Loading fine-tuned LLaMA 2 model and tokenizer...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device.upper()}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
        print("✅ Model loaded successfully!")
        return tokenizer, model, device
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Please ensure the model path '{MODEL_PATH}' is correct and accessible.")
        print("Exiting...")
        sys.exit(1)

tokenizer, model, device = load_model()

# --- Prediction Function ---
# Constructs a structured prompt including the flow details and queries the model.
# Returns the response from the LLaMA model, trimmed to the relevant output.
def predict_flow(flow_info):
    """
    Constructs the prompt and queries the LLaMA 2 model for prediction.
    Returns the extracted model response.
    """
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
    answer_start = response.find("### Response:") + len("### Response:")
    return response[answer_start:].strip()


# --- Flow Extraction and Validation ---
# Dynamically create a regex pattern based on REQUIRED_FIELDS
# This pattern looks for "Field1: <anything>, Field2: <anything>, ..."
# It's flexible with commas and whitespace between fields.
# We'll just look for the field names, and capture everything after them up to the next field name
# or the end of the line.
# This makes it robust to extra text before or after the actual flow details.
# This regex will find the first occurrence of "StartTime:" and then try to capture everything
# that looks like the required fields.
_pattern_parts = [re.escape(field) + r'\s*.*?' for field in REQUIRED_FIELDS]
# The regex tries to match the sequence of required fields, allowing anything in between them.
# The (?:...) creates a non-capturing group.
# We'll ensure the first field is found to anchor the match.
FLOW_REGEX = re.compile(
    r'.*?' + # Lazily match any characters at the beginning (conversational text)
    r'(' +  # Start capturing group for the actual flow details
    re.escape(REQUIRED_FIELDS[0]) + r'.*?' + # Match the first field and anything after it lazily
    r'(?:' + r',\s*'.join(_pattern_parts[1:]) + r')?' + # Optionally match the rest of the fields
    r'.*' + # Capture any remaining text that might be part of the last value
    r')' +  # End capturing group
    r'.*$', # Lazily match any characters at the end (more conversational text)
    re.DOTALL # Allows . to match newlines, useful if flow details are multiline
)

def extract_flow_details(full_input_str):
    """
    Extracts the structured network flow details from a larger string,
    ignoring conversational text before or after.
    Returns the extracted flow string or None if not found.
    """
    match = FLOW_REGEX.search(full_input_str)
    if match:
        extracted = match.group(1).strip()
        # After extraction, ensure all required fields are truly present in the extracted part.
        # This double-check makes it more robust.
        if all(field.replace(':', '') in extracted for field in [f.replace(':', '') for f in REQUIRED_FIELDS]):
             return extracted
    return None

def is_valid_flow_format(flow_str_to_validate):
    """Checks if a string (assumed to be just flow data) contains all required fields."""
    # This function is now simplified, as extract_flow_details does the heavy lifting.
    # It just checks if the clean, extracted flow string contains all essential parts.
    return all(field in flow_str_to_validate for field in REQUIRED_FIELDS)


# --- Main Terminal Interface ---
# Provides a CLI interface for users to interact with the CyberFlow AI Agent.
# Offers two modes: manual entry or random selection from the dataset.
if __name__ == "__main__":
    print("\n--- CyberFlow AI Agent (Terminal Mode) ---")
    print("Type '1' to enter flow details manually.")
    print("Type '2' to analyze a random flow from the dataset.")
    print("Type 'exit' to quit the agent.")
    print("------------------------------------------")

    while True:
        user_input = input("\nEnter your choice (1, 2, or 'exit'): ").strip().lower()

        if user_input == "exit":
            print("👋 Exiting CyberFlow AI Agent. Goodbye!")
            break
        elif user_input == "1":
            print("\n📋 Manual Flow Entry:")
            print("You can type your flow details along with any message you want.")
            print("The agent will try to extract the flow information.")
            print(f"Expected flow format: {', '.join([f'{field} <value>' for field in REQUIRED_FIELDS])}")
            print("    (Press Enter without typing to return to main menu)")

            show_example = input("Do you want to see an example flow? (y/n): ").strip().lower()
            if show_example == 'y':
                print("\nHere's an example of a correctly formatted flow:")
                print(f"    {EXAMPLE_FLOW}\n")
                print("You can type something like: 'Please analyze this flow: " + EXAMPLE_FLOW + "'")
            elif show_example == 'n':
                print("Okay, proceed with your input.\n")
            else:
                print("Invalid input. Proceeding without showing an example.\n")

            flow_input_raw = input("Enter your message and flow details: ").strip()
            
            if not flow_input_raw:
                continue
            
            # --- New Extraction Logic ---
            extracted_flow = extract_flow_details(flow_input_raw)

            if extracted_flow:
                print("\n🔍 Analyzing extracted flow details...")
                print(f"Extracted: {extracted_flow}") # Show the user what was extracted
                try:
                    model_response = predict_flow(extracted_flow) # Use the extracted flow
                    print("\n--- Model Response ---")
                    print(model_response)
                    print("----------------------")
                except Exception as e:
                    print(f"❌ An error occurred during prediction: {e}")
            else:
                print("⚠️ Could not find valid flow details in your input.")
                print("Please ensure your message contains the required fields like 'StartTime:', 'Duration:', etc.")
        
        elif user_input == "2":
            print("\n🎲 Analyzing a random flow from the dataset...")
            if not os.path.exists(DATASET_PATH):
                print(f"❌ Dataset not found at: '{DATASET_PATH}'. Please check the path and try again.")
                continue

            try:
                df = pd.read_csv(DATASET_PATH)
                if df.empty:
                    print("⚠️ The dataset is empty. Cannot select a random flow.")
                    continue
                random_row = df.sample(n=1).iloc[0]
                
                flow_info = f"""StartTime: {random_row.get('StartTime', 'N/A')}, Duration: {random_row.get('Dur', 'N/A')} seconds, Proto: {random_row.get('Proto', 'N/A')}, SrcAddr: {random_row.get('SrcAddr', 'N/A')}, Sport: {random_row.get('Sport', 'N/A')}, Dir: {random_row.get('Dir', 'N/A')}, DstAddr: {random_row.get('DstAddr', 'N/A')}, Dport: {random_row.get('Dport', 'N/A')}, State: {random_row.get('State', 'N/A')}, TotPkts: {random_row.get('TotPkts', 'N/A')}, TotBytes: {random_row.get('TotBytes', 'N/A')}, SrcBytes: {random_row.get('SrcBytes', 'N/A')}"""
                
                print("\nSelected Random Flow Details:")
                print(flow_info)
                
                print("\n🔍 Analyzing random flow...")
                model_response = predict_flow(flow_info)
                print("\n--- Model Response ---")
                print(model_response)
                print("----------------------")

            except FileNotFoundError:
                print(f"❌ Error: Dataset file not found at '{DATASET_PATH}'.")
            except pd.errors.EmptyDataError:
                print(f"❌ Error: Dataset file '{DATASET_PATH}' is empty.")
            except KeyError as ke:
                print(f"❌ Error: Missing expected column in dataset: {ke}. Please check your CSV file.")
            except Exception as e:
                print(f"❌ An unexpected error occurred while processing the dataset: {e}")
        else:
            print("🤔 Invalid choice. Please enter '1', '2', or 'exit'.")
