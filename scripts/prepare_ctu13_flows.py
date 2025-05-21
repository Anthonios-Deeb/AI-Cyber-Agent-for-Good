"""
CTU-13 Flow Preprocessing Script

This script loads binetflow files from selected CTU-13 scenario folders, filters them to include only 'Botnet' and 'Normal' labeled flows,
and saves the combined filtered dataset to a single CSV file for further analysis or model training.

"""
import pandas as pd
import os

# --- Configuration ---
# Set the base directory containing the CTU-13 dataset folders.
base_path = "../data/CTU-13-Dataset"

# --- Data Collection ---
# Initialize a list to store all individual DataFrames from the .binetflow files.
all_dataframes = []

# Loop through a predefined subset of CTU-13 folders (1 to 3 as an example).
# This can be expanded later to include all scenarios (1 to 13).
for i in range(1, 4):  # Load folders 1,2,3 for now
    folder_path = os.path.join(base_path, str(i))
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".binetflow"):
            # Look for .binetflow files inside each scenario folder.
            file_path = os.path.join(folder_path, filename)
            print(f"Loading: {file_path}")

            # Load binetflow normally since it already has headers
            df = pd.read_csv(
                file_path,
                delimiter=",",
                engine="python"
            )

            all_dataframes.append(df)

# Combine all flows into one big dataframe
full_dataset = pd.concat(all_dataframes, ignore_index=True)

print("\nAll flows loaded! Shape:", full_dataset.shape)
print("\nAvailable Columns:\n", list(full_dataset.columns))

# Filter only BOTNET and NORMAL traffic based on 'Label'
filtered_dataset = full_dataset[full_dataset["Label"].str.contains("Botnet|Normal", na=False)]

print("\nFiltered dataset shape (Botnet + Normal only):", filtered_dataset.shape)

# Save the filtered dataset to your specified output directory
output_dir = "../data/Filtered"
os.makedirs(output_dir, exist_ok=True)
filtered_dataset.to_csv(os.path.join(output_dir, "filtered_ctu13_flows.csv"), index=False)
print("\nSaved filtered flows to '../data/Filtered/filtered_ctu13_flows.csv'")