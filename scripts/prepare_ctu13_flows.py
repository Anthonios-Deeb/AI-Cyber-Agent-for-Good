import pandas as pd
import os

# 1. Set the base folder where CTU-13 scenario folders are located
base_path = "../data/CTU-13-Dataset"

# 2. Prepare an empty list to collect all flows
all_dataframes = []

# 3. Loop through folders (example: 1 to 3 now, later you can do 1 to 13)
for i in range(1, 4):  # Load folders 1,2,3 for now
    folder_path = os.path.join(base_path, str(i))
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".binetflow"):
            file_path = os.path.join(folder_path, filename)
            print(f"Loading: {file_path}")

            # Load binetflow normally since it already has headers
            df = pd.read_csv(
                file_path,
                delimiter=",",
                engine="python"
            )

            all_dataframes.append(df)

# 4. Combine all flows into one big dataframe
full_dataset = pd.concat(all_dataframes, ignore_index=True)

print("\nAll flows loaded! Shape:", full_dataset.shape)
print("\nAvailable Columns:\n", list(full_dataset.columns))

# 5. Filter only BOTNET and NORMAL traffic based on 'Label'
filtered_dataset = full_dataset[full_dataset["Label"].str.contains("Botnet|Normal", na=False)]

print("\nFiltered dataset shape (Botnet + Normal only):", filtered_dataset.shape)

# 6. Save the filtered dataset to your specified output directory
output_dir = "../data/Filtered"
os.makedirs(output_dir, exist_ok=True)
filtered_dataset.to_csv(os.path.join(output_dir, "filtered_ctu13_flows.csv"), index=False)
print("\nSaved filtered flows to '../data/Filtered/filtered_ctu13_flows.csv'")