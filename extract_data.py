import pyreadr
import os
import pandas as pd
import re


def export_all_patient_data(root_dir):
    """
    Recursively finds .RData files, extracts patient IDs, and exports
    contained dataframes to patient-specific folders.
    """
    # Pattern to find patient IDs like UPXXXX
    patient_pattern = re.compile(r'UP\d{4}')

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.RData'):
                file_path = os.path.join(subdir, file)

                # Identify the patient ID from the filename
                match = patient_pattern.search(file)
                patient_id = match.group(0) if match else "Unknown_Patient"

                # Create a specific output directory for this patient
                output_folder = os.path.join("data/patient_data", patient_id)
                os.makedirs(output_folder, exist_ok=True)

                print(f"Processing: {file} for Patient: {patient_id}")

                try:
                    # Read the RData file
                    result = pyreadr.read_r(file_path)

                    # Export each object found in the RData file
                    for object_name, df in result.items():
                        if isinstance(df, pd.DataFrame):
                            # Name the CSV based on the R object name
                            csv_name = f"{object_name}.csv"
                            csv_path = os.path.join(output_folder, csv_name)

                            df.to_csv(csv_path, index=True)
                            print(f"  -> Exported '{object_name}' to {csv_path}")
                        else:
                            print(f"  -> Skipped '{object_name}': Not a DataFrame")

                except Exception as e:
                    print(f"  !! Error reading {file}: {e}")


if __name__ == "__main__":
    DATA_ROOT = 'data'
    export_all_patient_data(DATA_ROOT)