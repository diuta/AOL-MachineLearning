import pandas as pd

# --- Configuration ---
CSV_FILE_PATH = './dataset/data_moods.csv'
JSON_FILE_PATH = 'data_moods.json' # Will be saved in the current directory

def convert_csv_to_json(csv_path, json_path):
    """Converts a CSV file to a JSON file (array of objects)."""
    print(f"Reading CSV from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        print("Please ensure the path is correct and the dataset directory is in the root of your project.")
        return

    print("Converting DataFrame to JSON...")
    # Orient='records' creates a list of records (like rows in the CSV)
    json_data = df.to_json(orient='records', indent=4)
    
    print(f"Saving JSON to {json_path}...")
    with open(json_path, 'w') as f:
        f.write(json_data)
    print("Conversion successful.")

if __name__ == '__main__':
    convert_csv_to_json(CSV_FILE_PATH, JSON_FILE_PATH)
