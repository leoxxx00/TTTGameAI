import pandas as pd

# Load the CSV file with error handling using on_bad_lines
file_path = 'tic_tac_toe_data.csv'  # Replace with your actual file path
print(f"Loading data from {file_path}...")

try:
    # Use on_bad_lines to skip problematic lines
    df = pd.read_csv(file_path, on_bad_lines='skip')  # Skips bad lines
    print("Data loaded successfully!")
except pd.errors.ParserError as e:
    print("ParserError encountered while reading the file:", e)

# Display the first few rows of the data to confirm
print("Initial data preview:")
print(df.head())

# Keep only rows where any of the values in the row are 1
print("Keeping only rows that contain 1 in any column...")
df_cleaned = df[df.isin([1]).any(axis=1)]
print("Rows with 1 retained.")

# Display the cleaned data for preview
print("Cleaned data preview:")
print(df_cleaned.head())

# Save the cleaned data back to the same CSV
df_cleaned.to_csv(file_path, index=False)
print(f"Cleaned data saved back to {file_path}")
