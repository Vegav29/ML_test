import sys
import os
import pandas as pd
from descriptive import DataLoader,Features
from visualize import Plots

#Wfrom src.features.build_features import Features
from config import DATA_PATH_RAW,DATA_PATH_INTERIM,FIGURES_PATH
# Set project root and import necessary modules
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
dataset = DataLoader(filepath=f"{DATA_PATH_RAW}/Intern Housing Data India.csv")
data = dataset.load_csv()
data.info()
features = Features(data)
processed_data = features.preprocess()

# Generate and print summary statistics
print(features.generate_summary_statistics())

# Check for missing values
print(features.check_missing_values())

# Print data types of the columns
print(features.get_data_types())
bn/ 
# Sabn/ ve processed data as a pickle object
output_pbn/ath = os.path.join(DATA_PATH_INTERIM, "data.pkl")
features.save(output_path)
plots = Plots(data)
num_cols = ["housing_median_age","total_rooms","total_bedrooms","population"]
for col in num_cols:
    plots.barplot(col,f"{FIGURES_PATH}/sales_trends.png")