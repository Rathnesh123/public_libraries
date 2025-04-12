import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# Load dataset
df = pd.read_csv('C:/Users/ASUS/Downloads/Public_Libraries.csv')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Show column names
print("Available Columns:")
print(df.columns.tolist())

# OPTIONAL: Display first few rows
print("\nPreview of data:")
print(df.head())
print("Satistics: ")
print(df.describe())
print("Dataset info: ")
print(df.info())

# Data Cleaning
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())


# Use actual column names from the dataset
county_col = 'County'
visits_col = 'Total Library Visits'
staff_col = 'Wages & Salaries Expenditures'  # Using as proxy for staff
circ_col = 'Total Circulation'

# Drop NA rows for plotting simplicity
df_clean = df[[county_col, visits_col, staff_col, circ_col]].dropna()

# Barplot: Average Visits per County (top 20 for readability)
plt.figure(figsize=(12, 6))
top_counties = df_clean.groupby(county_col)[visits_col].mean().sort_values(ascending=False).head(20).index
sns.barplot(x=county_col, y=visits_col, 
            data=df_clean[df_clean[county_col].isin(top_counties)], 
            estimator='mean')
plt.title('Average Visits per County (Top 20)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Histogram: Distribution of Staff Expenditures
plt.figure(figsize=(8, 5))
sns.histplot(df_clean[staff_col], bins=30, kde=True)
plt.title('Distribution of Staff Expenditures')
plt.show()

# Pairplot: Among Staff Expenditures, Visits, Circulation
sns.pairplot(df_clean[[staff_col, visits_col, circ_col]])
plt.suptitle('Pairplot of Staff Expenditures, Visits, Circulation', y=1.02)
plt.show()

# Boxplot: Circulation across Counties (top 20)
plt.figure(figsize=(12, 6))
top_circ_counties = df_clean.groupby(county_col)[circ_col].mean().sort_values(ascending=False).head(20).index
sns.boxplot(x=county_col, y=circ_col, 
            data=df_clean[df_clean[county_col].isin(top_circ_counties)])
plt.title('Boxplot of Circulation by County (Top 20)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Heatmap: Correlation matrix
plt.figure(figsize=(8, 6))
corr = df_clean[[staff_col, visits_col, circ_col]].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Scatterplot: Visits vs Circulation
plt.figure(figsize=(8, 6))
sns.scatterplot(x=visits_col, y=circ_col, data=df_clean)
plt.title('Scatterplot of Visits vs Circulation')
plt.show()

# Z-Test: Compare Visits between two counties (change as needed)
county1 = 'Hartford'
county2 = 'New Haven'

visits_1 = df_clean[df_clean[county_col] == county1][visits_col]
visits_2 = df_clean[df_clean[county_col] == county2][visits_col]

# Make sure there's data
if len(visits_1) > 1 and len(visits_2) > 1:
    # Means, SDs, and sizes
    mean1, mean2 = visits_1.mean(), visits_2.mean()
    std1, std2 = visits_1.std(), visits_2.std()
    n1, n2 = len(visits_1), len(visits_2)

    # Z-test
    z = (mean1 - mean2) / np.sqrt(std1**2 / n1 + std2**2 / n2)
    p = 2 * (1 - norm.cdf(abs(z)))

    print(f"\nZ-Test: Comparing Visits between {county1} and {county2}")
    print(f"Mean {county1}: {mean1:.2f}, Mean {county2}: {mean2:.2f}")
    print(f"Z-score: {z:.2f}")
    print(f"P-value: {p:.4f}")
else:
    print(f"\nNot enough data for {county1} or {county2} to perform Z-test.")
