# find_ages.py - Find where the real ages are stored
import pandas as pd
import gzip
import os

print("="*60)
print("SEARCHING FOR REAL AGES IN GSE40279 FILES")
print("="*60)

# Check 1: Look in the series matrix file for metadata
print("\n📊 Checking series_matrix file for age metadata...")
matrix_file = "data/GSE40279_series_matrix.txt"

if os.path.exists(matrix_file):
    print("Reading metadata lines from series matrix...")
    with open(matrix_file, 'r') as f:
        for i, line in enumerate(f):
            if i > 100:  # Stop after checking header
                break
            # Look for age-related lines
            if 'age' in line.lower() or 'characteristic' in line.lower():
                print(f"Line {i}: {line[:200].strip()}...")
                
# Check 2: Look at average_beta file structure
print("\n📊 Checking average_beta file structure...")
beta_gz = "data/GSE40279_average_beta.txt.gz"

if os.path.exists(beta_gz):
    print("Reading first few lines of average_beta...")
    with gzip.open(beta_gz, 'rt') as f:
        for i, line in enumerate(f):
            if i < 5:  # Just check first few lines
                print(f"Line {i}: {line[:200].strip()}...")
            else:
                break

# Check 3: The ages might be in the GEO database - let's download them
print("\n📊 Alternative: Download age data using GEOparse...")
print("\nThe ages are likely in the GEO metadata.")
print("Let me create a script to extract them properly...")

# Create script to get ages from GEO
extract_script = '''
# get_geo_ages.py - Extract ages from GEO metadata
import GEOparse
import pandas as pd
import numpy as np

print("Downloading metadata from GEO (this includes ages)...")
print("This may take 1-2 minutes...")

# Download the full dataset metadata
gse = GEOparse.get_GEO(geo="GSE40279", destdir="./data/")

# Get sample metadata
samples = gse.phenotype_data

print(f"\\nSample metadata shape: {samples.shape}")
print(f"Columns: {samples.columns.tolist()[:10]}...")  # Show first 10 columns

# Look for age in different possible columns
age_found = False
for col in samples.columns:
    if 'age' in col.lower() or 'characteristic' in col.lower():
        print(f"\\nFound potential age column: {col}")
        print(f"Sample values: {samples[col].head()}")
        
        # Try to extract numeric ages
        if 'age' in col.lower():
            try:
                # Extract numbers from strings like "age: 34"
                ages = samples[col].str.extract(r'(\\d+\\.?\\d*)').astype(float)
                if not ages.isna().all():
                    age_found = True
                    print(f"Extracted {len(ages.dropna())} ages")
                    ages.to_csv('data/extracted_ages.csv')
                    print("Saved to: data/extracted_ages.csv")
            except:
                pass

if not age_found:
    print("\\n⚠️ Could not automatically extract ages")
    print("The age data might be in the paper's supplementary materials")
'''

with open('get_geo_ages.py', 'w') as f:
    f.write(extract_script)

print("\n✅ Created 'get_geo_ages.py'")
print("\nTo get the real ages, you need to:")
print("1. Install GEOparse: pip install GEOparse")
print("2. Run: python get_geo_ages.py")
print("\nAlternatively, the ages might be in the original paper.")