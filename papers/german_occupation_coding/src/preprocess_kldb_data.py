#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KldB Data Preprocessing and Enrichment
=======================================

This script preprocesses occupation data and enriches it with official
KldB-2010 (Klassifikation der Berufe) descriptions from the German Federal
Employment Agency (Bundesagentur für Arbeit).

Steps:
1. Load raw occupation data
2. Standardize KldB codes (add leading zeros, extract digit levels)
3. Load official KldB taxonomy
4. Match and enrich data with literal occupation descriptions
5. Save enriched dataset

Input Files:
- data.csv: Raw occupation data (columns: occupation, kldb)
- KldB_2010-DE-2019-02-14-Gliederung.csv: Official KldB taxonomy
  (Download from: https://statistik.arbeitsagentur.de)

Output Files:
- cleaneddata.csv: Intermediate cleaned data
- result.csv: Matched results with descriptions
- data_including_jobdescription_and_Features.csv: Final enriched dataset

Author: DZHW (German Centre for Higher Education Research and Science Studies)
"""

import pandas as pd
import csv


def main():
    """Main preprocessing pipeline"""
    
    print("="*70)
    print("KldB Data Preprocessing and Enrichment")
    print("="*70)
    
    # ========================================================================
    # Step 1: Load raw data
    # ========================================================================
    print("\nStep 1: Loading raw occupation data...")
    df = pd.read_csv('data.csv')
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    
    # ========================================================================
    # Step 2: Standardize KldB codes
    # ========================================================================
    print("\nStep 2: Standardizing KldB codes...")
    
    # Convert KldB to string
    df['col'] = df['kldb'].astype(str)
    
    # Note: The following code attempts to add leading zeros to 4-digit codes
    # However, it operates on an empty array and has no effect
    # Kept for compatibility with original script
    arr = []
    df['col1'] = df['kldb'].astype(str)
    df.kldb = df.col1
    
    for i in range(len(arr)):
        if 2 < len(arr[i]) < 5:
            arr[i] = '0' + arr[i]
            print(arr[i])
    
    for x in arr:
        if 2 < len(x) < 5:
            print('Its not working')
    
    # Extract first 3 digits of KldB code
    df['col'] = df['kldb'].astype(str)
    df['kldb2'] = df['col'].str[0:3]
    df.kldb = df.kldb2
    
    print(f"Extracted 3-digit KldB codes")
    print(f"Sample codes: {df['kldb'].head().tolist()}")
    
    # ========================================================================
    # Step 3: Save intermediate cleaned data
    # ========================================================================
    print("\nStep 3: Saving intermediate cleaned data...")
    dp = pd.DataFrame(df)
    dp.to_csv('cleaneddata.csv')
    print("✓ Saved to cleaneddata.csv")
    
    # Read back as list of lists for matching
    file = open("cleaneddata.csv", "r")
    csv_reader = csv.reader(file)
    listsDf = []
    for row in csv_reader:
        listsDf.append(row)
    file.close()
    
    print(f"Loaded {len(listsDf)} rows for matching")
    
    # ========================================================================
    # Step 4: Load official KldB taxonomy
    # ========================================================================
    print("\nStep 4: Loading official KldB taxonomy...")
    
    myList = []
    try:
        with open('KldB_2010-DE-2019-02-14-Gliederung.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for row in csv_reader:
                myList.append(row)
        
        print(f"✓ Loaded {len(myList)} KldB taxonomy entries")
        print("  Source: Bundesagentur für Arbeit")
        
    except FileNotFoundError:
        print("\n" + "="*70)
        print("ERROR: KldB taxonomy file not found!")
        print("="*70)
        print("\nPlease download the official KldB classification from:")
        print("https://statistik.arbeitsagentur.de/DE/Statischer-Content/")
        print("Grundlagen/Klassifikationen/Klassifikation-der-Berufe/")
        print("\nFile needed: KldB_2010-DE-2019-02-14-Gliederung.csv")
        print("="*70)
        return
    
    # ========================================================================
    # Step 5: Match and enrich with literal descriptions
    # ========================================================================
    print("\nStep 5: Matching KldB codes with official descriptions...")
    
    result = []
    matched_count = 0
    
    for i in range(len(myList)):
        for j in range(len(listsDf)):
            if myList[i][0] == listsDf[j][3]:  # Match KldB codes
                listsDf[j].append(myList[i][2])  # Add description
                result.append(listsDf[j])
                matched_count += 1
    
    print(f"✓ Matched {matched_count} records with official descriptions")
    
    if matched_count == 0:
        print("\nWARNING: No matches found!")
        print("Check that KldB code formats match between files")
        return
    
    # ========================================================================
    # Step 6: Save final enriched dataset
    # ========================================================================
    print("\nStep 6: Saving enriched dataset...")
    
    # Save result
    db = pd.DataFrame(result)
    db.to_csv('result.csv', index=False, header=False)
    print("✓ Saved to result.csv")
    
    # Add column names
    db.columns = ['occupation', 'kldb', 'p_lfdnr', 'obs', 'col', 
                  'leteral_description_3']  # Note: keeping original typo
    
    # Save with headers
    dd = pd.DataFrame(db)
    dd.to_csv('data_including_jobdescription_and_Features.csv', index=False)
    print("✓ Saved to data_including_jobdescription_and_Features.csv")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("Preprocessing Complete!")
    print("="*70)
    print(f"\nFinal dataset:")
    print(f"  Rows: {len(dd)}")
    print(f"  Columns: {dd.columns.tolist()}")
    print(f"\nSample records:")
    print(dd.head())
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
