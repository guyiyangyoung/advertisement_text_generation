#!/usr/bin/env python3
"""
Test script to examine and validate the CSV data structure
"""

import pandas as pd
import json
import ast
from data_preprocessing import DataPreprocessor

def examine_csv_structure(csv_path: str):
    """Examine the structure of the CSV file"""
    print(f"Examining CSV file: {csv_path}")
    print("=" * 50)
    
    # Load the CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"✓ Columns: {list(df.columns)}")
        print()
        
        # Check if required columns exist
        required_columns = ['chapters_text', 'revise_asr']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return None
        else:
            print("✓ All required columns found")
        
        print("\nColumn Analysis:")
        print("-" * 30)
        
        # Analyze chapters_text column
        print("chapters_text column:")
        sample_chapters = df['chapters_text'].iloc[0]
        print(f"  Type: {type(sample_chapters)}")
        print(f"  Length: {len(str(sample_chapters))}")
        print(f"  First 200 characters: {str(sample_chapters)[:200]}...")
        
        # Try to parse the first chapters_text
        try:
            if isinstance(sample_chapters, str):
                try:
                    parsed = json.loads(sample_chapters)
                except json.JSONDecodeError:
                    parsed = ast.literal_eval(sample_chapters)
            else:
                parsed = sample_chapters
            
            if isinstance(parsed, list) and len(parsed) > 0:
                print(f"  Successfully parsed as list with {len(parsed)} chapters")
                first_chapter = parsed[0]
                if isinstance(first_chapter, dict):
                    print(f"  First chapter keys: {list(first_chapter.keys())}")
                    print(f"  First chapter title: {first_chapter.get('title', 'N/A')[:50]}...")
                    print(f"  First chapter content length: {len(first_chapter.get('content', ''))}")
                else:
                    print(f"  ❌ First chapter is not a dict: {type(first_chapter)}")
            else:
                print(f"  ❌ chapters_text is not a list or is empty")
        except Exception as e:
            print(f"  ❌ Error parsing chapters_text: {e}")
        
        print()
        
        # Analyze revise_asr column
        print("revise_asr column:")
        sample_asr = df['revise_asr'].iloc[0]
        print(f"  Type: {type(sample_asr)}")
        print(f"  Length: {len(str(sample_asr))}")
        print(f"  Content: {str(sample_asr)[:200]}...")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def test_preprocessing(csv_path: str):
    """Test the preprocessing pipeline"""
    print("\nTesting Preprocessing Pipeline:")
    print("=" * 50)
    
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor(csv_path)
        
        # Load data
        df = preprocessor.load_data()
        
        # Process one example
        first_row = df.iloc[0]
        print("Processing first row...")
        
        # Parse chapters
        chapters = preprocessor.parse_chapters_text(first_row['chapters_text'])
        print(f"✓ Parsed {len(chapters)} chapters")
        
        # Format chapters
        formatted_chapters = preprocessor.format_chapters_as_text(chapters)
        print(f"✓ Formatted chapters (length: {len(formatted_chapters)})")
        
        # Show formatted output
        print("\nFormatted Chapters Text (Input):")
        print("-" * 30)
        print(formatted_chapters[:500] + "..." if len(formatted_chapters) > 500 else formatted_chapters)
        
        print("\nTarget Output:")
        print("-" * 30)
        print(str(first_row['revise_asr'])[:300] + "..." if len(str(first_row['revise_asr'])) > 300 else str(first_row['revise_asr']))
        
        # Test full preprocessing
        print("\nRunning full preprocessing...")
        training_data = preprocessor.prepare_training_data()
        
        if training_data:
            print(f"✓ Successfully created {len(training_data)} training examples")
            
            # Show first training example
            first_example = training_data[0]
            print("\nFirst Training Example Structure:")
            print("-" * 30)
            for key, value in first_example.items():
                print(f"{key}: {len(str(value))} characters")
            
            return training_data
        else:
            print("❌ No training data created")
            return None
            
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    # Try different possible CSV file names
    possible_files = ['test.csv', 'clean_data.csv', 'data.csv']
    
    csv_file = None
    for filename in possible_files:
        try:
            import os
            if os.path.exists(filename):
                csv_file = filename
                break
        except:
            continue
    
    if csv_file is None:
        print("No CSV file found. Please ensure one of these files exists:")
        for filename in possible_files:
            print(f"  - {filename}")
        return
    
    print(f"Found CSV file: {csv_file}")
    print()
    
    # Examine structure
    df = examine_csv_structure(csv_file)
    
    if df is not None:
        # Test preprocessing
        training_data = test_preprocessing(csv_file)
        
        if training_data:
            print("\n" + "=" * 50)
            print("✅ PREPROCESSING TEST SUCCESSFUL!")
            print(f"✅ Ready to process {len(training_data)} training examples")
            print("✅ You can now run: python data_preprocessing.py")
            print("✅ Followed by: python qwen_finetune.py")
        else:
            print("\n" + "=" * 50)
            print("❌ PREPROCESSING TEST FAILED!")
            print("❌ Please check your data format")
    else:
        print("\n" + "=" * 50)
        print("❌ CSV EXAMINATION FAILED!")
        print("❌ Please check your CSV file format")

if __name__ == "__main__":
    main()