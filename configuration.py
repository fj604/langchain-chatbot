from datetime import datetime
import pandas as pd
import json
import re
import io
from typing import Dict, List, Tuple, Optional, Union
from langchain_aws.chat_models import ChatBedrockConverse
from boto3 import client
from botocore.config import Config

# Import the normalize function from the external file
from normalize import normalize

def process_csv_and_generate_config(
    data: Union[str, pd.DataFrame, io.StringIO], 
    output_json_path: str = None, 
    output_csv_path: str = None
) -> Tuple[Dict, str, pd.DataFrame]:
    """
    Process CSV data using Claude 3.7 and generate a configuration JSON file.
    
    Args:
        data: Input data as DataFrame, CSV string, or file-like object
        output_json_path: Path to save the output JSON configuration (optional)
        output_csv_path: Path to save the normalized CSV data (optional)
        
    Returns:
        Tuple containing:
        - Dict: The generated configuration JSON
        - str: The normalized CSV as a string
        - DataFrame: The processed CSV data as a DataFrame
    """
    # Step 1: Process the CSV using imported normalize function
    if isinstance(data, pd.DataFrame):
        # If data is already a DataFrame
        input_data = data
    elif isinstance(data, str):
        # If data is a file path or CSV string
        try:
            # Try to read as file path
            input_data = pd.read_csv(data)
        except:
            # If not a valid path, try as CSV string
            input_data = pd.read_csv(io.StringIO(data))
    elif isinstance(data, io.StringIO):
        # If data is a StringIO object
        input_data = pd.read_csv(data)
    else:
        raise ValueError("Input data must be a DataFrame, file path, or CSV string")
    
    # Call normalize function
    normalized_csv, error = normalize(input_data, input_data.columns)
    
    if error:
        print(f"Error during normalization: {error}")
        return None, None, None
    
    # Convert the normalized CSV string back to DataFrame
    csv_processed = pd.read_csv(io.StringIO(normalized_csv))
    print("CSV successfully processed.")
    
    # Step 2: Generate JSON configuration based on CSV columns
    config = generate_json_config(csv_processed)
    
    # Step 3: Save the configuration to a file if path is provided
    if output_json_path:
        with open(output_json_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {output_json_path}")
    
    # Step 4: Save the normalized CSV to a file if path is provided
    if output_csv_path:
        with open(output_csv_path, 'w') as f:
            f.write(normalized_csv)
        print(f"Normalized CSV saved to {output_csv_path}")
    
    return config, normalized_csv, csv_processed

def generate_json_config(csv_processed: pd.DataFrame) -> Dict:
    """
    Generate JSON configuration based on processed CSV columns
    
    Args:
        csv_processed: DataFrame with processed column names (including type annotations)
        
    Returns:
        Dict: The generated configuration
    """
    # Initialize the configuration template
    config = {
        "version": datetime.now().strftime("%Y-%m-%d"),  # Dynamic version based on current date
        "designations": {
            "blq": "BLQ",
            "missing": "NS",
            "concentration_unit": "ng/mL"
        },
        "columns": {
            "time": None,
            "unique_id": None,
            "concentration": {
                "observed": None,
                "analyzed": None
            },
            "sorting": [],
            "grouping": []
        },
        "dosing": {
            "amount": 10000,  # Default value, may be updated
            "unit": "mg",     # Default value, may be updated
            "frequency": "missing",
            "type": "missing"
        },
        "time": {
            "unit": "hr",     # Default value, may be updated
            "time_of_administration": 0,
            "tau": 24,
            "infusion": {
                "protocol": "rate",
                "amount": "1"
            }
        },
        "kel_rules": {
            "regression_statistic": "adj_r2",
            "minimum_statistic": 0,
            "maximum_extrapolation_linear": 0,
            "maximum_extrapolation_logarithmic": 0,
            "maximum_time_points": 0,
            "minimum_span": 0,
            "earliest_time_point": 0,
            "tie": [
                "kel_n",
                "kel_lower"
            ]
        }
    }
    
    # Extract column names and their types from the processed CSV
    column_types = {}
    original_columns = {}  # Store original column names without type annotations
    time_unit = "hr"  # Default time unit
    conc_unit = "ng/mL"  # Default concentration unit
    amount_unit = "mg"  # Default dose unit
    
    for col in csv_processed.columns:
        # Extract column type using regex
        match = re.search(r'\[(.*?)\]', col)
        if match:
            col_type = match.group(1)
            # Get column name without the type annotation
            col_name = col.split(' [')[0].strip()
            column_types[col_name] = col_type
            original_columns[col_name] = col_name  # Store original name without type annotation
            
            # Extract units if available
            unit_match = re.search(r'\((.*?)\)', col_name)
            if unit_match:
                unit = unit_match.group(1)
                if col_type == "TIME":
                    time_unit = unit
                elif col_type in ["OBSERVED CONC", "ANALYZED CONC"]:
                    conc_unit = unit
                elif col_type == "AMOUNT":
                    amount_unit = unit
    
    # Map columns to configuration
    for col_name, col_type in column_types.items():
        # Use the original column name without type annotation
        original_name = original_columns[col_name]
        
        if col_type == "TIME":
            config["columns"]["time"] = original_name
            config["columns"]["sorting"].append(original_name)
            
        elif col_type == "UNIQUE ID":
            config["columns"]["unique_id"] = original_name
            config["columns"]["sorting"].insert(0, original_name)
            config["columns"]["grouping"].append(original_name)
            
        elif col_type == "OBSERVED CONC":
            config["columns"]["concentration"]["observed"] = original_name
            
        elif col_type == "ANALYZED CONC":
            config["columns"]["concentration"]["analyzed"] = original_name
            
        elif col_type == "AMOUNT":
            # Try to extract amount value if it's a constant
            full_col_name = [c for c in csv_processed.columns if original_name in c][0]
            try:
                if csv_processed[full_col_name].nunique() == 1:
                    config["dosing"]["amount"] = float(csv_processed[full_col_name].iloc[0])
            except:
                pass
            
        elif col_type == "GROUP":
            if original_name not in config["columns"]["grouping"]:
                config["columns"]["grouping"].append(original_name)
    
    # Update units in the configuration
    config["designations"]["concentration_unit"] = conc_unit
    config["dosing"]["unit"] = amount_unit
    config["time"]["unit"] = time_unit
    
    return config

def extract_column_info_using_claude(csv_processed: str) -> Dict:
    """
    Use Claude 3.7 to extract column information from the processed CSV
    
    Args:
        csv_processed: The processed CSV data as a string
        
    Returns:
        Dict: Information about column types, units, etc.
    """
    # Initialize the Bedrock Converse model
    config = Config(read_timeout=1800)
    model = ChatBedrockConverse(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        region_name="us-east-1",
        temperature=0,
        top_p=0.999,
        config=config,
        max_tokens=40000,
    )
    
    system_prompt = """
    # Column Information Extraction

    Analyze the header row of the provided CSV data. The column headers are in a special format that includes type annotations in square brackets (e.g., "Time [TIME]").

    ## Task
    Extract and categorize all columns based on their type annotations. Return a JSON object with the following structure:

    ```json
    {
      "columns": {
        "time": {"name": "column_name", "unit": "unit_if_present"},
        "unique_id": {"name": "column_name"},
        "observed_conc": {"name": "column_name", "unit": "unit_if_present"},
        "analyzed_conc": {"name": "column_name", "unit": "unit_if_present"},
        "amount": {"name": "column_name", "unit": "unit_if_present"},
        "group": ["group_column_names"],
        "covariate": ["covariate_column_names"],
        "other": ["other_column_names"]
      }
    }
    ```

    ## Column Type Mapping:
    - [TIME] → time
    - [UNIQUE ID] → unique_id
    - [OBSERVED CONC] → observed_conc
    - [ANALYZED CONC] → analyzed_conc
    - [AMOUNT] → amount
    - [GROUP] → group
    - [COVARIATE] → covariate
    - Any other type → other

    ## Unit Extraction
    If a column name contains a unit in parentheses, extract it. For example:
    - "Time(hr) [TIME]" → unit is "hr"
    - "Conc(ng/mL) [ANALYZED CONC]" → unit is "ng/mL"

    ## Important
    - Return ONLY the JSON object, nothing else.
    - Use lowercase for column names in the output.
    - If a column type is not present in the CSV, include it in the JSON but set its value to null.
    - For group, covariate, and other categories, always return an array, even if empty.
    """
    
    try:
        response = model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "human", "content": csv_processed}
        ])
        
        content = response.content
        column_info = json.loads(content)
        return column_info
    except Exception as e:
        print(f"Error extracting column info: {e}")
        return None

# Example usage for standalone execution
if __name__ == "__main__":
    # Replace with your actual file paths
    input_csv_path = "IVdata.csv"
    output_json_path = "config.json"
    output_csv_path = "processed_data.csv"
    
    # Process CSV and generate JSON configuration and normalized CSV
    config, normalized_csv, csv_processed_df = process_csv_and_generate_config(
        data=input_csv_path,
        output_json_path=output_json_path,
        output_csv_path=output_csv_path
    )
    print("Processing completed successfully.")

# Example integration with Streamlit app (for reference)
"""
# In your answer.py file, you can integrate with this module as follows:

from helper.configuration import process_csv_and_generate_config
import streamlit as st
import pandas as pd
import io

# Assuming you have uploaded_file from st.file_uploader
if uploaded_file is not None:
    # Process the uploaded file
    config_json, normalized_csv_str, normalized_df = process_csv_and_generate_config(
        data=uploaded_file
    )
    
    # Store results in session state
    st.session_state.config_json = config_json
    st.session_state.normalized_csv = normalized_csv_str
    st.session_state.normalized_data = normalized_df
    
    # You can also save the files if needed
    # with open("config.json", "w") as f:
    #     json.dump(config_json, f, indent=2)
"""