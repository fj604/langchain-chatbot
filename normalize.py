from langchain_aws.chat_models import ChatBedrockConverse
import pandas as pd


from boto3 import client
from botocore.config import Config
config = Config(read_timeout=1800)


# Function to normalize data

def normalize(data, columns):
   """
   Use Amazon Bedrock to normalize data to required columns
   """

  # Claude Sonnet 3.5: us.anthropic.claude-3-5-sonnet-20241022-v2:0
  # Claude Sonnet 3.7: us.anthropic.claude-3-7-sonnet-20250219-v1:0
  # Initialize the Bedrock Converse model
   model = ChatBedrockConverse(
       model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
       region_name="us-east-1",
       temperature=0,
       top_p=0.999,
       config=config,
       max_tokens=40000,
   )


   # Convert the data to CSV format for processing
   csv_data = data.to_csv(index=False)

   # Create system prompt for Amazon Bedrock
   system_prompt = f"""

# PK Dataset Processing Instructions

## 1. Header Standardization
- Consolidate multiple header lines into a single header line using underscore delimiter
- Example: "Parameter" and "Unit" rows become "Parameter_Unit"

## 2. Column Naming Format
- If column headers contain units:
  * Format as: Name(unit) [TYPE]
  * Example: "Time (hr)" → "Time(hr) [TIME]"
- If column headers don't contain units:
  * Format as: Name [TYPE]
  * Example: "Time" → "Time [TIME]"

## 3. Subject Identifier Processing
- Check if subjects have multiple treatments or doses
- If yes:
  * Create "Unique_ID [UNIQUE ID]" by combining Subject ID and Treatment (e.g., "1_A")
  * Place this column first in dataset
  * Preserve original Subject ID as "Subject [ID]"
- If no:
  * Label original Subject column as "Subject [UNIQUE ID]"

## 4. Time Column Processing
- For decimal time representing elapsed time from dose:
  * Label as [TIME]
- For non-decimal time formats (e.g., hh:mm, date-time):
  * Label original as [ORI TIME]
  * Create new "Decimal_Time [TIME]" column
  * Calculate elapsed hours from dose administration for each unique subject

## 5. Concentration Column Processing
- For concentration columns containing any non-numeric values:
  * Label as [OBSERVED CONC]
  * If no numeric-only concentration column exists:
    - Create new "Analyzed_Conc [ANALYZED CONC]" column with:
      > BLQ/<LLOQ values → 0
      > Missing values (NS, Missing) → empty cells
      > Numeric values preserved
- For concentration columns containing only numeric values:
  * Label as [ANALYZED CONC]
  * If there is no concentration column containing any non-numeric values, create a duplicate column labeled as [OBSERVED CONC] using the same values

## 6. Column Type Mapping
Map columns based on content:
- Subject/Patient identifiers → [UNIQUE ID] or [ID] (see rule 3)
- Treatment/Formulation/Drug → [GROUP]
- Administration route → [ROUTE]
- Demographics (Age, Weight, Sex) → [COVARIATE]
- Dose amounts → [AMOUNT]
- Concentration measurements → [OBSERVED CONC] or [ANALYZED CONC] (see rule 5)
- Time → [TIME] or [ORI TIME] (see rule 4)
- Any other column not specified above → [CUSTOM COLUMN]

## 7. Missing Value Handling
- Standardize missing values as empty cells

## 8. Column Reordering
Rearrange columns in the following order:
1. [UNIQUE ID] columns first
2. [TIME] columns next
3. [AMOUNT] columns next
4. [ANALYZED CONC] columns next
5. [OBSERVED CONC] columns next
6. All other columns maintaining their original order

## 9. Output Format
- Return only the processed CSV data without any additional text
- If an error occurs, return only: "Error: [error message]"

## Examples
- "Time" → "Time [TIME]"
- "Cp (ng/mL)" with any non-numeric values → "Cp(ng/mL) [OBSERVED CONC]" + new "Analyzed_Conc [ANALYZED CONC]"
- "Concentration" with numeric-only values → "Concentration [ANALYZED CONC]" + new "Concentration_Observed [OBSERVED CONC]" (only if no other [OBSERVED CONC] column exists)

"""




   # Generate response
   try:
       response = model.invoke([
           {"role": "system", "content": system_prompt},
           {"role": "human", "content": csv_data}
       ])
       print(response)
       content = response.content


       # Process response
       if content.startswith("Error:"):
           error = content
           normalized_data = None
       else:
           error = None
           normalized_data = content
   except Exception as e:
       return None, f"Error during normalization: {e}"


   return normalized_data, error
