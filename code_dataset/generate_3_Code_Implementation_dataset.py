import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv('./labeled_comment_data.csv', header=0)

# Prepare data
data_list = []
for _, row in df.iterrows():
    temp_data = dict()
    temp_data['instruction'] = """Task: You will classify a comment from a Mozilla Firefox issue report discussion thread into one of four categories: Code Implementation, or Non-Code Implementation.
        Classification Criteria:
        Code Implementation: The comment describes or discusses the details of a solution that has been implemented, including code or specific fixes.
        Non-Code Implementation: The comment does not fit Code Implementation.
        Instructions:
        You will classify the provided comment using the above criteria and respond with one of the following categories:
        1 for Code Implementation
        0 for Non-Code Implementation
    """
    temp_data['input'] = f"""Issue report comment to classify (delimited by triple quotes):
        \"\"\"
        {row['text']}
        \"\"\"
    """
    if row['label'] == 1 and row['code'] == 'CODE_IMPLEMENTATION':
        temp_data["output"] = "1 for Code Implementation"
    else:
        temp_data["output"] = "0 for Non-Code Implementation"
    
    data_list.append(temp_data)

# Set the train/test split ratio (e.g., 80% train, 20% test)
train_ratio = 0.8

# Split the data into training and testing sets
train_data, test_data = train_test_split(data_list, train_size=train_ratio, random_state=42)

# Save the training set to JSON
with open('train_3_code_implementation.json', 'w') as train_json_file:
    json.dump(train_data, train_json_file, indent=4)

# Save the testing set to JSON
with open('test_3_code_implementation.json', 'w') as test_json_file:
    json.dump(test_data, test_json_file, indent=4)
