import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv('./labeled_comment_data.csv', header=0)

# Prepare data
data_list = []
for _, row in df.iterrows():
    temp_data = dict()
    temp_data['instruction'] = """Task: You will classify a comment from a Mozilla Firefox issue report discussion thread into one of four categories: Potential Solution Design, Solution Review, Code Implementation, or Non-Solution.
        Classification Criteria:
        Potential Solution Design: The comment proposes a solution to the issue or problem described, made by a developer or contributor.
        Solution Review: The comment assesses or critiques a proposed solution's feasibility, including feedback or suggestions made by other developers.
        Code Implementation: The comment describes or discusses the details of a solution that has been implemented, including code or specific fixes.
        Non-Solution: The comment does not fit any of the above categories. This includes automated messages (e.g., bot responses, links, attachments), general discussions, or comments that perform a code review (checking for problems in the patch/code of the implemented solution).
        Instructions:
        You will classify the provided comment using the above criteria and respond with one of the following categories:
        1 for Potential Solution Design
        2 for Solution Review
        3 for Code Implementation
        0 for Non-Solution
    """
    temp_data['input'] = f"""Issue report comment to classify (delimited by triple quotes):
        \"\"\"
        {row['text']}
        \"\"\"
    """
    if row['label'] == 1 and row['code'] == 'POTENTIAL_SOLUTION_DESIGN':
        temp_data["output"] = "1 for Potential Solution Design"
    elif row['label'] == 1 and row['code'] == 'SOLUTION_REVIEW':
        temp_data["output"] = "2 for Solution Review"
    elif row['label'] == 1 and row['code'] == 'CODE_IMPLEMENTATION':
        temp_data["output"] = "3 for Code Implementation"
    elif row['label'] == 0:
        temp_data["output"] = "0 for Non-Solution"
    else:
        continue
    
    data_list.append(temp_data)

# Set the train/test split ratio (e.g., 80% train, 20% test)
train_ratio = 0.8

# Split the data into training and testing sets
train_data, test_data = train_test_split(data_list, train_size=train_ratio, random_state=42)

# Save the training set to JSON
with open('train_4_classes.json', 'w') as train_json_file:
    json.dump(train_data, train_json_file, indent=4)

# Save the testing set to JSON
with open('test_4_classes.json', 'w') as test_json_file:
    json.dump(test_data, test_json_file, indent=4)
