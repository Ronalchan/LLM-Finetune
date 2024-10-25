import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv('./labeled_comment_data.csv', header=0)

# Prepare data
data_list = []
for _, row in df.iterrows():
    temp_data = dict()
    temp_data['instruction'] = """Task: You will classify a comment from a Mozilla Firefox issue report discussion thread into one of four categories: Solution Review, or Non-Solution Review.
        Classification Criteria:
        Solution Review: The comment assesses or critiques a proposed solution's feasibility, including feedback or suggestions made by other developers.
        Non-Solution Review: The comment does not fit Solution Review.
        Instructions:
        You will classify the provided comment using the above criteria and respond with one of the following categories:
        1 for Solution Review
        0 for Non-Solution Review
    """
    temp_data['input'] = f"""Issue report comment to classify (delimited by triple quotes):
        \"\"\"
        {row['text']}
        \"\"\"
    """
    if row['label'] == 1 and row['code'] == 'SOLUTION_REVIEW':
        temp_data["output"] = "1 for Solution Review"
    else:
        temp_data["output"] = "0 for Non-Solution Review"
    
    data_list.append(temp_data)

# Set the train/test split ratio (e.g., 80% train, 20% test)
train_ratio = 0.8

# Split the data into training and testing sets
train_data, test_data = train_test_split(data_list, train_size=train_ratio, random_state=42)

# Save the training set to JSON
with open('train_2_solution_review.json', 'w') as train_json_file:
    json.dump(train_data, train_json_file, indent=4)

# Save the testing set to JSON
with open('test_2_solution_review.json', 'w') as test_json_file:
    json.dump(test_data, test_json_file, indent=4)
