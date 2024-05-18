import pandas as pd

def load_data_from_excel(file_path, sheet_name=0):
    # Read the Excel file
    data = pd.read_csv(file_path)
    # Assuming the column names are 'Generated_Text' and 'Gold_Answer'
    predictions = data['generated_answer'].astype(str).tolist()
    gold_answers = data['expected_answer'].astype(str).tolist()
    return predictions, gold_answers

def calculate_accuracy(predictions, gold_answers):
    # Calculate accuracy as the percentage of exact matches
    matches = sum(1 for pred, gold in zip(predictions, gold_answers) if pred.lower() == gold.lower())
    accuracy = matches / len(predictions) * 100  # Convert to percentage
    return accuracy

# Define the path to your Excel file
file_path = '/Users/samvegshah/Downloads/Baseline_OriginalImage_Text_Prompts_PreTrainedLLM.csv'  # Update with your actual file path

# Load predictions and gold answers
predictions, gold_answers = load_data_from_excel(file_path)

# Calculate accuracy
accuracy = calculate_accuracy(predictions, gold_answers)

# Print the accuracy
print(f"Accuracy: {accuracy:.2f}%")
