import os
import numpy as np
import torch
from termcolor import colored
import json
import random
from sklearn.metrics import f1_score  # Import f1_score

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface'

# Import LLaVA related modules from utils
from utils.run_llava import run_proxy
from llava.mm_utils import get_model_name_from_path

# Import prompts
from utils.prompts import qdc_prompt

# Import data utility functions
from utils.data_utils import get_item_data, DATASET_CONFIGS

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
random.seed(42)

# --- LLaVA Model Initialization (Global, done once) ---
model_path = "liuhaotian/llava-v1.5-13b"
model_name = get_model_name_from_path(model_path)
print(f"Loading LLaVA model: {model_name} from {model_path}...")
llava_proxy = run_proxy(model_path, None)
print("LLaVA model loaded.")

# Arguments object for LLaVA proxy
args = type('Args', (), {
    "query": None,
    "conv_mode": None,
    "image_file": None,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 1024,
})()


def get_model_res(prompt: str, image_path: str):
    """Calls the LLaVA model proxy to get a response."""
    args.image_file = image_path
    args.query = prompt
    _, response = llava_proxy.run_model(args)
    return response


def find_answer_position(answer: str):
    """Determines if the answer indicates 'harmful' or 'harmless'."""
    answer = answer.lower().strip()
    if "harmful" in answer:
        return 1
    elif "harmless" in answer:
        return 0
    return None


def process_qdc_evaluation(dataset_name: str, prompt):
    """
    Performs Iterative Analog Inference (QDC) evaluation for a given dataset.
    It uses rules derived from similar memes to assess target memes.
    """
    print(f"\n--- Starting QDC evaluation for dataset: {dataset_name} ---")

    base_data_path = f"data/{dataset_name}"
    image_base_path = f"{base_data_path}/images"
    test_jsonl_path = f"{base_data_path}/test.jsonl"
    ea_result_path = f"EA/{dataset_name}_EA.jsonl"
    qdc_result_path = f"results/{dataset_name}_layer1.jsonl"

    os.makedirs(os.path.dirname(qdc_result_path), exist_ok=True)

    try:
        test_data = [json.loads(line) for line in open(test_jsonl_path, 'r').readlines()]
        print(f"Loaded {len(test_data)} test items for {dataset_name}.")
    except FileNotFoundError:
        print(colored(f"Error: Test data file not found at {test_jsonl_path}. Skipping {dataset_name}.", 'red'))
        return
    except json.JSONDecodeError:
        print(colored(f"Error: Could not decode JSON from {test_jsonl_path}. Skipping {dataset_name}.", 'red'))
        return

    try:
        ea_lines = [json.loads(line) for line in open(ea_result_path, 'r').readlines()]
        print(f"Loaded {len(ea_lines)} EA results from {ea_result_path}.")
    except FileNotFoundError:
        print(colored(f"Error: EA result file not found at {ea_result_path}. Skipping {dataset_name}.", 'red'))
        return
    except json.JSONDecodeError:
        print(colored(f"Error: Could not decode JSON from {ea_result_path}. Skipping {dataset_name}.", 'red'))
        return

    ratio = [0, 0]  # [total_processed, correct_predictions]
    start_idx = 0

    # Lists to store actual and predicted labels for F1-score calculation
    all_actual_labels = []
    all_predicted_labels = []

    # Continuation logic for interrupted runs (matching the provided reference)
    if os.path.exists(qdc_result_path):
        with open(qdc_result_path, 'r') as f_read:
            file_lines = f_read.readlines()
            start_idx = len(file_lines)
            if start_idx > 0:
                try:
                    last_result = json.loads(file_lines[-1])
                    if 'ratio' in last_result and isinstance(last_result['ratio'], list) and len(
                            last_result['ratio']) == 2:
                        ratio = last_result['ratio']
                        if 'accuracy' in last_result:
                            print(colored(
                                f"Dataset {dataset_name} seems to be fully processed. Delete {qdc_result_path} to re-run.",
                                'green'))
                            return
                    else:
                        print(colored("Warning: 'ratio' key not found or malformed in last result, resetting ratio.",
                                      'yellow'))
                        ratio = [0, 0]
                except json.JSONDecodeError:
                    print(colored("Warning: Could not decode last line for ratio, resetting ratio.", 'yellow'))
                    ratio = [0, 0]

                # Populate all_actual_labels and all_predicted_labels from previous runs
                for i in range(start_idx):
                    try:
                        prev_item = json.loads(file_lines[i])
                        if 'actual' in prev_item and 'predict' in prev_item:
                            all_actual_labels.append(prev_item['actual'])
                            # Handle None predictions for previous runs as well
                            all_predicted_labels.append(
                                prev_item['predict'] if prev_item['predict'] is not None else 1 - prev_item['actual'])
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

                test_data = test_data[start_idx:]
                ea_lines = ea_lines[start_idx:]
                print(colored(f'Continuing QDC evaluation from index: {start_idx}', 'cyan'))
                print(f"Current ratio: {ratio}")

    with open(qdc_result_path, 'a') as f_write:
        for idx_offset, (item, ea_line) in enumerate(zip(test_data, ea_lines)):
            current_absolute_idx = start_idx + idx_offset

            image_file_name, text_content, label = get_item_data(item, dataset_name)
            if image_file_name is None or text_content is None or label is None:
                print(colored(f"Warning: Missing essential data for test item {current_absolute_idx}. Skipping.",
                              'yellow'))
                continue

            if item.get('index') is not None and ea_line.get('index') is not None and item['index'] != ea_line[
                'index']:
                print(colored(
                    f"Error: Mismatch in test_data index ({item['index']}) and EA result index ({ea_line['index']}) at absolute index {current_absolute_idx}. Skipping.",
                    'red'))
                continue

            image_file_path = os.path.join(image_base_path, image_file_name)
            forward_rules = ea_line['forward']
            backward_rules = ea_line['backward']
            random_rules = ea_line['random']

            input_debater1 = prompt.format(text_content, forward_rules)
            output_debater1 = get_model_res(input_debater1, image_file_path)
            print(colored("\n--- Debater 1 Output ---", 'green'))
            print(colored(output_debater1, 'green'))
            predict_1 = output_debater1.split("Answer: ")[-1].split('.')[0].lower().strip().strip('[').strip(']')
            thought_1 = output_debater1.split("Thought: ")[-1]

            input_debater2 = prompt.format(text_content, backward_rules)
            output_debater2 = get_model_res(input_debater2, image_file_path)
            print(colored("\n--- Debater 2 Output ---", 'yellow'))
            print(colored(output_debater2, 'yellow'))
            predict_2 = output_debater2.split("Answer: ")[-1].split('.')[0].lower().strip().strip('[').strip(']')
            thought_2 = output_debater2.split("Thought: ")[-1]

            input_debater3 = prompt.format(text_content, random_rules)
            output_debater3 = get_model_res(input_debater3, image_file_path)
            print(colored("\n--- Debater 3 Output ---", 'white'))
            print(colored(output_debater3, 'white'))
            predict_3 = output_debater3.split("Answer: ")[-1].split('.')[0].lower().strip().strip('[').strip(']')
            thought_3 = output_debater3.split("Thought: ")[-1]

            if predict_1 == predict_2 == predict_3 or predict_1 == predict_2 or predict_1 == predict_3:
                predict = predict_1
            else:
                predict = predict_2
            real_predict = predict

            final_predict_val = find_answer_position(real_predict)

            # If final_predict_val is None, set it to the opposite of the actual label
            # This is a common practice to handle cases where the model fails to output a clear classification
            if final_predict_val is None:
                final_predict_val = 1 - label

            # Update ratio
            if final_predict_val == label:
                ratio[1] += 1
            ratio[0] += 1

            # Append to lists for F1-score calculation
            all_actual_labels.append(label)
            all_predicted_labels.append(final_predict_val)

            result = {
                'index': current_absolute_idx,
                'ratio': list(ratio),
                'actual': label,
                'real_predict': final_predict_val,
                'predict': predict,
                'text': text_content,
            }

            result['debater1_predict'] = predict_1
            result['debater1_thought'] = thought_1
            result['debater2_predict'] = predict_2
            result['debater2_thought'] = thought_2
            result['debater3_predict'] = predict_3
            result['debater3_thought'] = thought_3

            print(f"Actual: {label}, Predict: {final_predict_val}, ratio: {ratio}")

            json.dump(result, f_write)
            f_write.write('\n')
            f_write.flush()

        # After loop, dump final ratio, accuracy, and macro-F1 score
        accuracy = ratio[1] / ratio[0] if ratio[0] > 0 else 0

        # Calculate Macro F1 score
        macro_f1 = 0
        if len(all_actual_labels) > 0:
            # We assume labels are 0 and 1, as per 'harmless' (0) and 'harmful' (1)
            macro_f1 = f1_score(all_actual_labels, all_predicted_labels, average='macro')

        final_summary = {'ratio': ratio, 'accuracy': accuracy, 'macro_f1': macro_f1}
        json.dump(final_summary, f_write)
        f_write.write('\n')

    print(f"\n--- Finished QDC evaluation for dataset: {dataset_name} ---")
    print(f"Final Accuracy for {dataset_name}: {accuracy:.4f} ({ratio[1]}/{ratio[0]})")
    print(f"Final Macro F1 Score for {dataset_name}: {macro_f1:.4f}")


if __name__ == "__main__":
    datasets_to_process = ["HarM", "MAMI", "FHM"]
    for i in range(0, 3):
        process_qdc_evaluation(datasets_to_process[i], qdc_prompt)

    print("\nAll datasets processed for QDC evaluation.")