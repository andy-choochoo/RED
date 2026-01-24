import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:-1]:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def calculate_metrics():
    # You need to change the document location.
    data_a = load_data('D:/Fighting/paper/result/RED/results/MAMI_layer1.jsonl')
    data_b = load_data('D:/Fighting/paper/result/RED/results/MAMI_agree_.jsonl')
    data_c = load_data('D:/Fighting/paper/result/RED/results/MAMI_standard_.jsonl')
    data_d = load_data('D:/Fighting/paper/result/RED/results/MAMI_testify_.jsonl')
    data_e = load_data('D:/Fighting/paper/result/RED/results/MAMI_agree.jsonl')
    data_f = load_data('D:/Fighting/paper/result/RED/results/MAMI_testify.jsonl')

    min_len = min(len(data_a), len(data_b), len(data_c), len(data_d))

    y_true = []
    y_pred = []

    for i in range(min_len):
        a_line = data_a[i]
        b_line = data_b[i]
        c_line = data_c[i]
        d_line = data_d[i]
        e_line = data_e[i]
        f_line = data_f[i]

        actual = a_line.get('actual')

        debater1 = a_line.get('debater1_predict')
        debater2 = a_line.get('debater2_predict')
        debater3 = a_line.get('debater3_predict')

        if debater1 == debater2 == debater3:
            prediction = a_line.get('real_predict')
        else:
            pred_c = c_line.get('real_predict')
            pred_d = d_line.get('real_predict')
            pred_b = b_line.get('real_predict')
            pred_e = e_line.get('real_predict')
            pred_f = f_line.get('real_predict')

            count_0 = 0
            count_1 = 0
            if pred_b == 0:
                count_0 += 1
            else:
                count_1 += 1
            if pred_c == 0:
                count_0 += 1
            else:
                count_1 += 1
            if pred_d == 0:
                count_0 += 1
            else:
                count_1 += 1
            if pred_e == 0:
                count_0 += 1
            else:
                count_1 += 1
            if pred_f == 0:
                count_0 += 1
            else:
                count_1 += 1
            prediction = 1 if count_1 >= 2 else 0

        y_true.append(actual)
        y_pred.append(prediction)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # micro_f1 = f1_score(y_true, y_pred, average='micro')
    #
    # binary_f1 = f1_score(y_true, y_pred, average='binary')

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print("=" * 50)
    print(f"Sample number: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    # print(f"Micro-F1: {micro_f1:.4f}")
    # print(f"Binary F1(focusing on harmful memes): {binary_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("=" * 50)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        # 'micro_f1': micro_f1,
        # 'binary_f1': binary_f1,
        'precision': precision,
        'recall': recall,
        'y_true': y_true,
        'y_pred': y_pred
    }

if __name__ == "__main__":
    results = calculate_metrics()