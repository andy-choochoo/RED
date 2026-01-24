import os
import torch
import clip
from PIL import Image
import json
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
import copy

from utils.data_utils import get_item_data, DATASET_CONFIGS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the CLIP model and preprocessor
model, preprocess = clip.load("ViT-L/14@336px", device=device)
model.eval()


def get_similarity_score(embedding, ref_embeddings):
    # calculate the cos similarity

    embeddings_np = np.array([embedding])
    ref_embeddings_np = np.array(ref_embeddings)

    dot_products = np.dot(embeddings_np, ref_embeddings_np.T)
    norms_embeddings = norm(embeddings_np, axis=1, keepdims=True)
    norms_ref_embeddings = norm(ref_embeddings_np, axis=1, keepdims=True).T

    norms_embeddings[norms_embeddings == 0] = 1
    norms_ref_embeddings[norms_ref_embeddings == 0] = 1

    similarity_scores = dot_products / (norms_embeddings * norms_ref_embeddings)
    similarity_scores = np.clip(similarity_scores, -1.0, 1.0)
    similarity_scores[similarity_scores >= 1.0] = 0.0

    return copy.deepcopy(similarity_scores)


def compute_embedding(item, image_base_path, dataset_name):
    # calculate the single meme's text embedding and image embedding

    image_file_name, text_content, _ = get_item_data(item, dataset_name)
    if not image_file_name or not text_content:
        return None, None

    image_file_path = os.path.join(image_base_path, image_file_name)
    try:
        image = Image.open(image_file_path).convert('RGB')
    except Exception as e:
        print(f"Warning: Could not open image {image_file_path}: {e}")
        return None, None

    processed_image = preprocess(image).unsqueeze(0).to(device)
    tokenized_text = clip.tokenize([text_content], truncate=True).to(device)

    with torch.no_grad():
        image_features = model.encode_image(processed_image).squeeze().detach().cpu().numpy()
        text_features = model.encode_text(tokenized_text).squeeze().detach().cpu().numpy()

    return text_features, image_features


def process_clip_embeddings(dataset_name: str, k: int = 10):
    # process CLIP embeddings，calculate similarity and save top-k similar samples

    print(f"\n--- Processing dataset: {dataset_name} ---")

    base_data_path = f"data/{dataset_name}"
    image_base_path = f"{base_data_path}/images"
    test_jsonl_path = f"{base_data_path}/test.jsonl"
    train_jsonl_path = f"{base_data_path}/train.jsonl"
    result_path = f"SSR/{dataset_name}_SSR.jsonl"

    try:
        with open(test_jsonl_path, 'r') as f:
            test_data = [json.loads(line) for line in f]
        with open(train_jsonl_path, 'r') as f:
            train_data = [json.loads(line) for line in f]
    except FileNotFoundError as e:
        print(f"Error: Data files not found for {dataset_name}. Missing file: {e.filename}")
        return

    print(f"Loaded {len(test_data)} test items and {len(train_data)} train items for {dataset_name}.")

    # calculate the test embedding
    test_text_embeddings = []
    test_image_embeddings = []
    for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Test Embeddings"):
        text_features, image_features = compute_embedding(item, image_base_path, dataset_name)
        if text_features is not None and image_features is not None:
            test_text_embeddings.append(text_features)
            test_image_embeddings.append(image_features)

    # calculate the train embedding
    train_text_embeddings = []
    train_image_embeddings = []
    for idx, item in tqdm(enumerate(train_data), total=len(train_data), desc="Train Embeddings"):
        text_features, image_features = compute_embedding(item, image_base_path, dataset_name)
        if text_features is not None and image_features is not None:
            train_text_embeddings.append(text_features)
            train_image_embeddings.append(image_features)

    # heuristic search for the relevant memes of each test meme
    final_results = []
    print(f"Start heuristic relevant memes search for {dataset_name}")

    for idx in tqdm(range(len(test_text_embeddings)), desc="Heuristic Search"):
        w_text = [0.5]
        w_image = [0.5]

        for i in range(1, 11):
            embedding = test_text_embeddings[idx] * w_text[i - 1] + test_image_embeddings[idx] * w_image[i - 1]

            ref_embeddings = [
                train_text_embeddings[j] * w_text[i - 1] + train_image_embeddings[j] * w_image[i - 1]
                for j in range(len(train_text_embeddings))
            ]

            similarity_scores = get_similarity_score(embedding, ref_embeddings)

            current_scores = similarity_scores[0].copy()
            samples = []
            for _ in range(k):
                if np.max(current_scores) <= 0:
                    break
                j = int(np.argmax(current_scores))
                samples.append(j)
                current_scores[j] = -1

            top_k_ref_text_embeddings = [train_text_embeddings[j] for j in samples]
            top_k_ref_image_embeddings = [train_image_embeddings[j] for j in samples]

            top_k_ref_text_similarity = get_similarity_score(test_text_embeddings[idx], top_k_ref_image_embeddings)
            top_k_ref_image_similarity = get_similarity_score(test_image_embeddings[idx], top_k_ref_text_embeddings)

            epsilon = 1e-8
            tmp_text = np.average(top_k_ref_text_similarity) / (np.std(top_k_ref_text_similarity) + epsilon)
            tmp_image = np.average(top_k_ref_image_similarity) / (np.std(top_k_ref_image_similarity) + epsilon)

            total = tmp_text + tmp_image
            if total > 0:
                w_text.append(tmp_text / total)
                w_image.append(tmp_image / total)
            else:
                w_text.append(0.5)
                w_image.append(0.5)

            if abs(w_text[i] - w_text[i - 1]) < 0.001 and abs(w_image[i] - w_image[i - 1]) < 0.001:
                break

        print(f"Test meme {idx}: w_text={w_text[-1]:.4f}, w_image={w_image[-1]:.4f}")

        w_t = w_text[-1]
        w_i = w_image[-1]

        embedding = test_text_embeddings[idx] * w_t + test_image_embeddings[idx] * w_i
        ref_embeddings = [
            train_text_embeddings[j] * w_t + train_image_embeddings[j] * w_i
            for j in range(len(train_text_embeddings))
        ]

        similarity_scores = get_similarity_score(embedding, ref_embeddings)

        current_scores = similarity_scores[0].copy()
        samples = []
        scores = []
        for _ in range(k):
            if np.max(current_scores) <= 0:
                break
            j = int(np.argmax(current_scores))
            samples.append(j)
            scores.append(float(current_scores[j]))
            current_scores[j] = -1

        final_results.append({
            "index": idx,
            "samples": samples,
            "scores": scores,
        })

    print(f"Saving the result for {dataset_name}...")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        for result_item in final_results:
            json.dump(result_item, f)
            f.write("\n")
    print(f"Results saved to {result_path}")
    print(f"--- Finished processing {dataset_name} ---")


if __name__ == "__main__":
    datasets_to_process = ["FHM", "HarM", "MAMI"]
    for dataset in datasets_to_process:
        process_clip_embeddings(dataset)
