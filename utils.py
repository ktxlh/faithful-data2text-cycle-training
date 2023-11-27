import os
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


def process_split(dataset, file_path, split):
    e2e_data = []
    wtq_data = []
    wsql_data = []
    e2e_count = 0
    wtq_count = 0
    wsql_count = 0
    # Loop through each entry in the dataset
    for entry in tqdm(dataset):
        # Extract annotations and tripleset
        annotations = entry['annotations']
        triplet_set = entry['tripleset']

        for src, text in zip(annotations['source'], annotations['text']):
            # Format triplets
            triplet_text = ''
            for idx, triplet in enumerate(triplet_set):
                triplet_text += f'{idx + 1}. [S] {triplet[0]} [P] {triplet[1]} [O] {triplet[2]} '

            # Add the text and corresponding triplets to the list
            if 'e2e' in src:
                e2e_count += 1
                e2e_data.append({'text': text, 'triplet': triplet_text})
            elif 'WikiTableQuestions' in src:
                wtq_count += 1
                wtq_data.append({'text': text, 'triplet': triplet_text})
            elif 'WikiSQL' in src:
                wsql_count += 1
                wsql_data.append({'text': text, 'triplet': triplet_text})

    # Convert to pandas DataFrame and save data as tsv for each source
    e2e_df = pd.DataFrame(e2e_data)
    e2e_df.to_csv(os.path.join(file_path, f'e2e_{split}.tsv'), sep='\t', index=False)
    wtq_df = pd.DataFrame(wtq_data)
    wtq_df.to_csv(os.path.join(file_path, f'wtq_{split}.tsv'), sep='\t', index=False)
    wsql_df = pd.DataFrame(wsql_data)
    wsql_df.to_csv(os.path.join(file_path, f'wsql_{split}.tsv'), sep='\t', index=False)
    print(f"{split} split: e2e: {e2e_count}, wtq: {wtq_count}, wsql: {wsql_count}")


def preprocess_dart():
    # Create path for data
    if not os.path.isdir('data'):
        os.mkdir('data')

    # Loop over splits
    splits = ['train', 'validation', 'test']
    for split in splits:
        # Load the DART dataset
        dataset = load_dataset("dart", split=split)
        # Save the subsets for each split
        process_split(dataset, 'data', split)
    print("Datasets saved successfully.")


if __name__ == "__main__":
    preprocess_dart()
