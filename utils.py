import os
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode


def to_normal_string(s):
    # Replace underscores with spaces for snake_case
    s = s.replace('_', ' ')
    
    # Remove accents
    s = unidecode(s)

    # Insert spaces before capital letters in camelCase (avoiding the first character)
    return ''.join(' ' + char.lower() if char.isupper() and i != 0 else char for i, char in enumerate(s)).strip()


def process_split(dataset, file_path, split):
    e2e_data = []
    wtq_data = []
    wsql_data = []
    # Loop through each entry in the dataset
    for entry in tqdm(dataset):
        # Extract texts and triplets
        triplet_set = entry['tripleset']

        src = entry['target_sources'][0]
        text = 'Generate in English: ' + entry['target']
        # Format triplets
        triplet_text = 'Extract Triples: '
        for idx, triplet in enumerate(triplet_set):
            triplet_text += f'[S] {to_normal_string(triplet[0])} ' \
                            f'[P] {to_normal_string(triplet[1])} [O] {to_normal_string(triplet[2])} '

        if 'e2e' in src:
            e2e_data.append({'text': text, 'triplet': triplet_text.strip()})
        elif 'WikiTableQuestions' in src:
            wtq_data.append({'text': text, 'triplet': triplet_text.strip()})
        elif 'WikiSQL' in src:
            wsql_data.append({'text': text, 'triplet': triplet_text.strip()})

    # Convert to pandas DataFrame and save data as tsv for each source
    e2e_df = pd.DataFrame(e2e_data)
    e2e_df.to_csv(os.path.join(file_path, f'e2e_{split}.tsv'), sep='\t', index=False)
    wtq_df = pd.DataFrame(wtq_data)
    wtq_df.to_csv(os.path.join(file_path, f'wtq_{split}.tsv'), sep='\t', index=False)
    wsql_df = pd.DataFrame(wsql_data)
    wsql_df.to_csv(os.path.join(file_path, f'wsql_{split}.tsv'), sep='\t', index=False)
    
    # Sample subsets of 100 samples for low-resource training
    if split == 'train':
        seeds = [662, 66, 62, 6, 2]
        for idx, seed in enumerate(seeds):
            e2e_df.sample(n=100, random_state=seed).to_csv(os.path.join(file_path, f'e2e_{split}_sub{idx}.tsv'), sep='\t', index=False)
            wtq_df.sample(n=100, random_state=seed).to_csv(os.path.join(file_path, f'wtq_{split}_sub{idx}.tsv'), sep='\t', index=False)
            wsql_df.sample(n=100, random_state=seed).to_csv(os.path.join(file_path, f'wsql_{split}_sub{idx}.tsv'), sep='\t', index=False)
    
    print(f"{split} split: e2e: {e2e_df.shape[0]}, wtq: {wtq_df.shape[0]}, wsql: {wsql_df.shape[0]}")


def preprocess_dart():
    # Create path for data
    if not os.path.isdir('data'):
        os.mkdir('data')

    # Loop over splits
    splits = ['train', 'validation', 'test']
    for split in splits:
        # Load the DART dataset
        dataset = load_dataset("gem/dart", split=split)
        # Save the subsets for each split
        process_split(dataset, 'data', split)
    print("Datasets saved successfully.")


if __name__ == "__main__":
    preprocess_dart()
