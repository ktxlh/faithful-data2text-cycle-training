import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from unidecode import unidecode

def to_normal_string(s):
    # Replace underscores with spaces for snake_case
    s = s.replace('_', ' ')
    
    # Remove accents
    s = unidecode(s)

    # Insert spaces before capital letters in camelCase (avoiding the first character)
    return ''.join(' ' + char.lower() if char.isupper() and i != 0 else char for i, char in enumerate(s)).strip()

def process_split_webnlg(dataset, file_path, split):
    data = []
    predicates = set()
    triples_per_sample = []
    # lexicalizations_per_sample = []
    tokens_per_sample = []
    all_tokens = []
    num_lex = 0

    # Loop through each entry in the dataset
    for entry in tqdm(dataset):
        # Extract relevant information
        lex_entries = entry['lex']['text']
        triples_sets = entry['modified_triple_sets']['mtriple_set']
        num_lex += len(lex_entries)

        for lex_entry, triples_set in zip(lex_entries, triples_sets):
            # Calculate statistics
            # print('\n'.join(triples_set))
            predicates.update([triple.split(' | ')[1] for triple in triples_set])
            triples_per_sample.append(len(triples_set))

            # Count lexicalizations
            # lexicalizations_per_sample.append(1)  # Each lex entry is considered a separate lexicalization

            # Tokenize and count tokens
            lex_entry = to_normal_string(lex_entry)
            tokens = lex_entry.split()
            all_tokens = all_tokens + tokens
            tokens_per_sample.append(len(tokens))

            triple_set = [to_normal_string(triple) for triple in triples_set]

            # Add the data to the list
            data.append({'text': lex_entry, 'triples': triple_set, 'tokens': tokens})
            # break

        # break

    # Convert to pandas DataFrame and save data as tsv
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(file_path, f'webnlg_{split}.tsv'), sep='\t', index=False)

    # Calculate statistics
    # total_lexicalizations = sum(lexicalizations_per_sample)
    # total_tokens = sum(tokens_per_sample)

    # unique_predicates = len(predicates)

    # Print statistics
    print(f"{split} split: size = {len(dataset)}, lexicalizations = {num_lex}")
          
    return triples_per_sample, tokens_per_sample, all_tokens, predicates


def preprocess_webnlg():
    # Create path for data
    if not os.path.isdir('data'):
        os.mkdir('data')

    triples_per_sample = []
    tokens_per_sample = []
    all_tokens = []
    predicates = set()

    # Loop over splits
    splits = ['train', 'dev', 'test']
    for split in splits:
        # Load the WebNLG dataset
        dataset = load_dataset("web_nlg", 'release_v3.0_en', split=split)
        # Save the dataset for each split
        triples_per_sample_split, tokens_per_sample_split, all_tokens_split, pred_split = process_split_webnlg(dataset, 'data', split)
        triples_per_sample = triples_per_sample + triples_per_sample_split
        tokens_per_sample = tokens_per_sample + tokens_per_sample_split
        all_tokens = all_tokens + all_tokens_split
        predicates.update(pred_split)
        # print(len(pred_split))

    # Stats
    print(f"unique predicates = {len(predicates)}, triples/sample: median = {median(triples_per_sample)}, "
          f"max = {max(triples_per_sample)}, vocabulary size = {len(set(all_tokens))}, "
          f"tokens/sample: median = {median(tokens_per_sample)}, max = {max(tokens_per_sample)}")

    print("WebNLG Datasets saved successfully.")

# Helper function to calculate median
def median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    if n % 2 == 0:
        return (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2
    else:
        return sorted_lst[n // 2]

# Uncomment the following line to run the preprocessing
preprocess_webnlg()
