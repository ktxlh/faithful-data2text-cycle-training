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

def process_split_dart(dataset, file_path, split, source):
    data = []
    triples_per_sample = []
    tokens_per_sample = []
    all_tokens = []
    predicates = set()
    num_samples = 0
    num_lex = 0

    # Loop through each entry in the dataset
    for entry in tqdm(dataset):
        # Extract relevant information
        annotations = entry['annotations']
        triplet_set = entry['tripleset']
        source_type = annotations['source']
        

        # Check if the source matches the specified one (e.g., WTQ, WSQL, E2E)
        if source in source_type:
            num_samples += 1
            lex_entries = annotations['text']
            num_lex += len(lex_entries)

            for lex_entry, triples_set in zip(lex_entries, triplet_set):
                # Calculate statistics
                predicates.update([triple[1] for triple in triplet_set])
                triples_per_sample.append(len(triplet_set))

                # Tokenize and count tokens
                lex_entry = to_normal_string(annotations['text'][0])
                tokens = lex_entry.split()
                all_tokens = all_tokens + tokens
                tokens_per_sample.append(len(tokens))

                triple_set = [to_normal_string(triple[1]) for triple in triplet_set]

                # Add the data to the list
                data.append({'text': lex_entry, 'triples': triple_set, 'tokens': tokens})

    # Convert to pandas DataFrame and save data as tsv
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(file_path, f'{source.lower()}_{split}.tsv'), sep='\t', index=False)

    # Print statistics
    print(f"{split} split ({source}): size = {num_samples}, lexicalizations = {num_lex}")

    return triples_per_sample, tokens_per_sample, all_tokens, predicates

def preprocess_dart():
    # Create path for data
    if not os.path.isdir('data'):
        os.mkdir('data')

    # Loop over splits and sources
    splits = ['train', 'validation', 'test']
    sources = ['WikiTableQuestions_mturk', 'WikiSQL_decl_sents', 'e2e']

    for source in sources:
        print("----------------------------------------------------")
        print("-----------------"+source.upper()+"-----------------")
            
        triples_per_sample = []
        tokens_per_sample = []
        all_tokens = []
        predicates = set()
        for split in splits:
            print("----------------------------------------------------")
            print("-----------------"+split.upper()+"-----------------")
            
            # Load the DART dataset
            dataset = load_dataset("dart", split=split)
            # Save the dataset for each split and source
            triples_per_sample_split, tokens_per_sample_split, all_tokens_split, pred_split = process_split_dart(
                dataset, 'data', split, source)
            triples_per_sample = triples_per_sample + triples_per_sample_split
            tokens_per_sample = tokens_per_sample + tokens_per_sample_split
            all_tokens = all_tokens + all_tokens_split
            predicates.update(pred_split)

        # Stats
        print(f"unique predicates = {len(predicates)}, "
            f"triples/sample: median = {median(triples_per_sample)}, "
            f"max = {max(triples_per_sample)}, vocabulary size = {len(set(all_tokens))}, "
            f"tokens/sample: median = {median(tokens_per_sample)}, max = {max(tokens_per_sample)}")

    print("DART Datasets saved successfully.")

# Helper function to calculate median
def median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    if n % 2 == 0:
        return (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2
    else:
        return sorted_lst[n // 2]

# Uncomment the following line to run the preprocessing
preprocess_dart()
