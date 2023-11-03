import json
from tqdm import tqdm
from time import time
from itertools import product

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import RobertaTokenizer, RobertaForSequenceClassification

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def try_compute_model(batch_size, max_output_length):
    start = time()
    try:
        # batch_size = 2
        # max_output_length = 4
        min_output_length = 3
        num_beams = 5
        length_penalty = 0
        no_repeat_ngram_size = 0

        # NOTE: There will be *2* models on GPU simultaneously (although only one is trained at a time)
        model_text2data = T5ForConditionalGeneration.from_pretrained("t5-base")
        model_text2data.config.task_specific_params["summarization"]={
            "early_stopping": True,
            "length_penalty": length_penalty,
            "max_length": max_output_length,
            "min_length": min_output_length,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "num_beams": num_beams,
            "prefix": ""
        }
        model_text2data.to(device)

        model_data2text = T5ForConditionalGeneration.from_pretrained("t5-base")
        model_data2text.config.task_specific_params["summarization"]={
            "early_stopping": True,
            "length_penalty": length_penalty,
            "max_length": max_output_length,
            "min_length": min_output_length,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "num_beams": num_beams,
            "prefix": ""
        }
        model_data2text.to(device)

        optimizer = torch.optim.AdamW(model_data2text.parameters(), lr=3e-4)
        loss_function = torch.nn.CrossEntropyLoss()

        load_duration = time() - start
        start = time()

        vocab_size = model_data2text.encoder.embed_tokens.weight.shape[0]
        input_ids = torch.randint(low=0, high=vocab_size-1, size=(batch_size, max_output_length)).to(device)

        with torch.no_grad():
            model_text2data.eval()
            output = model_text2data(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids, return_dict=True)

        input_ids = torch.randint(low=0, high=vocab_size-1, size=(batch_size, max_output_length)).to(device)

        model_data2text.train()
        output = model_data2text(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids, return_dict=True)

        output.loss.backward()
        optimizer.step()

        step_duration = time() - start

    except RuntimeError as e:
        load_duration = 0
        step_duration = 0
        print('error', batch_size, max_output_length)

    return (batch_size, max_output_length, load_duration, step_duration)


if __name__ == "__main__":
    options = torch.pow(torch.full((8,), 2), torch.arange(8) + 1).int().tolist()

    outputs = [try_compute_model(batch_size, max_output_length) for batch_size, max_output_length in tqdm(list(product(options, options)))]
    print(len(outputs))
    
    with open("t5-base.json", "w") as out_file:
        json.dump(outputs, out_file, indent=2)
