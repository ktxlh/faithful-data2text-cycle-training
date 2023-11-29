import os
import numpy as np
from tqdm import tqdm
import json

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_metric
from nltk import word_tokenize

from finetune import get_dataloader
from torch.nn import CrossEntropyLoss

def evaluate(test_dataloader, source, sub):
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model.load_state_dict(torch.load(f"checkpoints/{source}_{sub}.pt"))
    model.to(device)
    model.eval()

    total_loss = 0.0
    generated_texts, source_triplets, target_texts = [], [], []
    progress_bar = tqdm(range(len(test_dataloader)))
    for batch in test_dataloader:
        source_triplets.extend(batch.pop("triplet"))
        target_texts.extend(batch.pop("text"))

        inputs = {k: torch.stack(batch[k], dim=1).to(device) for k in ["input_ids", "attention_mask", "labels"]}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='mean')
        loss = loss_fct(logits.view(-1, logits.size(-1)), inputs['labels'].view(-1))
        total_loss += loss.item()
        progress_bar.set_description("Test  Batch Loss: %f" % (loss.item()))
        progress_bar.update(1)

        del inputs['labels']
        with torch.no_grad():
            generated_outputs = model.generate(**inputs, min_length=3, max_length=256, num_beams=4, early_stopping=True, length_penalty=1.0)
        decoded_outputs = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        generated_texts += decoded_outputs

        # for t, g in zip(batch["triplet"], decoded_outputs):
        #     print(t)
        #     print(g)
        #     print()
        #     break ###

    return total_loss, source_triplets, generated_texts, target_texts


def update_results(total_loss, source_triplets, generated_texts, target_texts):
    resulting_metrics = {'test_loss': total_loss / len(test_dataloader)}
    meteor_p = []
    meteor_g = []
    bleu_p = []
    bleu_g = []
    bertscore_p = []
    bertscore_g = []
    rouge_p = []
    rouge_g = []
    last_s = ""
    for s, p, g in zip(source_triplets, generated_texts, target_texts):
        s = s.strip()
        p = p.strip()
        g = g.replace("<s>", "").strip()

        if last_s == s:
            g_tokens = word_tokenize(g)
            meteor_g[-1].append(' '.join(g_tokens))
            bleu_g[-1].append(g_tokens)
            bertscore_g[-1].append(g)
            rouge_g[-1].append(g)
        else:
            p_tokens = word_tokenize(p)
            g_tokens = word_tokenize(g)
            meteor_p.append(' '.join(p_tokens))
            meteor_g.append([' '.join(g_tokens)])
            bleu_p.append(p_tokens)
            bleu_g.append([g_tokens])
            bertscore_p.append(p)
            bertscore_g.append([g])
            rouge_p.append(p)
            rouge_g.append([g])
        last_s = s
    resulting_metrics['meteor'] = metric_meteor.compute(predictions=meteor_p, references=meteor_g)['meteor']
    resulting_metrics['bleu'] = metric_bleu.compute(predictions=bleu_p, references=bleu_g)['bleu']
    resulting_metrics['bertscore'] = np.mean(
        metric_bertscore.compute(predictions=bertscore_p, references=bertscore_g, lang='en')['f1'])
    rouge_result = metric_rouge.compute(predictions=rouge_p, references=rouge_g)
    resulting_metrics['rouge1'] = rouge_result['rouge1'].mid.fmeasure
    resulting_metrics['rouge2'] = rouge_result['rouge2'].mid.fmeasure
    resulting_metrics['rougeL'] = rouge_result['rougeL'].mid.fmeasure


    result_path = f"results/{source}_test.json"
    if os.path.isfile(result_path):
        results = json.load(open(result_path, "r"))
    else:
        results = {k: [None]*5 for k in resulting_metrics.keys()}
    for k, v in resulting_metrics.items():
        results[k][sub] = v
    json.dump(results, open(result_path, "w"), indent=2)
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=256)

    # Evaluation functions
    metric_meteor = load_metric("meteor")
    metric_bleu = load_metric("bleu")
    bertscore_gpu_device = device
    metric_bertscore = load_metric("bertscore", device=bertscore_gpu_device)
    print("bertscore device = ", bertscore_gpu_device)
    metric_rouge = load_metric("rouge")

    sources = ['e2e', 'wsql', 'wtq', 'webnlg']
    sub_range = 5
    batch_size = 32
    for source in sources:
        test_dataloader = get_dataloader(f"{source}_test.tsv", batch_size)
        for sub in range(sub_range):
            print("Evaluating", source, sub)
            total_loss, source_triplets, generated_texts, target_texts = evaluate(test_dataloader, source, sub)
            results = update_results(total_loss, source_triplets, generated_texts, target_texts)
            # with open(f"generations/{source}_{sub}.txt", "w") as f:
            #     f.write("\n".join(generated_texts) + "\n")
        #     break ###
        # break ###c
