import os
import argparse

import torch
from torch.nn import CrossEntropyLoss
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import numpy as np
import pandas as pd
import random
import math

from nltk.tokenize import word_tokenize

from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader

from torch.optim import AdamW
from transformers import get_scheduler

from tqdm.auto import tqdm

import sys
import json

sys.argv = ['--gpu_device', '0',
            '--num_epochs', '1',
            '--output_dir', 'playground/',
            '--data2text_model', 't5-base',
            '--text2data_model', 't5-base',
            '--train_file', 'data/wsql_train.tsv',
            '--do_train', '--do_eval', '--do_test', '--do_generate',
            '--scorer_model', 'roberta-base',
            '--validation_file', 'data/wsql_validation.tsv',
            '--test_file', 'data/wsql_test.tsv',
            '--per_gpu_train_batch_size', '8',
            '--per_gpu_eval_batch_size', '8']

parser = argparse.ArgumentParser()
# General
parser.add_argument("--config_file", default=None, type=str,
                    help="Optional use of config file for passing the arguments")
parser.add_argument("--output_dir", default=None, type=str,
                    help="The output directory where the model predictions and checkpoints will be written")
parser.add_argument("--gpu_device", default=0, type=int,
                    help="GPU device id")
parser.add_argument("--bertscore_gpu_device", default=0, type=int,
                    help="GPU device id for bertscore model")

# Model
parser.add_argument("--t5_tokenizer", default="t5-base", type=str,
                    help="Tokenizer for T5 models")
parser.add_argument("--data2text_model", default=None, type=str,
                    help="Local or Huggingface transformer's path to the data2text model")
parser.add_argument("--text2data_model", default=None, type=str,
                    help="Local or Huggingface transformer's path to the text2data_model model")

# Data
parser.add_argument("--train_file", default=None, type=str,
                    help="Train file used for cycle training")

parser.add_argument("--validation_file", default=None, type=str,
                    help="Validation set for development")

parser.add_argument("--test_file", default=None, type=str,
                    help="Test set for evaluation")

# Generation Parameters
parser.add_argument("--max_input_length", default=256, type=int,
                    help="Maximum input length including prompt after tokenization")
parser.add_argument("--min_output_length", default=3, type=int,
                    help="Minimum output length")
parser.add_argument("--max_output_length", default=256, type=int,
                    help="Maximum output length")
parser.add_argument("--num_beams", default=4, type=int,
                    help="Number of beams for beam search")
parser.add_argument("--no_repeat_ngram_size", default=0, type=int,
                    help="No repeat ngram size")
parser.add_argument("--length_penalty", default=0, type=float,
                    help="Length penalty")

# Cycle Training Parameters
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--seed", default=1, type=int,
                    help="Random seed")
parser.add_argument("--num_epochs", default=25, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                    help="Batch size per GPU/CPU for training")
parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                    help="Batch size per GPU/CPU for evaluation")
parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                    help="Number of updates steps to accumulate before performing a backward/update pass; effective training batch size equals to per_gpu_train_batch_size * gradient_accumulation_steps")
parser.add_argument("--data2text_learning_rate", default=3e-4, type=float,
                    help="The initial learning rate of AdamW for the data2text model; larger learning rate is suggested for T5 families")
parser.add_argument("--text2data_learning_rate", default=3e-4, type=float,
                    help="The initial learning rate of AdamW for the text2data model; larger learning rate is suggested for T5 families")
parser.add_argument('--scheduler_type', type=str, default="linear",
                    help="Learning rate scheduler type (linear/cosine/cosine_with_restarts/polynomial/constant/constant_with_warmup)")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Scheduler warmup steps")

# Adaptive Cycle Training Parameters
parser.add_argument("--adaptive_type", default=0, type=int,
                    help="0: No adaptive learning; 1: adaptive instance weighted loss; 2: adaptive learning rate")
parser.add_argument("--scorer_model_tokenizer", default="roberta-base", type=str,
                    help="Tokenizer for the scorer model")
parser.add_argument("--scorer_model", default=None, type=str,
                    help="Local path to the scorer model")

# Evaluation Parameters
parser.add_argument("--do_eval", action='store_true',
                    help="Whether to run eval on the dev set")

parser.add_argument("--do_generate", action='store_true',
                    help="Whether to run generation for the evaluation of the dev set")

parser.add_argument("--do_test", action='store_true',
                    help="Whether to run eval on the test set")

# Model Selection & Saving Strategy
parser.add_argument('--save_epochs', type=int, default=1,
                    help="Save model every X updates epochs")
parser.add_argument('--selection_metric', type=str, default="loss",
                    help="The metric used for model section; --do_generate required for metric other than loss")
parser.add_argument('--delta', type=float, default=0.01,
                    help="Minimum requirement of improvement")
parser.add_argument('--patience', type=int, default=3,
                    help="Terminate the training after n epochs without any improvement")

args = parser.parse_args(sys.argv)

if args.config_file is not None:
    with open(args.config_file, 'r') as f:
        args.__dict__ = json.load(f)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

print(args.__dict__)
with open(os.path.join(args.output_dir, 'config.dict'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device('cuda:' + str(args.gpu_device)) if torch.cuda.is_available() else torch.device('cpu')

tokenizer = T5Tokenizer.from_pretrained(args.t5_tokenizer, model_max_length=512)
if args.text2data_model is not None:
    model_text2data = T5ForConditionalGeneration.from_pretrained(args.text2data_model)
    model_text2data.config.task_specific_params["summarization"] = {
        "early_stopping": True,
        "length_penalty": args.length_penalty,
        "max_length": args.max_output_length,
        "min_length": args.min_output_length,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "num_beams": args.num_beams,
        "prefix": ""
    }
    model_text2data.to(device)

if args.data2text_model is not None:
    model_data2text = T5ForConditionalGeneration.from_pretrained(args.data2text_model)
    model_data2text.config.task_specific_params["summarization"] = {
        "early_stopping": True,
        "length_penalty": args.length_penalty,
        "max_length": args.max_output_length,
        "min_length": args.min_output_length,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "num_beams": args.num_beams,
        "prefix": ""
    }
    model_data2text.to(device)

if args.scorer_model is not None:
    tokenizer_scorer = RobertaTokenizer.from_pretrained(args.scorer_model_tokenizer, model_max_length=512)
    model_scorer = RobertaForSequenceClassification.from_pretrained(args.scorer_model, num_labels=1)
    model_scorer.to(device)


def tokenize_content(sample, column):
    return tokenizer(sample[column], padding='max_length', truncation=True, max_length=args.max_input_length)


def reformat_target(sample, column):
    return sample[column].split(':')[1]


def get_train_dataloader(file_name, batch_size):
    df = pd.read_table(file_name, delimiter='\t')
    text_dataset = Dataset.from_pandas(df[['text']])
    tokenized_text = text_dataset.map(lambda sample: tokenize_content(sample, 'text'), batched=True)
    tokenized_text.set_format('torch', ['attention_mask', 'input_ids'], output_all_columns=True)

    triplet_dataset = Dataset.from_pandas(df[['triplet']])
    tokenized_triplets = triplet_dataset.map(lambda sample: tokenize_content(sample, 'triplet'), batched=True)
    tokenized_triplets.set_format('torch', ['attention_mask', 'input_ids'], output_all_columns=True)

    return DataLoader(tokenized_text, shuffle=True, batch_size=batch_size), \
           DataLoader(tokenized_triplets, shuffle=True, batch_size=batch_size)


def get_eval_dataloader(file_name, batch_size):
    df = pd.read_table(file_name, delimiter='\t')
    text2triplet_df = df.copy()
    text2triplet_df['triplet'] = text2triplet_df['triplet'].apply(lambda sample: sample.split(':')[1])
    text2triplet_df.rename(columns={'text': 'source', 'triplet': 'target'}, inplace=True)
    text2triplet_dataset = Dataset.from_pandas(text2triplet_df)
    text2triplet_dataset = text2triplet_dataset.map(lambda sample: tokenize_content(sample, 'source'), batched=True)
    text2triplet_dataset.set_format('torch', ['attention_mask', 'input_ids'], output_all_columns=True)

    triplet2text_df = df.copy()
    triplet2text_df['text'] = triplet2text_df['text'].apply(lambda sample: sample.split(':')[1])
    triplet2text_df.rename(columns={'triplet': 'source', 'text': 'target'}, inplace=True)
    triplet2text_dataset = Dataset.from_pandas(triplet2text_df)
    triplet2text_dataset = triplet2text_dataset.map(lambda sample: tokenize_content(sample, 'source'), batched=True)
    triplet2text_dataset.set_format('torch', ['attention_mask', 'input_ids'], output_all_columns=True)

    return DataLoader(text2triplet_dataset, shuffle=True, batch_size=batch_size), \
           DataLoader(triplet2text_dataset, shuffle=True, batch_size=batch_size)

# Process training data and initialize training parameters
if args.do_train and args.train_file is not None:
    text_dataloader, triplets_dataloader = get_train_dataloader(args.train_file, args.per_gpu_train_batch_size)

    optimizer_text2data = AdamW(list(model_text2data.parameters()), lr=args.text2data_learning_rate)
    optimizer_data2text = AdamW(list(model_data2text.parameters()), lr=args.data2text_learning_rate)

    num_text_training_steps = args.num_epochs * len(text_dataloader)
    num_data_training_steps = args.num_epochs * len(triplets_dataloader)

    lr_scheduler_text2data = get_scheduler(
        args.scheduler_type,
        optimizer=optimizer_text2data,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_text_training_steps
    )

    lr_scheduler_data2text = get_scheduler(
        args.scheduler_type,
        optimizer=optimizer_data2text,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_data_training_steps
    )

# Process validation data
if args.do_eval and args.validation_file is not None:
    val_text2triplet_dataloader, val_triplet2text_dataloader = get_eval_dataloader(args.validation_file,
                                                                                   args.per_gpu_eval_batch_size)

# Process testing data
if args.do_test and args.test_file is not None:
    test_text2triplet_dataloader, test_triplet2text_dataloader = get_eval_dataloader(args.test_file,
                                                                                     args.per_gpu_eval_batch_size)


# Main cycle training function that performs training on one direction
def train_one_direction(model1, model2, data_loader, num_training_steps, optimizer, lr_scheduler, inter_type):
    model1.eval()  # frozen model
    model2.train()  # trained model

    progress_bar = tqdm(range(math.ceil((len(data_loader)) / args.gradient_accumulation_steps)))
    step = 0
    batch_loss = 0
    total_loss = 0
    batch_score = 0
    total_score = 0

    for batch in data_loader:
        # Prepare Inputs
        raw_input = batch.pop('text' if inter_type == 'data' else 'triplet')
        model1_input = {k: v.to(device) for k, v in batch.items()}

        # Generate intermediate outputs
        with torch.no_grad():
            intermediate_outputs = model1.generate(**model1_input, min_length=args.min_output_length,
                                                   max_length=args.max_output_length, num_beams=args.num_beams,
                                                   early_stopping=True)

        decoded_intermediate_outputs = tokenizer.batch_decode(intermediate_outputs, skip_special_tokens=True)

        # Hot prefixing
        scorer_input = None
        if inter_type == 'text':
            decoded_intermediate_outputs = ['Extract Triplets: ' + item for item in decoded_intermediate_outputs]
            scorer_decoded_intermediate_outputs = [
                im_text + ' ' + im_triplets.replace('Generate in English', 'Triplets') for im_text, im_triplets in
                zip(decoded_intermediate_outputs, raw_input)]
        elif inter_type == 'data':
            decoded_intermediate_outputs = ['Generate in English: ' + item for item in decoded_intermediate_outputs]
            scorer_decoded_intermediate_outputs = [
                im_text + ' ' + im_triplets.replace('Generate in English', 'Triplets') for im_text, im_triplets in
                zip(raw_input, decoded_intermediate_outputs)]

        # prepare labels and inputs
        tokenized_intermediate_outputs = tokenizer(decoded_intermediate_outputs, return_tensors="pt", padding=True,
                                                   truncation=True, max_length=args.max_input_length)
        model2_label = [t.split(': ')[1] for t in raw_input]
        processed_labels = tokenizer(model2_label, return_tensors="pt", padding=True, truncation=True,
                                     max_length=args.max_input_length)['input_ids']
        processed_labels[processed_labels == 0] = -100
        tokenized_intermediate_outputs['labels'] = processed_labels

        model2_input = {k: v.to(device) for k, v in tokenized_intermediate_outputs.items()}

        # Scoring
        if args.scorer_model is not None:
            tokenized_scorer_input = tokenizer_scorer(scorer_decoded_intermediate_outputs, return_tensors="pt",
                                                      padding=True, truncation=True, max_length=args.max_input_length)
            scorer_input = {k: v.to(device) for k, v in tokenized_scorer_input.items()}

            with torch.no_grad():
                score = model_scorer(**scorer_input).logits

        # Final outputs and loss calculation
        outputs = model2(**model2_input)
        logits = outputs.logits

        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), model2_input['labels'].view(-1))

        # Handling of adaptive cycle training
        if args.adaptive_type == 1:
            loss = (loss * score).mean()
        else:
            loss = loss.mean()
            if args.scorer_model is not None:
                score = torch.mean(score)

        if args.adaptive_type == 2:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * score

        loss = loss / args.gradient_accumulation_steps
        if args.scorer_model is not None:
            score = score / args.gradient_accumulation_steps

        loss.backward()

        if args.scorer_model is not None:
            batch_score += score.item()
            total_score += score.item()

        batch_loss += loss.item()
        total_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.scorer_model is not None:
                progress_bar.set_description("Train Batch Loss: %f Score: %f" % (batch_loss, batch_score))
            else:
                progress_bar.set_description("Train Batch Loss: %f" % (batch_loss))
            batch_loss = 0
            batch_score = 0
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        step += 1
    return total_loss, total_score


# Evaluation function
metric_meteor = load_metric("meteor")
metric_bleu = load_metric("bleu")
metric_bertscore = load_metric("bertscore", device=args.bertscore_gpu_device)
print("bertscore device = ", args.bertscore_gpu_device)
metric_rouge = load_metric("rouge")


def eval_model(model, data_loader, test_mode=False):
    model.eval()
    progress_bar = tqdm(range(math.ceil((len(data_loader)) / args.gradient_accumulation_steps)))
    step = 0
    batch_loss = 0
    total_loss = 0
    source_texts = []
    generated_texts = []
    target_texts = []
    for batch in data_loader:
        raw_input = batch.pop('source')
        model_label = batch.pop('target')
        model_input = {k: v.to(device) for k, v in batch.items()}

        processed_labels = tokenizer(model_label, return_tensors="pt", padding='max_length', truncation=True,
                                     max_length=args.max_input_length)['input_ids']
        processed_labels[processed_labels == 0] = -100
        model_input['labels'] = processed_labels.to(device)

        with torch.no_grad():
            outputs = model(**model_input)

        if args.do_generate:
            del model_input['labels']
            with torch.no_grad():
                generated_outputs = model.generate(**model_input, min_length=args.min_output_length,
                                                   max_length=args.max_output_length, num_beams=args.num_beams,
                                                   early_stopping=True)
            decoded_outputs = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
            generated_texts += decoded_outputs
            target_texts += model_label
            source_texts += raw_input
        loss = outputs.loss
        total_loss += loss.item()

        loss = loss / args.gradient_accumulation_steps

        batch_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            progress_bar.set_description("Eval Batch Loss: %f" % (batch_loss))
            batch_loss = 0
            progress_bar.update(1)

        step += 1

    resulting_metrics = {'loss': total_loss / len(data_loader)}
    if args.do_generate or test_mode:
        meteor_p = []
        meteor_g = []
        bleu_p = []
        bleu_g = []
        bertscore_p = []
        bertscore_g = []
        rouge_p = []
        rouge_g = []
        last_s = ""
        for s, p, g in zip(source_texts, generated_texts, target_texts):
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

    return resulting_metrics, generated_texts


# Training and validating scripts
if args.do_train:
    epoch_progress_bar = tqdm(range(args.num_epochs))
    text2data_best = 1000000
    text2data_best_metrics = None
    text2data_patience = 0
    data2text_best = 1000000
    data2text_best_metrics = None
    data2text_patience = 0
    for epoch in range(args.num_epochs):
        epoch_progress_bar.set_description("Cycle/Epoch %d: " % (epoch))
        print("\nTraining: data-text-data direction")
        total_text2data_loss, total_text2data_score = train_one_direction(model_data2text, model_text2data,
                                                                          triplets_dataloader, num_data_training_steps,
                                                                          optimizer_text2data, lr_scheduler_text2data,
                                                                          'text')
        print('model_text2data - train total loss: %.4f score: %.4f' % (total_text2data_loss, total_text2data_score))

        if args.do_eval and args.validation_file is not None:
            text2data_dev_metrics, _ = eval_model(model_text2data, val_text2triplet_dataloader)
            total_text2data_dev_loss = text2data_dev_metrics['loss']
            if args.selection_metric == 'loss':
                text2data_selection_metric = text2data_dev_metrics[args.selection_metric]
            else:
                text2data_selection_metric = 1 - text2data_dev_metrics[args.selection_metric]
            print('model_text2data - dev total loss: %.4f' % (total_text2data_dev_loss))
            print(text2data_dev_metrics)

            if text2data_patience <= args.patience and text2data_best - text2data_selection_metric >= args.delta:
                text2data_best = text2data_selection_metric
                text2data_best_metircs = text2data_dev_metrics
                text2data_patience = 0
                model_text2data.save_pretrained(os.path.join(args.output_dir, 'text2data-best'))
                print("text2data-best saved: epoch/cycle %d" % (epoch))
            else:
                text2data_patience += 1

        if epoch % args.save_epochs == 0:
            model_text2data.save_pretrained(os.path.join(args.output_dir, 'text2data-' + str(epoch)))
            print('text2data-' + str(epoch) + ' saved')

        print("\nTraining: text-data-text direction")
        total_data2text_loss, total_data2text_score = train_one_direction(model_text2data, model_data2text,
                                                                          text_dataloader, num_text_training_steps,
                                                                          optimizer_data2text, lr_scheduler_data2text,
                                                                          'data')
        print('model_data2text - train total loss: %.4f score: %.4f' % (total_data2text_loss, total_data2text_score))

        if args.do_eval and args.validation_file is not None:
            data2text_dev_metrics, _ = eval_model(model_data2text, val_triplet2text_dataloader)
            total_data2text_dev_loss = data2text_dev_metrics['loss']
            if args.selection_metric == 'loss':
                data2text_selection_metirc = data2text_dev_metrics[args.selection_metric]
            else:
                data2text_selection_metirc = 1 - data2text_dev_metrics[args.selection_metric]
            print('model_data2text - dev total loss: %.4f' % (total_data2text_dev_loss))
            print(data2text_dev_metrics)

            if data2text_patience <= args.patience and data2text_best - data2text_selection_metirc >= args.delta:
                data2text_best = data2text_selection_metirc
                data2text_best_metircs = data2text_dev_metrics
                data2text_patience = 0
                model_data2text.save_pretrained(os.path.join(args.output_dir, 'data2text-best'))
                print("data2text-best saved: epoch/cycle %d" % (epoch))
            else:
                data2text_patience += 1

        if epoch % args.save_epochs == 0:
            model_data2text.save_pretrained(os.path.join(args.output_dir, 'data2text-' + str(epoch)))
            print('data2text-' + str(epoch) + ' saved')

        epoch_progress_bar.update(1)
        if text2data_patience > args.patience and data2text_patience > args.patience:
            print("Both models exceed the patience, training terminated")
            break
    print("\nTraining completed")
    if args.do_eval:
        print("\nBest data2text model:")
        print(data2text_best_metrics)
        del model_data2text

        print("\nBest text2data model:")
        print(text2data_best_metrics)
        del model_text2data

    if args.do_test:
        if args.data2text_test_file is not None:
            model_data2text = T5ForConditionalGeneration.from_pretrained(
                os.path.join(args.output_dir, 'data2text-best'))
            model_data2text.to(device)

        if args.text2data_test_file is not None:
            model_text2data = T5ForConditionalGeneration.from_pretrained(
                os.path.join(args.output_dir, 'text2data-best'))
            model_text2data.to(device)

# Testing scripts
if args.do_test:
    if args.data2text_model is not None and args.test_file is not None:
        data2text_test_metrics, data2text_generations = eval_model(model_data2text, test_triplet2text_dataloader,
                                                                   test_mode=True)
        print("\ndata2text test:")
        print(data2text_test_metrics)
        out = open(os.path.join(args.output_dir, 'data2text.generations'), 'w')
        out.write('\n'.join(data2text_generations))
        out.close()

    if args.text2data_model is not None and args.test_file is not None:
        text2data_test_metrics, text2data_generations = eval_model(model_text2data, test_text2triplet_dataloader,
                                                                   test_mode=True)
        print("\ntext2data test:")
        print(text2data_test_metrics)
        out = open(os.path.join(args.output_dir, 'text2data.generations'), 'w')
        out.write('\n'.join(text2data_generations))
        out.close()
