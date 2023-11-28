# %%
from copy import deepcopy
from tqdm import tqdm
import json
import math

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset

from torch.utils.data import DataLoader

from torch.optim import AdamW
from transformers import get_scheduler
from torch.nn import CrossEntropyLoss

sources = ['e2e', 'wsql', 'wtq']
batch_size = 16
num_epochs = 50
sub_range = 5
lr = 3e-4
patience = 5
gradient_accumulation_steps = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=256)

# %%
def get_dataloader(data_files, batch_size):
    dataset = load_dataset(path="data", data_files=data_files, split="train")
    tokenized_triplet = dataset.map(
        lambda sample: tokenizer(sample["triplet"], padding='max_length', truncation=True, max_length=256, return_tensors="pt"), 
        batched=True,
    )
    dataset = dataset.add_column("input_ids", tokenized_triplet["input_ids"])
    dataset = dataset.add_column("attention_mask", tokenized_triplet["attention_mask"])

    tokenized_text = dataset.map(
        lambda sample: tokenizer(sample["text"], padding='max_length', truncation=True, max_length=256, return_tensors="pt"), 
        batched=True,
    )
    tokenized_text["input_ids"][tokenized_text["input_ids"] == 0] = -100
    dataset = dataset.add_column(f"labels", tokenized_text["input_ids"])
    return DataLoader(dataset, shuffle=True, batch_size=batch_size)



# %%
def train(train_dataloader, model, optimizer, lr_scheduler):
    model.train()

    progress_bar = tqdm(range(math.ceil((len(train_dataloader)) / gradient_accumulation_steps)))
    step = 0
    batch_loss = 0
    total_loss = 0
    for batch in train_dataloader:
        inputs = {k: torch.stack(batch[k], dim=1).to(device) for k in ["input_ids", "attention_mask", "labels"]}
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='mean')
        loss = loss_fct(logits.view(-1, logits.size(-1)), inputs['labels'].view(-1))
        loss = loss / gradient_accumulation_steps

        loss.backward()

        batch_loss += loss.item()
        total_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            progress_bar.set_description("Train Batch Loss: %f" % (batch_loss))
            batch_loss = 0
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        step += 1
    
    return total_loss / len(train_dataloader)


def eval(val_dataloader, model):
    """
    "We select the best model by the validation set's METEOR score"
    """
    model.eval()

    progress_bar = tqdm(range(math.ceil((len(val_dataloader)) / gradient_accumulation_steps)))
    step = 0
    batch_loss = 0
    total_loss = 0
    generated_texts = []
    for batch in val_dataloader:
        inputs = {k: torch.stack(batch[k], dim=1).to(device) for k in ["input_ids", "attention_mask", "labels"]}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='mean')
        loss = loss_fct(logits.view(-1, logits.size(-1)), inputs['labels'].view(-1))
        loss = loss / gradient_accumulation_steps

        total_loss += loss.item()
        batch_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            progress_bar.set_description("Eval  Batch Loss: %f" % (batch_loss))
            batch_loss = 0
            progress_bar.update(1)

        step += 1


    val_loss = total_loss / len(val_dataloader)
    return val_loss, generated_texts


def finetune(source, sub):
    train_dataloader = get_dataloader(f"{source}_train_sub{sub}.tsv", batch_size)
    val_dataloader = get_dataloader(f"{source}_validation.tsv", batch_size)

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model.to(device)
    
    optimizer = AdamW(list(model.parameters()), lr=lr)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    best_epoch, best_score, best_model = None, None, None
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_loss = train(train_dataloader, model, optimizer, lr_scheduler)
        val_loss, val_generated_texts = eval(val_dataloader, model)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping
        if best_score is None or val_loss < best_score:
            best_epoch, best_score = epoch, val_loss

            model.cpu()
            best_model = deepcopy(model)
            model.to(device)

        elif epoch - best_epoch == patience:
            # Save model
            torch.save(best_model.state_dict(), f"checkpoints/{source}_{sub}.pt")

            # Save results
            results = {
                'best_epoch': best_epoch,
                'best_score': best_score,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }
            with open(f"results/{source}_{sub}.json", "w") as f:
                json.dump(results, f, indent=2)
            break


# %%
if __name__ == "__main__":
    for source in sources:
       for sub in range(sub_range):
           print("Finetuning", source, sub)
           finetune(source, sub)

