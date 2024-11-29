import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np

class DialogueDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        # Get conversation and its reply
        conv = self.conversations[idx]
        if 'reply_to' in conv and conv['reply_to'] in self.conversations_dict:
            prev_message = self.conversations_dict[conv['reply_to']]['text']
            current_message = conv['text']
            
            # Combine previous and current message with special tokens
            full_text = f"<|prompter|>{prev_message}<|assistant|>{current_message}<|endoftext|>"
            
            # Tokenize
            encodings = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            return {
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze()
            }

def prepare_data(file_path):
    # Load your preprocessed JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to format suitable for training
    conversations = []
    for conv_id, messages in data.items():
        for message in messages:
            conversations.append({
                'id': message['id'],
                'text': message['text'],
                'speaker': message['speaker'],
                'reply_to': message.get('reply_to', None)
            })
    
    return conversations

def train_model():
    # Initialize model and tokenizer
    model_name = "gpt2"  # You can change this to other models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = {
        'additional_special_tokens': ['<|prompter|>', '<|assistant|>', '<|endoftext|>']
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare data
    conversations = prepare_data('path_to_your_processed_data.json')
    dataset = DialogueDataset(conversations, tokenizer)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./dialogue_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,  # You might want to split into train/eval
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and tokenizer
    model.save_pretrained("./dialogue_model_final")
    tokenizer.save_pretrained("./dialogue_model_final")

if __name__ == "__main__":
    train_model()