import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_PATH = "./processed_data/processed_conversations.pkl"

class DialogueDataset(Dataset):
    def __init__(self, conversations_dict, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Flatten the conversations into a list of messages with their metadata
        self.conversations = []
        print("Processing conversations...")
        for conv_id, messages in tqdm(conversations_dict.items(), desc="Loading conversations"):
            for message in messages:
                if isinstance(message, dict):  # Ensure we have a valid message
                    self.conversations.append(message)
        
        # Create a lookup dictionary for faster access
        print("Creating lookup dictionary...")
        self.conversations_dict = {
            msg['id']: msg for msg in tqdm(self.conversations, desc="Building lookup dict") 
            if isinstance(msg, dict)
        }
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        # Ensure idx is treated as a single integer
        if isinstance(idx, list):
            idx = idx[0]
        
        # Get conversation and its reply
        conv = self.conversations[idx]
        
        # Get the text for the current message
        current_message = conv['text']
        
        # Get the previous message if it exists
        prev_message = ""
        if 'reply_to' in conv and conv['reply_to'] in self.conversations_dict:
            prev_message = self.conversations_dict[conv['reply_to']]['text']
        
        # Combine previous and current message with special tokens
        if prev_message:
            full_text = f"<|prompter|>{prev_message}<|assistant|>{current_message}<|endoftext|>"
        else:
            full_text = f"<|assistant|>{current_message}<|endoftext|>"
        
        try:
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
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            print(f"Full text: {full_text}")
            raise e

def prepare_data(file_path):
    print(f"Loading data from {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def train_model(data_path=DATA_PATH):
    # Initialize model and tokenizer
    model_name = "gpt2"
    print(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = {
        'additional_special_tokens': ['<|prompter|>', '<|assistant|>', '<|endoftext|>']
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Set padding token to eos token
    tokenizer.pad_token = tokenizer.eos_token
    
    model.resize_token_embeddings(len(tokenizer))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Prepare data
    conversations_dict = prepare_data(data_path)
    dataset = DialogueDataset(conversations_dict, tokenizer)
    
    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)  # For reproducibility
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./dialogue_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=250,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        max_grad_norm=1.0,
        learning_rate=2e-5,
        gradient_accumulation_steps=2,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained("./dialogue_model_final")
    tokenizer.save_pretrained("./dialogue_model_final")
    print("Training complete!")

if __name__ == "__main__":
    train_model()