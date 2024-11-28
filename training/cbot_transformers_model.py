import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqGeneration,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import json
from tqdm import tqdm

class PreprocessedDialogDataset(Dataset):
    def __init__(self, conversations):
        self.examples = []
        
        # Convert preprocessed data into training examples
        for conv_id, messages in conversations.items():
            for i in range(len(messages) - 1):
                # Use the encoded data from preprocessing
                self.examples.append({
                    'input_ids': messages[i]['encoded']['input_ids'].squeeze(),
                    'attention_mask': messages[i]['encoded']['attention_mask'].squeeze(),
                    'labels': messages[i + 1]['encoded']['input_ids'].squeeze()
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class DialogTrainer:
    def __init__(
        self,
        model_name='facebook/bart-base',
        batch_size=8,
        num_epochs=3,
        learning_rate=5e-5,
        output_dir='dialog_model'
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        
        # Initialize model (tokenizer should match what was used in preprocessing)
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name)

    def train(self, train_conversations, val_conversations=None):
        # Create datasets from preprocessed data
        train_dataset = PreprocessedDialogDataset(train_conversations)
        val_dataset = PreprocessedDialogDataset(val_conversations) if val_conversations else None

        # Define training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=self.learning_rate,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=100,
            save_total_limit=2,
            save_steps=1000,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            report_to="tensorboard"
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        
        # Save training configuration
        config = {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate
        }
        with open(f'{self.output_dir}/training_config.json', 'w') as f:
            json.dump(config, f)

def main():
    # Load your preprocessed data
    with open('./processed_data/processed_conversations.pkl', 'rb') as f:
        conversations = pickle.load(f)
    
    # Split conversations into train/val (90/10 split)
    conv_ids = list(conversations.keys())
    split_idx = int(len(conv_ids) * 0.9)
    train_convs = {k: conversations[k] for k in conv_ids[:split_idx]}
    val_convs = {k: conversations[k] for k in conv_ids[split_idx:]}
    
    # Initialize and train model
    trainer = DialogTrainer(
        model_name='facebook/bart-base',  # Use the same model as in preprocessing
        batch_size=8,
        num_epochs=3
    )
    
    trainer.train(train_convs, val_convs)

if __name__ == "__main__":
    main()