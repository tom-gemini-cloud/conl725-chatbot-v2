import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

class ChatbotDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=128):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        # Tokenize inputs
        encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Prepare target
        target_encoding = self.tokenizer(
            answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

class TransformerChatbot(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=None):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add response generation layers
        self.decoder = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, self.transformer.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.transformer.config.hidden_size, num_labels if num_labels else self.tokenizer.vocab_size)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        logits = self.decoder(sequence_output)
        return logits

def train_chatbot(model, train_dataloader, num_epochs=5, learning_rate=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            # Reshape outputs and labels for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

def generate_response(model, tokenizer, question, max_length=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        question,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predicted_tokens = torch.argmax(outputs, dim=-1)
    
    response = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
    return response

questions = ["How are you?", "What's the weather like?"]
answers = ["I'm doing well, thank you!", "I don't have access to weather information."]
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = ChatbotDataset(questions, answers, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = TransformerChatbot('bert-base-uncased')

train_chatbot(model, dataloader, num_epochs=5)

question = "How are you today?"
response = generate_response(model, tokenizer, question)
print(f"Question: {question}\nResponse: {response}")