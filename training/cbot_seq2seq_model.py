import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm

class DialogDataset(Dataset):
    def __init__(self, conversations, vocab, max_length=30):
        self.vocab = vocab
        self.max_length = max_length
        self.word2idx = {word: idx for idx, (word, _) in enumerate(vocab.most_common())}
        self.word2idx['<PAD>'] = len(self.word2idx)
        self.word2idx['<START>'] = len(self.word2idx)
        self.word2idx['<END>'] = len(self.word2idx)
        self.word2idx['<UNK>'] = len(self.word2idx)
        
        self.pairs = []
        for conv_id, messages in conversations.items():
            for i in range(len(messages) - 1):
                self.pairs.append((
                    messages[i]['processed_text'],
                    messages[i + 1]['processed_text']
                ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_text, target_text = self.pairs[idx]
        
        # Convert to indices
        input_indices = self.text_to_indices(input_text)
        target_indices = self.text_to_indices(target_text)
        
        return {
            'input': torch.tensor(input_indices),
            'target': torch.tensor(target_indices)
        }
    
    def text_to_indices(self, text):
        words = text.split()
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        indices = [self.word2idx['<START>']] + indices + [self.word2idx['<END>']]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices += [self.word2idx['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
            
        return indices

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.3):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        
    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.3):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output)
        return output, hidden

class Seq2SeqModel:
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, device='cuda'):
        self.device = device
        self.encoder = EncoderRNN(vocab_size, embedding_dim, hidden_dim).to(device)
        self.decoder = DecoderRNN(vocab_size, embedding_dim, hidden_dim).to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab_size-4)  # ignore PAD token
        lr = 0.001
        self.encoder_optimizer = optim.AdamW(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.AdamW(self.decoder.parameters(), lr=lr)
        
        # Add learning rate schedulers
        self.encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder_optimizer, mode='min', patience=3, factor=0.5
        )
        self.decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.decoder_optimizer, mode='min', patience=3, factor=0.5
        )
        
    def train_step(self, input_batch, target_batch):
        # Zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        # Get encoder outputs
        encoder_output, encoder_hidden = self.encoder(input_batch)
        
        # Teacher forcing
        decoder_output, _ = self.decoder(target_batch, encoder_hidden)
        
        # Calculate loss
        loss = self.criterion(
            decoder_output.view(-1, decoder_output.size(-1)),
            target_batch.view(-1)
        )
        
        # Backpropagate
        loss.backward()
        
        # Update parameters
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        return loss.item()

def train_model(model, train_loader, val_loader, num_epochs=10, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Get the original dataset from the train_loader
    original_dataset = train_loader.dataset.dataset  # Access the parent dataset
    
    for epoch in range(num_epochs):
        # Training phase
        model.encoder.train()
        model.decoder.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            input_batch = batch['input'].to(model.device)
            target_batch = batch['target'].to(model.device)
            
            loss = model.train_step(input_batch, target_batch)
            total_train_loss += loss
            progress_bar.set_postfix({'train_loss': f'{loss:.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.encoder.eval()
        model.decoder.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_batch = batch['input'].to(model.device)
                target_batch = batch['target'].to(model.device)
                
                # Get encoder outputs
                encoder_output, encoder_hidden = model.encoder(input_batch)
                decoder_output, _ = model.decoder(target_batch, encoder_hidden)
                
                loss = model.criterion(
                    decoder_output.view(-1, decoder_output.size(-1)),
                    target_batch.view(-1)
                )
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Update learning rate schedulers
        model.encoder_scheduler.step(avg_val_loss)
        model.decoder_scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'encoder_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'word2idx': original_dataset.word2idx  # Use the original dataset
            }, 'best_dialog_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break

def main():
    # Load processed data
    with open('./processed_data/processed_conversations.pkl', 'rb') as f:
        conversations = pickle.load(f)
    with open('./processed_data/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
    
    # Create dataset and dataloader
    dataset = DialogDataset(conversations, vocabulary)
    
    # Split data into train and validation
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    vocab_size = len(dataset.word2idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqModel(vocab_size, device=device)
    
    # Train model with validation
    train_model(model, train_loader, val_loader)
    
    # Save model
    torch.save({
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'word2idx': dataset.word2idx
    }, 'dialog_model.pt')

if __name__ == "__main__":
    main()