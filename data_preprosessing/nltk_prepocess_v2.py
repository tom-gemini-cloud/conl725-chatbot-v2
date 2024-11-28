import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import json
import pickle
import os
import ssl

# Download required NLTK data ignoring SSL errors
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class NLTKDialogPreprocessor:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.conversations = {}  # Dictionary to store conversations by conversation_id
        self.vocabulary = FreqDist()

    def load_json_conversations(self, json_data):
        """Load conversations from JSON format"""
        print("Loading conversations...")
        
        # Group messages by conversation_id
        for line in json_data.split('\n'):
            if line.strip():
                message = json.loads(line)
                conv_id = message['conversation_id']
                
                if conv_id not in self.conversations:
                    self.conversations[conv_id] = []
                    
                self.conversations[conv_id].append({
                    'id': message['id'],
                    'text': message['text'],
                    'speaker': message['speaker'],
                    'reply_to': message['reply-to'],
                    'parsed': message['meta']['parsed']
                })
        
        print(f"Loaded {len(self.conversations)} conversations")

    def clean_text(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = ' '.join(word_tokenize(text))
        
        return text.strip()

    def preprocess_text(self, text):
        """Full text preprocessing pipeline using NLTK"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]
        
        return ' '.join(processed_tokens)

    def build_vocabulary(self):
        """Create vocabulary from processed texts"""
        all_words = []
        for conv_id, messages in self.conversations.items():
            for message in messages:
                if 'processed_text' in message:
                    words = word_tokenize(message['processed_text'])
                    all_words.extend(words)
        
        self.vocabulary = FreqDist(all_words)
        print(f"Vocabulary size: {len(self.vocabulary)}")

    def process_dataset(self):
        """Run the complete preprocessing pipeline"""
        print("Starting preprocessing pipeline...")
        
        # Process all messages
        total_messages = sum(len(messages) for messages in self.conversations.values())
        processed = 0
        
        for conv_id, messages in self.conversations.items():
            for message in messages:
                # Process text
                message['processed_text'] = self.preprocess_text(message['text'])
                
                # Use POS tags from parsed data
                if message['parsed']:
                    message['pos_tags'] = [
                        (tok['tok'], tok['tag'])
                        for sent in message['parsed']
                        for tok in sent['toks']
                    ]
                
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed}/{total_messages} messages...")
        
        # Build vocabulary
        self.build_vocabulary()
    
    def save_data(self, output_path):
        """Save processed conversations and vocabulary"""
        os.makedirs(output_path, exist_ok=True)
        
        # Save conversations
        with open(os.path.join(output_path, 'processed_conversations.pkl'), 'wb') as f:
            pickle.dump(self.conversations, f)
        
        # Save vocabulary
        with open(os.path.join(output_path, 'vocabulary.pkl'), 'wb') as f:
            pickle.dump(self.vocabulary, f)
        
        print(f"Saved processed data to {output_path}")

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = NLTKDialogPreprocessor()
    
    # Load and process the dataset
    with open('./data/movie_corpus/utterances.jsonl', 'r') as f:
        json_data = f.read()
    
    preprocessor.load_json_conversations(json_data)
    preprocessor.process_dataset()
    
    # Save the processed data
    preprocessor.save_data('processed_data')
    
    # Print sample conversation
    sample_conv_id = list(preprocessor.conversations.keys())[0]
    print("\nSample processed conversation:")
    for message in preprocessor.conversations[sample_conv_id]:
        print(f"Speaker {message['speaker']}: {message['processed_text']}")
        print(f"POS Tags: {message['pos_tags']}\n")