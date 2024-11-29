from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import torch
from typing import Dict, List
import logging
from pathlib import Path
import html

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your custom model classes
from training.cbot_seq2seq_model import EncoderRNN, DecoderRNN

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class CustomChatbot:
    def __init__(self, model_path: str = 'models/dialog_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self._load_model()
        self._initialize_models()
        
    def _load_model(self) -> None:
        try:
            self.checkpoint = torch.load(self.model_path, map_location=self.device)
            self.word2idx: Dict[str, int] = self.checkpoint['word2idx']
            self.idx2word: Dict[int, str] = {idx: word for word, idx in self.word2idx.items()}
            
            required_tokens = {'<PAD>', '<START>', '<END>', '<UNK>'}
            missing_tokens = required_tokens - set(self.word2idx.keys())
            if missing_tokens:
                raise ValueError(f"Missing required special tokens: {missing_tokens}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def _initialize_models(self) -> None:
        try:
            vocab_size = len(self.word2idx)
            self.embedding_dim = 256
            self.hidden_dim = 512
            
            self.encoder = EncoderRNN(vocab_size, self.embedding_dim, self.hidden_dim).to(self.device)
            self.decoder = DecoderRNN(vocab_size, self.embedding_dim, self.hidden_dim).to(self.device)
            
            self.encoder.load_state_dict(self.checkpoint['encoder_state_dict'])
            self.decoder.load_state_dict(self.checkpoint['decoder_state_dict'])
            
            self.encoder.eval()
            self.decoder.eval()
            
            self.max_length = 30
            
            logger.info(f"Models initialized successfully. Vocabulary size: {vocab_size}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def text_to_indices(self, text: str) -> torch.Tensor:
        text = html.escape(text.strip().lower())
        if not text:
            raise ValueError("Empty input text")
            
        words = text.split()
        if len(words) > self.max_length:
            words = words[:self.max_length]
            
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        indices = [self.word2idx['<START>']] + indices + [self.word2idx['<END>']]
        
        padding_length = self.max_length - len(indices)
        if padding_length > 0:
            indices.extend([self.word2idx['<PAD>']] * padding_length)
        else:
            indices = indices[:self.max_length]
            
        # Create tensor with shape [batch_size, sequence_length]
        input_tensor = torch.tensor(indices, dtype=torch.long, device=self.device).unsqueeze(0)
        logger.info(f"Input tensor shape: {input_tensor.shape}")
        return input_tensor

    def generate_response(self, user_message: str) -> str:
        try:
            with torch.no_grad():
                input_tensor = self.text_to_indices(user_message)
                print(f"Input tensor passed to encoder: {input_tensor.shape}")
                logger.info(f"Input tensor passed to encoder: {input_tensor.shape}")

                # Encoder forward pass
                encoder_output, encoder_hidden = self.encoder(input_tensor)
                logger.info(f"Encoder output shape: {encoder_output.shape}")
                logger.info(f"Encoder hidden state shape: {encoder_hidden.shape}")
                
                # Initialize decoder input [batch_size, 1]
                decoder_input = torch.tensor([[self.word2idx['<START>']]], device=self.device)
                decoder_hidden = encoder_hidden
                
                response_words: List[str] = []
                for _ in range(self.max_length):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    logger.info(f"Decoder output shape: {decoder_output.shape}")
                    
                    topv, topi = decoder_output.topk(1)
                    word_idx = topi.squeeze().item()
                    
                    if word_idx == self.word2idx['<END>']:
                        break
                        
                    word = self.idx2word[word_idx]
                    if word not in {'<PAD>', '<START>', '<UNK>'}:
                        response_words.append(word)
                        
                    decoder_input = topi.detach()
                    
                response = ' '.join(response_words)
                return html.escape(response) if response else "I'm not sure how to respond to that."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return "I apologize, but I'm having trouble processing your message."

# Initialize the chatbot with error handling
try:
    chatbot = CustomChatbot()
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {e}")
    raise

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    try:
        form_data = await request.form()
        user_message = form_data.get('message', '').strip()
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
            
        bot_response = chatbot.generate_response(user_message)
        
        return templates.TemplateResponse(
            "chat_response.html",
            {
                "request": request,
                "user_message": html.escape(user_message),
                "response": bot_response
            }
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return templates.TemplateResponse(
            "chat_response.html",
            {
                "request": request,
                "user_message": html.escape(user_message) if 'user_message' in locals() else "",
                "response": "I apologise, but something went wrong. Please try again."
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
