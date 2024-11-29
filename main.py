import json
import random
import re
import ssl
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data ignoring SSL errors
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data for tokenization, stopwords and lemmatisation
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialise the FastAPI app and templates
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load training data
try:
    with open('data/training_data.json', 'r') as f:
        training_data = json.load(f)
except FileNotFoundError:
    raise Exception("Training data file not found")
except json.JSONDecodeError:
    raise Exception("Invalid training data format")

class NLPProcessor:
    def __init__(self, training_data):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.training_data = training_data

    def preprocess_text(self, text):
        try:
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) 
                     for token in tokens if token not in self.stop_words]
            return tokens
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            return []

    def get_intent(self, tokens):
        try:
            intent_scores = {}
            message_text = ' '.join(tokens)
            
            for intent, data in self.training_data.items():
                # Improved scoring: Consider partial matches and keyword importance
                score = 0
                for keyword in data['keywords']:
                    if keyword in message_text:
                        # Give more weight to longer keyword matches
                        score += len(keyword.split())
                
                if score > 0:
                    intent_scores[intent] = score

            return max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else 'default'
        except Exception as e:
            print(f"Error determining intent: {e}")
            return 'default'

    def process_message(self, message):
        try:
            tokens = self.preprocess_text(message)
            intent = self.get_intent(tokens)
            responses = self.training_data[intent]['responses']
            return random.choice(responses)
        except Exception as e:
            print(f"Error processing message: {e}")
            return "I apologize, but I'm having trouble processing your message."

# Initialize NLP processor with loaded training data
nlp = NLPProcessor(training_data)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    # Get user message from form data
    form_data = await request.form()
    user_message = form_data['message']
    # Process the message and get bot response
    bot_response = nlp.process_message(user_message)
    # Render the chat response page
    return templates.TemplateResponse(
        "chat_response.html", 
        {
            "request": request, 
            "user_message": user_message, 
            "response": bot_response
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)