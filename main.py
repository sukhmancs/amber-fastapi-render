from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your model here
tokenizer = AutoTokenizer.from_pretrained("qbwmwsap/amber-model-mine")
model = AutoModelForCausalLM.from_pretrained("qbwmwsap/amber-model-mine")

# Define a Pydantic model for the input data
class Item(BaseModel):
    text: str

app = FastAPI()


origins = [
    "*",  # Allow requests from your local machine
    # Add any other origins (websites) that should be allowed to make requests to your API
]

# Add CORS middleware to your FastAPI app
# This will allow your API to accept requests from any origin
# This will allow multiple headers and methods are allowed 
# eg. allow_methods=["GET", "POST"], allow_headers=["X-Custom-Header"] etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],
)

@app.post('/predict')
async def predict(item: Item):
    # Preprocess your data here
    input_ids = tokenizer(item.text, return_tensors="pt").input_ids

    # Run the model and get the output
    outputs = model.generate(input_ids)

    # Postprocess your output here if necessary    
    output = tokenizer.decode(outputs[0])

    return {'output': output}