import os
from groq import Groq
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv, dotenv_values 
import time

# loading variables from .env file
load_dotenv() 
# Use the Llama3 70b model
model = os.getenv("MODEL")

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

def get_response(prompt):
    completion = client.chat.completions.create(
      model=model,
      messages=[
          {
           "role":"system",
            "content":"You are a Story writter"
          },
          {
              "role": "user",
              "content": prompt
          }
      ]
    )
    return completion.choices[0].message.content

simple_prompt = "tell a story about peacock?"

response = get_response(simple_prompt)

print(response)
