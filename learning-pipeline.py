from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel 

from transformers import AutoModelForMultipleChoice
import torch







def predictions():
  classifier = pipeline("sentiment-analysis")
  results = classifier(
      [
          "I've been waiting for a HuggingFace course my whole life.",
          "I hate this so much!",
      ]
  )
  print(results)

# predictions() 
def tokenize():
  
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print('----Start tokenizar-----')
    raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
     ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)
    print('---- End tokenizar-----')
     #Create the model with the same checkpoint as the tokenizer
    model = AutoModel.from_pretrained(checkpoint)
    #Get the last hidden state of the model
    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)


# tokenize()



def multiple_choice():
    model_name = "LIAMF-USP/roberta-large-finetuned-race"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)

    prompt = "Love is a very important thing in life. It can make us feel happy and fulfilled."
    choices = ["Devil", "Love", "Angry", "happy" , "Sad"]

    encodings = tokenizer(
        [[prompt, c] for c in choices] ,
        return_tensors="pt", padding=True, truncation=True
    )
    inputs = {k: v.unsqueeze(0) for k, v in encodings.items()}
    outputs = model(**inputs)
    predicted = torch.argmax(outputs.logits).item()
    print(f"Answer: {outputs.logits}, Predicted choice: {choices[predicted]}")

multiple_choice()