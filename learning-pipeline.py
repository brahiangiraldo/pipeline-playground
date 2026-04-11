from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel 



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


tokenize()


