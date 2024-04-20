import cohere
import os
import pandas as pd

os.environ['COHERE_API_KEY'] = 'gtNwvMCXjn0HnBpZ42YEbMMoXz6BqDvXY2aoQRoM'
co = cohere.Client(os.environ['COHERE_API_KEY'])

texts = [
   'When was Israeli Prime Minister Yitzhak Rabin assassinated'
]
response = co.embed(texts=texts, model='multilingual-22-12')
embeddings = response.embeddings # All embeddings for the texts
print(embeddings[0][:5]) # Let's check embeddings for the first text

with open('./Data/query.txt', 'w') as f:
    f.write(str(embeddings).replace('[','').replace(']','').replace(',',''))
    f.close()