import streamlit as st
from fastai.collab import *
import torch
from torch import nn
import pickle
import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import sentencepiece
import string
import requests

@st.cache_resource
def load_stuff():
  # Load the data loader 
  dls= pd.read_pickle('dataloader.pkl')
  # Create an instance of the model
  learn = collab_learner(dls, use_nn=True,layers=[20,10],y_range=(0,10.5))
  # Load the saved state dictionary
  state_dict = torch.load('myModel.pth',map_location=torch.device('cpu'))
  # Assign the loaded state dictionary to the model's load_state_dict() method
  learn.model.load_state_dict(state_dict)   
  #load books dataframe
  books = pd.read_csv('./data/BX_Books.csv', sep=';',encoding='latin-1')
  #load tokenizer
  tokenizer = PegasusTokenizer.from_pretrained("pszemraj/pegasus-x-large-book-summary")
  #load model
  model = PegasusForConditionalGeneration.from_pretrained("pszemraj/pegasus-x-large-book-summary")
  return dls, learn, books, tokenizer, model

dls, learn, books, tokenizer, model = load_stuff()

#function to get recommendations
def get_3_recs(book):
  book_factors = learn.model.embeds[1].weight
  idx = dls.classes['title'].o2i[book]
  distances = nn.CosineSimilarity(dim=1)(book_factors, book_factors[idx][None])
  idxs = distances.argsort(descending=True)[1:4]
  recs = [dls.classes['title'][i] for i in idxs]
  return recs

#function to get descriptions from Google Books
def search_book_description(title):
  # Google Books API endpoint for book search
  url = "https://www.googleapis.com/books/v1/volumes"
   # Parameters for the book search
  params = {
    "q": title,
    "maxResults": 1
    }
  # Send GET request to Google Books API
  response = requests.get(url, params=params)
  # Check if the request was successful
  if response.status_code == 200:
    # Parse the JSON response to extract the book description
      data = response.json()

      if "items" in data and len(data["items"]) > 0:
        book_description = data["items"][0]["volumeInfo"].get("description", "No description available.")
        return book_description
      else:
        print("No book found with the given title.")
        return None
  else:
    # If the request failed, print the error message
    print("Error:", response.status_code, response.text)
    return None
  
#function to ensure summaries end with punctuation
def cut(sum):
    last_punc_idx = max(sum.rfind(p) for p in string.punctuation)
    output = sum[:last_punc_idx + 1]
    return output


#function to summarize
def summarize(des_list):
    if "No description available." in des_list:
      idx = des_list.index("No description available.")
      des = des_list.copy()
      des.pop(idx)
      rest = summarize(des)
      rest.insert(idx,'No description available.')
      return rest
    else: 
      # Tokenize all the descriptions in the list
      encoded_inputs = tokenizer(des_list, truncation=True, padding="longest", return_tensors="pt")

      # Generate summaries for all the inputs
      summaries = model.generate(**encoded_inputs, max_new_tokens=100)

      # Decode the summaries and process them
      outputs = tokenizer.batch_decode(summaries, skip_special_tokens=True)
      outputs = list(map(cut, outputs))
      return outputs

#function to get cover images  
def get_covers(recs):
  imgs = [books[books['Book-Title']==r]['Image-URL-L'].tolist()[0]for r in recs]
  return imgs

#streamlit app construction
st.title('Your digital librarian')
st.markdown("Hi there! I recommend you books based on one you love (which might not be in the same genre because that's boring) and give you my own synopsis of each book. Enjoy!")
options = books["Book-Title"].tolist()
input = st.selectbox('Select your favorite book', options)
if st.button("Get recommendations"):
   recs = get_3_recs(input)
   descriptions = list(map(search_book_description,recs))
   des_sums = summarize(descriptions)
   imgs = get_covers(recs)
   
   col1, col2, col3 = st.columns(3)
   col1.image(imgs[0])
   col1.markdown(f"**{recs[0]}**")
   col1.write(des_sums[0])

   col2.image(imgs[1])
   col2.markdown(f"**{recs[1]}**")
   col2.write(des_sums[1])

   col3.image(imgs[2])
   col3.markdown(f"**{recs[2]}**")
   col3.write(des_sums[2])



