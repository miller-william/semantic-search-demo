import spacy
from sentence_transformers import CrossEncoder
from transformers import pipeline
import tensorflow_hub as hub

########################### Model imports ###########################

nlp = spacy.load("en_core_web_sm")
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") 

#model downloaded and saved locally because of issues using link above
#use_model = hub.load('/Users/william.miller/Downloads/universal-sentence-encoder_4')

#cross-encoder model
cross_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

#meeting summarizer
summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
