# import packages and functions
from utils.data_setup import text_to_processed_file, load_json_text_objects, text_object_to_dict
from utils.models import nlp, use_model, cross_model, summarizer
from utils.semantic_search import bi_semantic_search, bi_cross_semantic_search

text_to_processed_file('text1.txt','text1.json', nlp, use_model, replace_list=None, remove_list=None)
text_to_processed_file('text2.txt','text2.json', nlp, use_model, replace_list=None, remove_list=None)