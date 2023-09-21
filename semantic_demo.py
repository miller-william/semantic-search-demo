######## Semantic search demo #######
# Requirements to run:
# utils folder must be in the same location as this semantic_demo.py file.
# Must have two files: output.json and output_2.json int this folder. These must be created by data_setup.text_to_processed_file().

import sys
import streamlit as st
from datetime import datetime, timedelta, date

from utils.data_setup import load_json_text_objects, text_object_to_dict
from utils.models import nlp, use_model, cross_model, summarizer
from utils.semantic_search import keyword_search, bi_semantic_search, bi_cross_semantic_search
from utils.misc import random_date

st.set_page_config(
    page_title=None,
    page_icon=None,
    layout='wide',
    initial_sidebar_state='auto'
)

def load_source_texts(selected_option):
    # Use the selected option to load the corresponding file
    if selected_option == 'Text 1':
        source_texts = load_json_text_objects('text1.json')
    elif selected_option == 'Text 2':
        source_texts = load_json_text_objects('text2.json')
    return text_object_to_dict(source_texts)

# Check if 'session_state' has been initialized already, if not, then initialize it.
if 'session_state' not in st.session_state:
    st.session_state['session_state'] = {"keyword_results": [], "ai_search_results": []}

# Initialise some parameters
summary_length = 50   # max length of huggingface summariser output
paragraph_length = 4  # Number of output sentence either side of matching sentence
progress_button = False

# Show control panel if 'View' is selected

st.sidebar.subheader('Control Panel')

# Set some system variables
summary_length = st.sidebar.slider('Summary max length', 0, 100, summary_length)
paragraph_length = st.sidebar.slider('No. of output sentences', 0,8,paragraph_length)
    
    
# this sets the title of your app
st.title('Semantic search: Demo')

# Create two columns at the top
col1, col2 = st.columns(2)

# In the first column, put the selection box for texts
with col1:
    # Define the list of options that you want to display in the drop-down. 
    # These can be the names of your files.
    options = ['Text 1', 'Text 2']

    # Create the select box
    selected_option = st.selectbox('Select a text:', options)

with col2:
    # In the second column, put the slider for date range
    min_date = date(2020, 1, 1)
    max_date = date.today()
    date_dummy = st.date_input('Select a date range (dummy):',[min_date,max_date] )

source_texts = load_source_texts(selected_option)

# this will create a numeric input box with the label 'Number of results:'
num_results = st.number_input('Number of results:', min_value=1, value=5)

# create two columns
col1, col2 = st.columns(2)

# in the first column, display Keyword search
with col1:
    st.header('Keyword search')
    keyword = st.text_input('Enter your keyword')

    # Button for executing keyword search
    if st.button('Keyword Search'):
        if keyword:
            st.session_state['session_state']["keyword_results"] = []  # Clear previous keyword search results
            keyword_results = keyword_search(keyword, source_texts, num_results)
            st.session_state['session_state']["keyword_results"].append(keyword_results)
            # Display keyword search results
            
        if not st.session_state['session_state']["keyword_results"][0]:
            st.write('No results')
            
    for keyword_result_set in st.session_state['session_state']["keyword_results"]:
        for i, result in enumerate(keyword_result_set):
            # Replace the keyword in the result with the keyword wrapped in markdown syntax for bold and html for color
            highlighted_result = result[1].replace(keyword, f'<span style="color:red; font-weight: bold">{keyword}</span>')
            st.markdown(f'<div style="text-align: center; font-weight: bold; font-size: 120%;">Result {i+1}</div><br/>',unsafe_allow_html=True)
            st.markdown(f'{highlighted_result}', unsafe_allow_html=True)  # Enable HTML rendering


# in the second column, display AI-powered search
with col2:
    st.header('Semantic search')

    # this will create a text input box with the label 'Enter your query here:'
    input_sentence = st.text_input('Enter your query here:')
    
    # Create a progress bar
    progress_bar = st.empty()  # Placeholder for the progress bar
    
    # button for executing the search
    if st.button('AI search'):
        
        progress_button = True
        
        if input_sentence:
            st.session_state['session_state']["ai_search_results"] = []  # Clear previous AI search results
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # this calls your function and stores the result in a variable
            _, _, search_result, diag = bi_cross_semantic_search(input_sentence, source_texts, num_results, 
                                                                 nlp_model = nlp, embed_model = use_model, cross_model = cross_model, 
                                                                 redact=True, paragraph=True, p_n = paragraph_length, verbose=True, progress_bar=progress_bar)
            st.session_state['session_state']["ai_search_results"].append(search_result)
            st.write(f'{diag[0]} casenotes searched in {diag[1]:.2f} seconds')
            st.markdown("---")
            
    # Display AI-powered search results
    for ai_search_result_set in st.session_state['session_state']["ai_search_results"]:
        for i, output_text in enumerate(ai_search_result_set):
            st.markdown(f'<div style="text-align: center; font-weight: bold; font-size: 120%;">Result {i+1}</div>',unsafe_allow_html=True)

            if len(output_text)>900:
                summary = summarizer(output_text,max_length=summary_length)
                st.markdown(f'<span style="color:red; font-weight: bold">AI summary:</span> <span style="font-style: italic;">{summary[0]["summary_text"]}</span>', unsafe_allow_html=True)
            st.markdown(f'<span style="font-weight: bold">Casenote on {random_date().strftime("%B %d, %Y")}</span><br/>{output_text}', unsafe_allow_html=True)
            st.markdown("---")
        if ai_search_result_set != st.session_state['session_state']["ai_search_results"][-1]: # to prevent line after last result
            st.markdown("---")
        
        # Update the progress bar to 100 after the search is complete and results are displayed
        if progress_button:
            progress_bar.progress(100)