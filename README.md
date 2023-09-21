# semantic-search-demo

This repository contains the prototype for a semantic search tool that uses open-source models to search, retrieve and summarise information from large text datasets. The project is organized into a utility library, a Jupyter notebook for testing, and a Streamlit application for live demonstrations.
This current prototype matches the input search text against **each sentence** in each of the search texts.

## Structure

- **utils/**: A folder containing utility functions and classes for data loading, semantic search, and other necessary operations.
  - **data_loading.py**: Module for loading and preprocessing data.
  - **semantic_search.py**: Functions related to semantic searching.
  - **models.py**: Imports the open-source language models. 
  - **misc.py**: Other functions used for the app.
- **semantic_search_test.ipynb**: A Jupyter notebook to demonstrate the capabilities of the semantic search functions using simple examples.
- **semantic_demo.py**: A Streamlit application that showcases the semantic search functionality. 

## Getting Started

### Prerequisites

You can install the required libraries using:
```bash
pip install -r requirements.txt
```

### Jupyter notebook demo

Open `semantic_search_test.ipynb` in Jupyter.
Execute the cells to see the semantic search in action on the `text1.txt`. This is a ChatGPT generated set of 100+ free text entries. 

You can replace `text1.txt` with any of your own text data to test out the semantic search functionality. You will need to delete `text1.json` from your local area first.

You can also modify the code to run on `text2.txt`, which is a sample of 200 random Amazon food reviews.

### Running the Streamlit app

This app currently demonstrates the difference between a keyword search and semantic search for two input texts. It's been lifted from a hastily developed demo so is slightly clunky to run.

There are two example text files in the folder:

- text1.txt (a randomly generated set of text from ChatGPT)
- text2.txt (a random sample of 200 Amazon food reviews)

These will be the datasets you run the demo on. You can replace these examples but, to work seamlessly, these will need to be text data where each 'item' is on a new line.

You will need to process these first by running `process_data.py`, to create:

- text1.json
- text2.json

This is essentially a batch processing and generation of embeddings.

You can then run `streamlit run semantic_demo.py`.

## Semantic search approaches

### Sentence-Level Semantic Search
In this current prototype, the semantic search is performed at the **sentence level**. This means:

- For each source text provided, the system breaks down the content into individual sentences.
- A vector embedding is calculated for each of these sentences rather than for the entire text or paragraphs.
- The input search text is then matched semantically against each of these sentence-level embeddings.

Advantages:
- **Precision**: By comparing at the sentence level, the system can pinpoint the most relevant sections of a document with high precision.
- **Contextual Accuracy**: Sentences inherently provide a concise context, making them ideal units for semantic comparison.

Contrast with Other Approaches:
- Other systems might compute vector embeddings for entire texts or paragraphs. While this can be faster for large datasets, it might overlook nuanced matches that are evident at the sentence level. This approach should work better for casenotes data, which can vary in length and cover a wide range of topics.

The code at `utils/semantic_search.py` includes two implementations of a sentence level semantic search:

### 1. Bi-Encoder Semantic Search (bi_semantic_search)

This approach utilises a single step semantic search mechanism using bi-encoders to fetch the most semantically relevant sentences from the source texts.

Steps:

- Vectorise the input sentence using a bi-encoder model.
- Compute cosine similarities between the input sentence vector and the vectors of all sentences in the source texts.
- Retrieve the top `num_results` sentences based on the cosine similarity scores.
- If specified, provides additional context by adding `p_n` sentences around each matched sentence.

**Use Case**: Suitable for straightforward semantic matching tasks where precision and speed are critical.

### 2. Bi-Encoder followed by Cross-Encoder Ranking (bi_cross_semantic_search)
This two-step approach first retrieves potential matches using a bi-encoder and then refines and ranks those candidates using a cross-encoder.

Steps:

- Vectorise the input sentence using a bi-encoder model.
- Retrieve a larger pool of potential matches (typically 10x the desired number of final results) from the source texts based on cosine similarity scores.
- For each candidate in the initial pool, create a pair consisting of the input sentence and the candidate.
- Use a cross-encoder to compute similarity scores for each pair.
- Re-rank the initial candidates based on cross-encoder scores and return the top num_results.
- If specified, provides additional context by adding p_n sentences around each matched sentence.

**Use Case**: Offers improved precision over the single step bi-encoder approach. Useful when accuracy in ranking is more crucial than speed.

This approach has been used in the streamlit demo.

## Models used

These are imported in `utils/models.py`. Other models could be experimented with. 

### 1. **Spacy's NLP Model (`nlp`)**
- **Model**: `nlp = spacy.load("en_core_web_sm")`
  
- **Purpose**: 
  This model, from the spaCy library, is trained on web text. It's utilized for a variety of natural language processing tasks such as tokenization, part-of-speech tagging, named entity recognition, and dependency parsing.

- **Use in Search**:
  - **Text processing**: Used to tokenise the texts (split them out into sentences).

---

### 2. **Universal Sentence Encoder (`use_model`)**
- **Model**: `use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")`

- **Purpose**: 
  Developed by Google, the Universal Sentence Encoder (USE) is designed to encode sentences into fixed-size embeddings. These embeddings can then be used for semantic similarity comparisons.

- **Use in Search**:
  - **Embedding Generation**: In both search approaches, `use_model` is used to generate embeddings for input sentences and the sentences in the provided source texts. These embeddings serve as the foundation for semantic matching.

---

### 3. **Cross-Encoder (`cross_model`)**
- **Model**: `cross_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')`

- **Purpose**: 
  The cross-encoder is a type of transformer model fine-tuned for tasks where pairs of sentences (or sentence and text) are input, and it returns a relevance score. In this case, the model is specifically trained on the MS MARCO dataset with TinyBERT architecture.

- **Use in Search**:
  - **Ranking and Relevance**: In the `bi_cross_semantic_search` method, after initial matches are retrieved with the `use_model`, the `cross_model` is used to re-rank these matches by comparing the input sentence with each candidate sentence to determine their semantic relevance.

---

### 4. **Meeting Summarizer (`summarizer`)**
- **Model**: `summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")`

- **Purpose**: 
  This model, based on the HuggingFace Transformers library, is specifically fine-tuned to generate concise summaries, particularly for meetings.

- **Use in Search**:
  - **Content Summarisation**: Although not directly used in the two search approaches provided, the `summarizer` can be employed post-search to summarise the content of matched texts, providing a concise overview to users.
