import re
import json
import numpy as np

########################### Set up Text and Sentence classes ###########################

class Text:
    def __init__(self, id, content):
        self.id = id  # A unique identifier for each text in your list
        self.content = content  # The content of your text
        self.sentences = []  # list of 'child' Sentence objects contained within the text

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def to_dict(self): #convert to a dictionary representation for storing as json
        return {
            "id": self.id,
            "content": self.content,
            "sentences": [sentence.to_dict() for sentence in self.sentences]
        }

class Sentence:
    def __init__(self, index, sentence, cleaned_sentence, vector_embedding, text_id):
        self.index = index  # sentence number within text
        self.sentence = sentence  # unprocessed sentence
        self.cleaned_sentence = cleaned_sentence  # cleaned and preprocessed sentence 
        self.vector_embedding = vector_embedding  # vector representation (depends on which pre-trained model used to populate)
        self.text_id = text_id  # parent text id

    def __str__(self):
        return f"Text {self.text_id}, Sentence {self.index}: {self.sentence}"
    
    def to_dict(self): #convert to a dictionary representation for storing as json
        return {
            "index": self.index,
            "sentence": self.sentence,
            "cleaned_sentence": self.cleaned_sentence,
            "vector_embedding": self.vector_embedding.tolist() if self.vector_embedding is not None else None,
            "text_id": self.text_id, 
        }

########################### Load and process data ###########################

def sentence_vectoriser(sentence, embed_model):
    """
    Convert a sentence into a vector representation using a pre-trained model.

    Parameters:
    sentence (str or list of str): The input sentence as a string or a list of tokenized words.
    model (tensorflow.keras.Model): A pre-trained TensorFlow model for sentence embedding.

    Returns:
    numpy.ndarray: The vector representation of the input sentence as a NumPy array.

    Raises:
    ValueError: If the input sentence is empty or contains only whitespace characters.

    Example:
    >>> from tensorflow_hub import load
    >>> use_model = load("https://tfhub.dev/google/universal-sentence-encoder/4")
    >>> sentence = "This is an example sentence."
    >>> vector = sentence_vectoriser(sentence, use_model)
    """
    # If the input sentence is a list, join the words into a single string
    if isinstance(sentence, list):
        sentence = " ".join(sentence)

    # Make sure the input sentence is not empty
    if not sentence.strip():
        print(f'Blank sentence: {sentence}')
        raise ValueError("The input sentence should not be empty or contain only whitespace")

    # Use the pre-trained model to convert the sentence into a vector representation
    vector = embed_model([sentence])  # Wrap the sentence in a list to match the model's input format

    # Return the vector representation of the sentence (as a NumPy array)
    return vector[0].numpy()

def preprocess_text(text):
    if not isinstance(text,str):
        raise ValueError("Input is not a character string")

    #remove newline indicators from pdf scrape
    cleaned_text = text.replace('\n', ' ')

    # Remove bullet points and other special characters
    cleaned_text = re.sub(r'[\u2022•\u2219]', '', cleaned_text)

    # Remove excess whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text

def load_text_objects(text_list, nlp_model, embed_model):
    """
    Take a list of text strings (text_list) and convert it into the custom Text and Sentence classes.
    
    Parameters:
    text_list (list of str): List of strings to be converted.
    nlp_model: Pre-trained NLP model for tokenising sentences.
    embed_model: Pre-trained sentence vector embeddings model.

    Returns:
    text_objects (list of Text class objects): List of processed Text objects with sentences stored as Sentence class objects.
    
    """
    text_objects = []
    
    for i, content in enumerate(text_list):
        text = Text(id=i, content=content)
        # tokenize into sentences using the chosen nlp model
        doc = nlp_model(content) 
        sentences = [sent.text for sent in doc.sents]
        
        for j, sentence in enumerate(sentences):
            cleaned_sentence = preprocess_text(sentence)
            if cleaned_sentence.strip() == "":
                continue
                
            # generate vector embeddings using choden embeddings model   
            vector_embedding = sentence_vectoriser(cleaned_sentence, embed_model) if cleaned_sentence else None
            sentence_obj = Sentence(
                index=j,
                sentence=sentence,
                cleaned_sentence=cleaned_sentence,
                vector_embedding=vector_embedding,
                text_id=text.id
            )
            text.add_sentence(sentence_obj)
        text_objects.append(text)
    return text_objects

def text_to_processed_file(in_path, out_path, nlp_model, embed_model, replace_list=None, remove_list=None): 
    """
    Read in a .txt file at 'in_path', replace any specific strings or remove other items, process using load_text_objects and save as JSON file. 
    Then return the processed Text objects
    
    Parameters:
    in_path (str): Path to a .txt file which contains rows of free text data.
    out_path (str): Path to write processed text data as JSON file.
    nlp_model
    use_model
    replace_list (list of str): List of strings to be replaced with REDACTED.
    remove_list (list of str): List of strings which, if present in a row of data, will result in that row being removed completely from the output.

    Returns:
    source_texts (list of Text class objects): List of processed Text objects with sentences stored as Sentence class objects.
    
    """
    # Read in a .txt file, replace any specific strings or remove other items, process using load_text_objects and save as json representation. 
    # Also return the processed Text objects

    text_raw = []

    # Read the file line by line
    with open(in_path, 'r') as file:
        for line in file:
            stripped_line = line.strip('\n , " ')
            if stripped_line and len(stripped_line.split()) > 5:  # filter out blanks and short sentences
                text_raw.append(stripped_line)
    
    # Process replacements and removals
    if replace_list is not None:
        # List comprehension is more efficient for larger datasets
        text_raw = [' '.join('REDACTED' if word in replace_list else word for word in i.split()) for i in text_raw]

    if remove_list is not None:
        text_raw = [i for i in text_raw if not any(word in i for word in remove_list)]
                
    #process text into text and sentence objects
    source_texts = load_text_objects(text_raw, nlp_model, embed_model)
    
    # Convert texts to dictionaries and save as JSON
    data_to_save = [text.to_dict() for text in source_texts]
    
    with open(out_path, 'w') as f:
        json.dump(data_to_save, f)
    
    return source_texts


def load_json_text_objects(json_path):
    # Load in json file created in text_to_processed_file() as list of Text objects
    
    with open(json_path, 'r') as f:
        data_loaded = json.load(f)

    texts_loaded = []
    for text_dict in data_loaded:
        text_obj = Text(text_dict['id'], text_dict['content'])
        for sentence_dict in text_dict['sentences']:
            sentence_obj = Sentence(
                sentence_dict['index'], 
                sentence_dict['sentence'], 
                sentence_dict['cleaned_sentence'], 
                np.array(sentence_dict['vector_embedding']) if sentence_dict['vector_embedding'] is not None else None,
                text_obj.id  # Pass the parent text object
            )
            text_obj.add_sentence(sentence_obj)
        texts_loaded.append(text_obj)
    return texts_loaded


def text_object_to_dict(text_obj):
    
    # Turn Text objects list created in load_text_objects() into a dictionary where the key is the index - for easy referencing later on.
    text_objects_dict = {}
    
    for text in text_obj:
        text_objects_dict[text.id] = text
        
    return text_objects_dict