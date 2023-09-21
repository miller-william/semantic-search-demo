import time
import heapq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#local module imports
from .data_setup import Text, Sentence, sentence_vectoriser, preprocess_text, load_text_objects, text_to_processed_file, load_json_text_objects, text_object_to_dict


########################### Semantic search functions ###########################

def keyword_search(keyword, text_objects, num_results=10):
    """
    Searches for a specified keyword within a collection of Text objects and returns the context 
    around the keyword (previous sentence, the sentence containing the keyword, and the next sentence).

    Parameters:
    - keyword (str): The keyword or phrase to search for.
    - text_objects (dict): A dictionary of Text objects, where the key is an identifier and the value is a Text object.
    - num_results (int, optional): Maximum number of results to return. Default is 10.

    Returns:
    - results (list of tuples): Each tuple contains the identifier of the Text object where the keyword was found 
      and the contextual sentences around the keyword. The format of the tuple is: (text_id, context).

    Note:
    - The function stops searching and returns once the desired number of results (specified by num_results) is reached.
    - If fewer than 'num_results' are found, all identified contexts are returned.
    """

    results = []
    for text in text_objects.values():
        sentences = text.sentences  # get the list of Sentence objects from the Text object
        for i, sentence_obj in enumerate(sentences):
            if keyword.lower() in sentence_obj.sentence.lower():
                # If this is the first sentence, we won't have a previous sentence
                prev_sentence = sentences[i - 1].sentence if i > 0 else ''
                # If this is the last sentence, we won't have a next sentence
                next_sentence = sentences[i + 1].sentence if i < len(sentences) - 1 else ''
                context = f"{prev_sentence} {sentence_obj.sentence} {next_sentence}"
                results.append((text.id, context))
                # If we've reached the desired number of results, stop searching
                if len(results) == num_results:
                    return results
    return results


def find_n_cosine_matches(input_vector, text_objects, n, min_length=5):
    """
    Finds the top n sentences in a list of Text objects that have the highest cosine similarity to the input.

    Args:
        input_vector (numpy.ndarray): A vector representation of the test sentence based on a pre-trained model.
        text_objects (list): A list of Text objects, each of which contains a list of Sentence objects.
        n (int): The number of top sentences to return.
        min_length (int): The minimum length of a sentence to be considered.

    Returns:
        list: A list of tuples, each containing a Sentence object and its corresponding cosine similarity score.
    """
    start_time = time.time()
    top_n = []
    counter = 0 

    # loop through all Text objects
    for text_obj in text_objects.values():
        # loop through all Sentence objects in the Text
        for sentence_obj in text_obj.sentences:
            counter += 1 
            if 'see the country policy and information note on' in sentence_obj.cleaned_sentence or len(sentence_obj.cleaned_sentence.split()) < min_length:
                continue
            score = cosine_similarity(sentence_obj.vector_embedding.reshape(1,-1), input_vector.reshape(1,-1))[0][0]
            if len(top_n) < n:
                heapq.heappush(top_n, (score, counter, sentence_obj))
            else:
                if score > top_n[0][0]:
                    heapq.heappushpop(top_n, (score, counter, sentence_obj))
    top_n = sorted(top_n, reverse=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Search time: {elapsed_time:.2f} seconds")
    print(f"Number of Texts searched: {len(text_objects)}")
    print(f"Number of sentences searched: {counter}\n")
    return top_n


def bi_semantic_search(input_sentence, source_texts, num_results, embed_model, redact=True, paragraph=False, p_n = 2, verbose=True):
    """
    Conducts a semantic search across a dictionary of Text objects using bi-encoders.

    Parameters:
    - input_sentence (str): The sentence to be searched for.
    - source_texts (dict): A dictionary of indexed Text class objects.
    - num_results (int): Number of results to return.
    - embed_model (any): Model to generate bi-encoder embeddings for the sentences.
    - redact (bool, optional): If True, redacts certain identified information. Default is True.
    - paragraph (bool, optional): If True, returns p_n sentences around the matched sentence. Default is False.
    - p_n (int, optional): Specifies the number of context sentences to return if 'paragraph' is True. Default is 2.
    - verbose (bool, optional): If True, prints search results and other information. Default is True.

    Returns:
    - top_sentences (list): List of top N scoring Sentence class objects.
    - loc (list): List of locations/indexes of top sentences.
    - all_output_text (list): List of all text segments corresponding to top sentences.
    """
    
    input_sentence_vector = sentence_vectoriser(input_sentence,embed_model)
    top_sentences = find_n_cosine_matches(input_sentence_vector, source_texts, n=num_results)

    loc = []
    all_output_text = []
    for output in top_sentences:
        loc.append([[x.index] for x in source_texts[output[2].text_id].sentences].index([output[2].index]))

    if verbose:
        print(f"Input sentence: {input_sentence}\n")
        print('RESULTS\n--------------------')

        for n,(score,counter, sentence_obj) in enumerate(top_sentences):
            if paragraph == False:
                p_n = 0
            output_text = ".".join([x.cleaned_sentence for x in source_texts[sentence_obj.text_id].sentences[loc[n]-p_n:loc[n]+1+p_n]])
            print(f'Text ID: {sentence_obj.text_id}')
            print(f'Score: {round(score,2)}\n')
            print(output_text)
            print("")
            all_output_text.append(output_text)

    return top_sentences, loc, all_output_text

# bi-encoding initial search then cross-encoder to rank

def bi_cross_semantic_search(input_sentence, source_texts, num_results, nlp_model, embed_model, cross_model, redact=True, paragraph=False, p_n = 2, verbose=True, progress_bar=None):
    """
    Performs a two-step semantic search across a collection of texts. Initially, it retrieves potential matches
    using a bi-encoder, then ranks those candidates using a cross-encoder.

    Parameters:
    - input_sentence (str): The sentence to be searched for.
    - source_texts (dict): Dictionary of indexed Text class objects to be searched.
    - num_results (int): Number of final results to return.
    - nlp_model (any): Natural Language Processing model for tokenization and other linguistic operations.
    - embed_model (any): Model to generate bi-encoder embeddings for the sentences.
    - cross_model (any): Cross-encoder model to rank initial candidates based on similarity.
    - redact (bool, optional): If True, redacts certain identified information. Default is True.
    - paragraph (bool, optional): If True, returns p_n sentences around the matched sentence. Default is False.
    - p_n (int, optional): Number of context sentences to return around the matched sentence. Default is 2.
    - verbose (bool, optional): If True, prints search results and other information. Default is True.
    - progress_bar (any, optional): Progress bar object to display progress (if applicable).

    Returns:
    - top_sentences (list): List of top N scoring Sentence class objects.
    - loc (list): List of locations/indexes of top sentences.
    - all_output_text (list): List of all text segments corresponding to top sentences.
    - [int, float]: List containing the number of source texts and the time taken for the search.
    """
    
    start_time = time.time()
    
    # Initial retrieval with bi-encoder
    input_sentence_obj = load_text_objects([input_sentence], nlp_model, embed_model)[0]
    input_sentence_vector = input_sentence_obj.sentences[0].vector_embedding
    
    initial_candidates = find_n_cosine_matches(input_sentence_vector, source_texts, n=num_results*10)  # retrieve more candidates than needed for final results
        
    # Update progress to 25% - used for streamlit app only
    if progress_bar is not None:
        progress_bar.progress(20)
    
    # Pair each initial candidate with the input sentence
    pairs = [[input_sentence_obj.sentences[0].cleaned_sentence, text.cleaned_sentence] for _,_,text in initial_candidates]

    # Get similarity scores from the cross-encoder
    cross_scores = cross_model.predict(pairs)
    #print(f'cross_scores:\n{cross_scores}')
    
    # Update progress to 50%
    if progress_bar is not None:
        progress_bar.progress(40)

    # Re-rank initial candidates with cross-encoder scores
    top_sentences = [(score,sentence) for score,sentence in sorted(zip(cross_scores, [ s for _,_,s in initial_candidates]), key=lambda x: x[0], reverse=True)][:num_results]

    # Update progress to 50%
    if progress_bar is not None:
        progress_bar.progress(60)
            
    #printing output
    loc = []
    all_output_text = []
    for output in top_sentences:
        loc.append([[x.index] for x in source_texts[output[1].text_id].sentences].index([output[1].index]))

    if verbose:
        print(f"Input sentence: {input_sentence}\n")
        print('RESULTS\n--------------------')

        if paragraph == False:
            p_n = 0

        for n,(score, sentence_obj) in enumerate(top_sentences):
            # Fetch the associated Text object using text_id
            parent_text = source_texts[sentence_obj.text_id]
            
            output_text = "".join([x.cleaned_sentence for x in parent_text.sentences[loc[n]-p_n:loc[n]+1+p_n]])
            print(f'Text ID: {sentence_obj.text_id}')
            print(f'Score: {round(score,4)}\n')
            print(output_text)
            print("")
            all_output_text.append(output_text)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
            
    # Update progress to 100%
    if progress_bar is not None:
        progress_bar.progress(80)

    return top_sentences, loc, all_output_text, [len(source_texts),elapsed_time]
