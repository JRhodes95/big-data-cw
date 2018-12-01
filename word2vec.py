import spacy

def feature_extraction(text):
    """Takes a token as an input and returns the associated word vector from spaCy."""
    nlp = spacy.load('en_core_web_lg')  # Load the model
    tokens = nlp(text.decode('utf8'))   # Apply the model
    output = []
    for token in tokens:
        output.append(token.vector)
    return output                # Extract the vector from the doc
