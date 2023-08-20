import spacy
import pandas as pd

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 4000000


# Define function: Filter out the stop words and the punctuation and apply lemmatization
def filter_lemmatize(doc):
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text != '\n']
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# Read the JSON file into a pandas DataFrame
combined_file_path = 'material/collected_data.json'
print('Reading JSON file...')
df = pd.read_json(combined_file_path)

# Apply spaCy model to all documents and perform NER
print('Applying spaCy model and NER to all documents...')
df['spacy_document'] = df['text'].apply(nlp)
df['ner_person'] = df['spacy_document'].apply(lambda doc: [ent.text for ent in doc.ents if ent.label_ == 'PERSON'])

# Process files: Filter and lemmatize
print('Filtering and lemmatizing files...')
df['cleaned_lemmatized_filtered'] = df['spacy_document'].apply(filter_lemmatize)

# Drop the 'text' and 'spacy_document' columns
df.drop(['text', 'spacy_document'], axis=1, inplace=True)

# Print the DataFrame
print('DataFrame:')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

# Save the DataFrame to a CSV file
output_file_path = 'material/processed_data.csv'
df.to_csv(output_file_path, index=False)

print('DataFrame saved to:', output_file_path)
