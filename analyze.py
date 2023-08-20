import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

def load_dataframe(file_path):
    """Load CSV file into a DataFrame."""
    df = pd.read_csv(file_path)
    # print("Loaded DataFrame:")
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(df)
    return df


def analyze_named_entity(df, named_entity):
    """Analyze the DataFrame for a given named entity."""
    results = []

    for index, row in df.iterrows():
        identifier = row['identifier']
        year = row['year']
        ner_cell_value = row['ner_person']

        # Print the ner_cell_value for troubleshooting
        # print("Named Entity List:", ner_cell_value)

        frequency = ner_cell_value.count(named_entity)

        # Print the named_entity, frequency, and row info for troubleshooting
        # print(f"Named Entity: {named_entity}, Frequency: {frequency}, Year: {year}")

        result_dict = {
            'identifier': identifier,
            'year': year,
            'count': frequency
        }

        results.append(result_dict)

    # Print the list of dictionaries after the loop
    # for result in results:
    #    print(result)

    return results


def plot_graph(data, named_entity):
    """Plot a graph using data."""
    years = [item['year'] for item in data]
    counts = [item['count'] for item in data]

    plt.figure(figsize=(10, 6))
    plt.plot(years, counts, marker='o', linestyle='', color='b')  # Plot only points
    plt.title(f'Frequency of "{named_entity}" Over Years')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.xticks(years, rotation=90)  # Rotate x-axis labels vertically
    plt.grid(True)

    # Add y-value labels next to each point
    for i, count in enumerate(counts):
        plt.text(years[i], count, str(count), fontsize=12, ha='center', va='bottom')

    plt.show()
    print("Graph printed.")


def extract_context(df, named_entity):
    analysis_data = analyze_named_entity(df, named_entity)

    complete_text = " ".join(df['cleaned_lemmatized_filtered'].dropna())

    # print("Complete Text:", complete_text)  # Print complete text for troubleshooting

    word_context = []  # List to hold individual words around the named entity
    context_window = 5  # Number of words to include before and after the named entity

    words = complete_text.split()  # Split the complete text into words

    # print("All Words:", words)  # Print all words for troubleshooting

    for i, word in enumerate(words):
        if word == named_entity:
            start_index = max(0, i - context_window)
            end_index = min(len(words), i + context_window + 1)
            context_words = words[start_index:end_index]
            word_context.extend(context_words)

    # print("Word Context:", word_context)  # Print word context for troubleshooting

    return word_context


def generate_word_cloud(named_entity, word_context):
    """Generate a word cloud based on the context words around the named entity."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_text(" ".join(word_context))

    # Display the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for "{named_entity}" Context')
    plt.show()
    print("Word Cloud generated.")


def sentiment_analysis_textblob(word_context):
    text = " ".join(word_context)  # Concatenate words into a single string
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        return f'positive ({sentiment:.2f})'
    elif sentiment < 0:
        return f'negative ({sentiment:.2f})'
    else:
        return f'neutral ({sentiment:.2f})'


def main():
    while True:
        data_file_path = 'material/processed_data.csv'
        df = load_dataframe(data_file_path)

        user_choice = input("What would you like to do? Enter 'freq' for frequency analysis, 'con' for building a "
                            "context-based word cloud, or 'sent' for sentiment analysis: ")

        if user_choice == 'freq':
            named_entity = input("Enter a named entity: ")
            analysis_data = analyze_named_entity(df, named_entity)
            plot_graph(analysis_data, named_entity)
        elif user_choice == 'con':
            named_entity = input("Enter a named entity (single word only atm): ")
            word_context = extract_context(df, named_entity)
            generate_word_cloud(named_entity, word_context)
        elif user_choice == 'sent':
            named_entity = input("Enter a named entity (single word only atm): ")
            word_context = extract_context(df, named_entity)
            sentiment = sentiment_analysis_textblob(word_context)
            print(f"The sentiment of the context words is {sentiment}.")
        else:
            print("Invalid choice. Please enter 'freq', 'con', or 'sent'.")

        repeat_choice = input("Perform another analysis? (yes/no): ")
        if repeat_choice.lower() != 'yes':
            break

if __name__ == "__main__":
    main()
