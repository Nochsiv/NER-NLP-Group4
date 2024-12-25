# NER-NLP-Group4

![Github Banner](Github%20Banner.gif)

## Project Overview
NER-NLP-Group4 is a project focused on **Named Entity Recognition (NER)** in the field of **Natural Language Processing (NLP)**. Our goal is to identify and extract relevant NLP-related keywords from given text data, specifically targeting **academic texts** or **scientific articles**.

## Key Features
- Extracts key entities related to NLP from input text.
- Focused on processing academic texts to identify specific terms like techniques, models, or methodologies.
- Implements and evaluates NER using state-of-the-art tools and libraries.

## Dataset
The project uses **Abstract 10 Dataset** as the dataset. It contains a collection of academic texts, providing a rich source of NLP-related keywords for entity recognition.

## Tools and Libraries
- Python
- Libraries: `spaCy`, `NLTK`, `sklearn`, `pandas`

## Contributors
- Group 4

## Installation
To install the required libraries, run the following command:
```bash
pip install spacy nltk scikit-learn pandas
```
## Usage
To use the NER model, follow these steps:
```python
import spacy

# Load the pre-trained NER model
nlp = spacy.load("en_core_web_sm")

# Process the input text
text = "This is a sample text for NER."
doc = nlp(text)

# Extract the entities
entities = [(ent.text, ent.label_) for ent in doc.ents]

# Print the extracted entities
print(entities)
```
## Evaluation
To evaluate the performance of the NER model, use the following metrics:
```python
from sklearn.metrics import accuracy_score, classification_report

# Define the true labels
true_labels = ["PERSON", "ORGANIZATION", "LOCATION"]

# Define the predicted labels
predicted_labels = ["PERSON", "ORGANIZATION", "LOCATION"]

# Calculate the accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Print the classification report
print(classification_report(true_labels, predicted_labels))
```# NER-NLP-Group4

![Github Banner](Github%20Banner.gif)


## Project Overview
NER-NLP-Group4 is a project focused on **Named Entity Recognition (NER)** in the field of **Natural Language Processing (NLP)**. Our goal is to identify and extract relevant NLP-related keywords from given text data, specifically targeting **academic texts** or **scientific articles**.

## Key Features
- Extracts key entities related to NLP from input text.
- Focused on processing academic texts to identify specific terms like techniques, models, or methodologies.
- Implements and evaluates NER using state-of-the-art tools and libraries.

## Dataset
The project uses **Abstract 10 Dataset** as the dataset. It contains a collection of academic texts, providing a rich source of NLP-related keywords for entity recognition.

## Tools and Libraries
- Python
- Libraries: `spaCy`, `NLTK`, `sklearn`, `pandas`

## Contributors
- Group 4

