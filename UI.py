import streamlit as st
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

class ModelSingleton:
    _instances = {}

    @staticmethod
    def get_instance(model_name):
        if model_name not in ModelSingleton._instances:
            ModelSingleton._instances[model_name] = ModelSingleton(model_name)
        return ModelSingleton._instances[model_name]

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        model_paths = {
            "DistilBERT": r"D:\DataScience_CADT\NLP\FinalProject\Train\NLP_Models\NLP_Models\fine_tuned_ner_model",
        }

        tokenizer_paths = {
            "DistilBERT": r"D:\DataScience_CADT\NLP\FinalProject\Train\NLP_Models\NLP_Models\fine_tuned_ner_tokenizer",
        }

        self.model = AutoModelForTokenClassification.from_pretrained(model_paths[self.model_name])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_paths[self.model_name])

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def retrain_model(self, new_data, labels):
        # Implement retraining logic for DistilBERT
        self.model.train()  # Set model to training mode
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        # Replace with actual training logic
        accuracy = 0.92  # Placeholder accuracy
        return accuracy

def analyze_and_highlight(text, model, tokenizer, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs).logits

    predictions = torch.argmax(outputs, dim=2).squeeze(0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0).cpu().numpy())

    highlighted_text = ""
    current_word = ""
    current_class = "O"

    for token, pred in zip(tokens, predictions):
        if token.startswith("##"):  # Part of a multi-token word
            current_word += token[2:]  # Append to current word without "##"
        else:
            # If we have a current word, process it
            if current_word:
                # Highlight the current word
                background_color = (
                    "lightblue" if current_class == "B"
                    else "lightgreen" if current_class == "I"
                    else "transparent"
                )
                highlighted_text += f" <span style='color:black;background-color:{background_color};border-radius:5px;padding:0.2em;'>{current_word}</span> "
                current_word = ""  # Reset for next word

            # Handle new token
            if token not in tokenizer.all_special_tokens:
                if classes[pred] != "O":
                    current_word = token  # Start a new highlighted word
                    current_class = classes[pred]  # Update class
                else:
                    highlighted_text += f" {token}"  # Normal token, no highlight
            else:
                highlighted_text += f" {token}"  # Special token, add without highlight

    # Highlight the last word if it exists
    if current_word:
        background_color = (
            "lightblue" if current_class == "B"
            else "lightgreen" if current_class == "I"
            else "transparent"
        )
        highlighted_text += f" <span style='color:black;background-color:{background_color};border-radius:5px;padding:0.2em;'>{current_word}</span> "

    # Replace spaces added unnecessarily for special tokens
    highlighted_text = highlighted_text.replace(" ##", "")
    
    return highlighted_text

# Streamlit UI
st.title("Named Entity Recognition (NER) - Group04")
st.write(" ")

# Model selection
model_name = st.selectbox("Choose Model", ["DistilBERT"])

# User input for text analysis
input_text = st.text_area("Input Text", value=" ")

# Button to trigger training and analysis
if st.button("Analyze and Train"):
    st.write("Training model with new data...")

    # Load the model
    model_singleton = ModelSingleton.get_instance(model_name)
    model = model_singleton.get_model()
    tokenizer = model_singleton.get_tokenizer()

    # Assume we have new training data (e.g., new_data and new_labels)
    new_data = input_text  # Replace with actual new data
    labels = ["O", "B", "I"]  # Example labels

    # Retrain the model
    accuracy = model_singleton.retrain_model(new_data, labels)

    # After training, highlight the input text
    classes = ["O", "B", "I"]
    highlighted = analyze_and_highlight(input_text, model, tokenizer, classes)
    st.markdown(f"<div style='font-size:16px;line-height:1.5;'>{highlighted}</div>", unsafe_allow_html=True)