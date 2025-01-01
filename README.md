# NER-NLP-Group4

![GitHub Banner](Github%20Banner.gif)

## Project Overview
**NER-NLP-Group4** is a project focused on **Named Entity Recognition (NER)** in the field of **Natural Language Processing (NLP)**. The objective is to identify and extract NLP-related keywords from academic texts and scientific articles using state-of-the-art models and techniques. 

---

## Key Features
- **Entity Extraction**: Identifies key entities such as techniques, models, and methodologies from academic texts.
- **Model Diversity**: Implements and evaluates multiple models, providing comparative insights.
- **Custom Workflows**: Tailored workflows to meet the specific needs of NER tasks in academic contexts.

---

## Models Used
### 1. **BiLSTM_CRF (Davann)**
- Combines Bi-directional LSTM with Conditional Random Fields for sequential tagging.
- Ideal for handling sequential dependencies in text data.

### 2. **CRF (Sabos)**
- Employs Conditional Random Fields as a standalone model for sequence labeling.
- Focuses on capturing contextual relationships within text.

### 3. **DistilBERT (Noch)**
- Utilizes DistilBERT, a lightweight version of BERT, for token classification.
- Offers efficient and high-performing entity recognition.

### 4. **DistilGPT2 (Yongyi)**
- Leverages the capabilities of DistilGPT2 for generating and classifying text.
- Adapted for identifying entities within structured academic datasets.

---

## Dataset
The project is based on the **Abstract 10 Dataset**, which includes academic texts, annotated with relevant keywords. The dataset provides a robust foundation for training and evaluating NER models.

---

## Technologies and Tools
- **Programming Language**: Python
- **Core Libraries**:
  - `spaCy`
  - `NLTK`
  - `sklearn`
  - `pandas`
  - `transformers` (for DistilBERT and DistilGPT2)

---

## System Workflow
1. **Data Preparation**:
   - Load and preprocess the Abstract 10 Dataset.
   - Tokenize, clean, and annotate text data.
2. **Model Training**:
   - Train each model (BiLSTM_CRF, CRF, DistilBERT, DistilGPT2) on annotated data.
3. **Evaluation**:
   - Compare model performance using metrics such as precision, recall, and F1-score.
4. **Integration**:
   - Deploy the best-performing model into an application for academic text analysis.

---

## Contributors
Developed by **Group 4**:
- **Sreynoch Siv** (DistilBERT Model)
- **Tet Davann** (BiLSTM_CRF Model)
- **Ngang PuthSabos** (CRF Model)
- **Sok Yongyi** (DistilGPT2 Model)

---

## How to Clone the Repository

1. **Copy the Repository URL**:
   ```plaintext
   https://github.com/Nochsiv/NER-NLP-Group4.git
