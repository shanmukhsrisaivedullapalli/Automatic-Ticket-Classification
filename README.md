# Automatic Ticket Classification

This project implements a machine learning-based pipeline to classify customer complaints into predefined categories using Natural Language Processing (NLP) and a neural network model. The aim is to automate the classification of complaints into topics like "Bank Account Services" or "Mortgage/Loan," enhancing customer service efficiency.

## Features
- Preprocessing of textual data, including:
  - Text cleaning (lowercasing, punctuation removal, etc.).
  - Lemmatization using SpaCy.
  - Part-of-Speech (POS) filtering.
- Visualization of word frequencies and n-grams (uni-grams, bi-grams, tri-grams).
- Topic modeling using NMF (Non-Negative Matrix Factorization).
- Deep learning-based classification using TensorFlow/Keras.
- Real-time predictions for new complaints.

## Data Source
- The dataset is loaded from a JSON file containing customer complaints and associated categories.

## Requirements
- Python 3.8+
- Jupyter Notebook or a Python environment
- Key Libraries:
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - SpaCy
  - TensorFlow
  - Scikit-learn
  - WordCloud
  - TQDM

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/automatic-ticket-classification.git
   cd automatic-ticket-classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the SpaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. Place the dataset file (https://www.kaggle.com/datasets/venkatasubramanian/automatic-ticket-classification) in the `input/` directory.

## Usage
1. Open the `Automatic Ticket Classification` Jupyter Notebook:
   ```bash
   jupyter notebook automatic_ticket_classification.ipynb
   ```

2. Run the cells step by step to:
   - Preprocess the data.
   - Train the classification model.
   - Generate visualizations.
   - Make predictions.

3. To predict a topic for new complaints, use the pre-trained model and follow the steps in the notebook.

## Model Architecture
The deep learning model is built using TensorFlow/Keras:
- Input layer with 128 neurons.
- Hidden layers with ReLU activation and dropout regularization.
- Output layer with a softmax function for multi-class classification.

## Visualizations
- N-gram frequency plots.
- Word clouds for complaint text.
- Topic-wise distribution of complaints.

## Example Prediction
```python
text_sample = "I want a loan of rupees 400000 from Axis bank"
predicted_topic = predict(text_sample)
print(f"Predicted Topic: {predicted_topic}")
# Output: "Mortgage/Loan"
```

## Results
- **Accuracy**: Achieved high accuracy on the test dataset.
- **Topics**:
  - Bank Account Services
  - Credit Card or Prepaid Card
  - Others
  - Theft/Dispute Reporting
  - Mortgage/Loan

## Contributing
Contributions are welcome! Submit a pull request or raise an issue to discuss potential improvements.
