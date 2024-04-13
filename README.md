# Project Report: Text Classification

## Step 1: Data Preprocessing

### Data Sampling and Splitting
- **Data Sampling**: A subset of data from one sample category was selected for analysis.
- **Data Splitting**: The sampled data was split into training and testing sets.

### Class Distribution Visualization
- **Imbalanced Data**: The class distribution in both `y_train` and `y_test` revealed an imbalanced dataset, suggesting potential challenges in model training and evaluation.

### Preprocessing Steps
1. **Text Cleaning**:
   - Removal of extra spaces and period elements.
   - Replacement of consecutive whitespaces with a single space.
2. **Text Normalization**:
   - Conversion to lowercase.
   - Tokenization, removal of special characters, and stop words.
   - Lemmatization of words to their base form.

### Vocabulary Creation and Text Saving
- **Vocabulary Set**: Extracted from the preprocessed text data, removing duplicate words.
- **Train Data**: Preprocessed and cleaned data was saved for further use.

## Step 2: Preparing Data for Modeling

### TF-IDF Encoding
- **TF-IDF Vectorizer**: Created and fitted to the documents.
- **TF-IDF Vectors**: Transformed the documents into TF-IDF vectors.
- **Data Saving**: TF-IDF vectors were saved as npz files.

### Word2Vec Embeddings
- **Word2Vec Model Training**: Trained with the train data to generate word embeddings.
- **Normalization**: Document vectors were normalized using Word2Vec embeddings.
- **Data Saving**: Normalized data saved as CSV files.

### GloVe Embeddings
- **GloVe Embeddings Loading**: Loaded pre-trained GloVe embeddings.
- **Word Mapping**: Mapped words in the vocabulary to GloVe embeddings.
- **Vector Creation**: Created vectorized train and test data using GloVe embeddings.
- **Data Saving**: Vectorized data saved as npz files.

## Step 3: Modeling

### Naive Bayes
- **TF-IDF Data**: Trained and evaluated naive bayes model.
- **Word2Vec Data**: Applied naive bayes model with Word2Vec embeddings.
- **GloVe Data**: Utilized naive bayes model with GloVe embeddings.

### Random Forest
- **TF-IDF Data**: Trained and evaluated random forest model.
- **Word2Vec Data**: Applied random forest model with Word2Vec embeddings.
- **GloVe Data**: Utilized random forest model with GloVe embeddings.

### Support Vector Machine (SVM)
- **TF-IDF Data**: Trained and evaluated SVM model.
- **Word2Vec Data**: Applied SVM model with Word2Vec embeddings.
- **GloVe Data**: Utilized SVM model with GloVe embeddings.

### LSTM
- **Word2Vec Data**: Trained LSTM model with Word2Vec embeddings.
- **GloVe Data**: Utilized LSTM model with GloVe embeddings.

## Results
The performance of each model was evaluated using the macro-averaged F1-score. Below are the F1-scores for each model:

- Naive Bayes: 
  - In hand: 0.192
  - With TF-IDF: 0.046
  - With Word2Vec: 0.144
  - With GloVe: 0.211
- Random Forest: 
  - With TF-IDF: 0.160
  - With Word2Vec: 0.171
  - With GloVe: 0.133
- SVM: 
  - With TF-IDF: 0.327
  - With Word2Vec: 0.255
  - With GloVe: 0.288
- LSTM: 
  - With Word2Vec: 0.104
  - With GloVe: 0.086

## Conclusion
In this project, I explored various preprocessing techniques, feature encodings, and modeling approaches for text classification. Despite the challenges posed by imbalanced data, I achieved results with SVM using TF-IDF encoding. Further improvements could be made by fine-tuning model hyperparameters and exploring ensemble methods. Additionally, leveraging deep learning architectures like LSTM showed potential but requires more data and computational resources for effective training.