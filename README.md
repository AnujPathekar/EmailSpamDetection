# Treue_Technologies_EmailSpamDetection

![EmailSpam](https://github.com/AnujPathekar/Images/blob/main/EmailSpamDetection.jpg)

# <u>Objective</u> - The objective of this project is to create an email spam detection system that uses machine learning algorithms to classify incoming emails. By training the model on a labeled dataset of spam and non-spam emails, we aim to develop an accurate and efficient spam detector that can reliably identify and categorize emails based on their content and characteristics.

# Loading Data
Firsly we loaded a dataset from a CSV file into a Pandas DataFrame named 'df.' This dataset contains 5572 rows and 5 columns. The first column 'v1' appears to contain labels, possibly indicating whether a message is 'ham' (not spam) or 'spam.' The second column 'v2' seems to contain the text messages. The remaining columns, 'Unnamed: 2,' 'Unnamed: 3,' and 'Unnamed: 4,' do not seem to have meaningful data and contain NaN values. It might be necessary to further explore and clean this dataset to use it effectively for any analysis or machine learning tasks.

# Data Cleaning

- Checked the DataFrame information using `df.info()` to understand data types and missing values.
- Identified columns 'Unnamed: 2', 'Unnamed: 3', and 'Unnamed: 4' for removal due to potential irrelevance.
- Removed leading and trailing whitespaces from column names using `df.columns.str.strip()`.
- Dropped the specified columns using `df.drop(columns=columns_to_drop, inplace=True)`.
- Renamed columns 'v1' to 'Ham/Spam' and 'v2' to 'Messages' for clarity.
- Encoded the 'Ham/Spam' column using `LabelEncoder` to convert labels to numerical values.
- Checked for and found no missing values using `df.isnull().sum()`.
- Identified and removed duplicate rows using `df.drop_duplicates(keep='first')`.
- Verified the removal of duplicates with `df.duplicated().sum()` showing 0 duplicates remaining.

# Exploratory Data Analysis

- Plotted a pie chart to visualize the distribution of 'Ham' and 'Spam' messages using `matplotlib`.
- Observed that the data is imbalanced, with more 'ham' messages than 'spam' messages.
- Downloaded the 'punkt' package from `nltk` for text processing.
- Added a new column 'Characters' to calculate the character count in each message.
- Added a new column 'NumberofWords' to calculate the number of words in each message using `nltk.word_tokenize()`.
- Added a new column 'NumberofSentences' to calculate the number of sentences in each message using `nltk.sent_tokenize()`.

These preprocessing steps help prepare the data for further analysis and modeling.

# Data Pre-Processing

- Lower Case: Converted all text to lowercase to ensure uniformity.
- Tokenization: Split the text into individual words or tokens.
- Removing Special Characters: Eliminated special characters (e.g., @, #, $) from the text.
- Removing Stop Words and Punctuation: Excluded common stop words (e.g., "the," "and") and punctuation marks.
- Stemming: Reduced words to their root forms using stemming techniques.

These data preprocessing steps were applied to prepare the text data for analysis and modeling.

# Model Building

## Data Preparation
- We began with textual input data in the 'FinalWords' column and binary output labels in 'Ham/Spam.'
- Text data needed conversion into a numerical format suitable for machine learning.

## Bag of Words (BoW) Vectorization
- Utilized `CountVectorizer` to convert text into numerical vectors.
- Applied CountVectorizer to 'FinalWords,' resulting in a (5169, 6708) matrix of features.
- Used 'Ham/Spam' as the target variable.

## Model Building
- Employed three Naive Bayes models: Gaussian Naive Bayes (GNB), Multinomial Naive Bayes (MNB), and Bernoulli Naive Bayes (BNB).
- Split the data into training and testing sets using `train_test_split` from `sklearn.model_selection`.
- Trained each model on the training data and evaluated their performance on the test data.

## Model Evaluation - BoW Vectorization
### Gaussian Naive Bayes (GNB)
- Accuracy: ~88.01%
- Precision: 0.53

### Multinomial Naive Bayes (MNB)
- Accuracy: ~96.42%
- Precision: 0.83

### Bernoulli Naive Bayes (BNB)
- Accuracy: ~97.00%
- Precision: 0.97

## TF-IDF Vectorization
- Used TF-IDF vectorization to convert 'FinalWords' into numerical vectors.
- Resulted in a (5169, 6708) matrix.

## Model Evaluation - TF-IDF Vectorization
### Gaussian Naive Bayes (GNB)
- Accuracy: ~87.62%
- Precision: 0.52

### Multinomial Naive Bayes (MNB)
- Accuracy: ~95.94%
- Precision: 1.0

### Bernoulli Naive Bayes (BNB)
- Accuracy: ~97.00%
- Precision: 0.97

These results showcase the effectiveness of Naive Bayes models for text classification, with TF-IDF vectorization providing high accuracy and precision scores, making it ideal for spam detection.

# Deployment

# Saving the TF-IDF Vectorizer and Naive Bayes Model

In this project, we've saved our TF-IDF vectorizer and Naive Bayes model for future use. This allows us to reuse the trained model without needing to retrain it every time we want to make predictions on new data.

## Saving the TF-IDF Vectorizer
- We used the TF-IDF vectorizer to convert text data into numerical vectors.
- The TF-IDF vectorizer was saved using pickle, which is a Python library for serializing and deserializing Python objects.

```python
import pickle

# Save the TF-IDF vectorizer
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
```

# Final Output
![EmailDeployment](https://github.com/AnujPathekar/Images/blob/main/Screenshot%20(11).png)

![EmailDeployment](https://github.com/AnujPathekar/Images/blob/main/Screenshot%20(12).png)

![EmailDeployment](https://github.com/AnujPathekar/Images/blob/main/Screenshot%20(13).png)

