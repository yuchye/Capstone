# Capstone Project - Classifying clinically actionable genetic mutations

## Problem Statement

To build and train a classifier to propose the correct classification of genetic variations based on an expert-annotated knowledge base of cancer mutation annotations, so that clinical pathologists can review the medical literature to make the classification, faster and with less effort.

Fast and accurate classification of cancer mutation annotations can help to speed up diagnosis and identification of the correct treatment to deliver to affected patients.

The classifier's performance will be measured by the following:
- The best balanced F1 score: this considers both the precision and recall, and is weighted by the number of true instances of each variation class to account for class imbalance.
- The best balanced accuracy score: this metric caters for class imbalance and is the average of recall obtained on each variation class. This is especially important in our context, because we seek high recall (i.e. sensitivity) to ensure that we carry out the appropriate interventions in a timely manner, based on the  variation classification. The goal is to achieve a balanced accuracy score that is at least 10% better than the baseline accuracy, which is defined as the proportion of the majority variant class in the given training set.

---

## Executive Summary

Once sequenced, a cancer tumor can be found to have thousands of genetic mutations (or variations). The challenge is distinguishing the mutations that contribute to tumor growth versus those that do not. This interpretation of genetic mutations is currently done manually and is very time consuming. A clinical pathologist needs to manually review and classify every single genetic mutation based on evidence from text-based clinical literature. The goal of this capstone project is to develop a classifier that can help with automatic classification - this will speed up the classification process and lead to more timely interventions for cancer patients.

We obtained training and testing datasets from Kaggle (https://www.kaggle.com/c/msk-redefining-cancer-treatment/data). These datasets had missing values replaced appropriately and merged so that the clinical text, genes and variations were combined for easier processing. We observed that the training dataset was highly imbalanced with two classes taking up almost 50% of all classes.

Pre-processing was performed by performing parts-of-speech (POS) tagging on the words in the clinical text - time-consuming lemmatisation of the text was done by the NLTK WordNet lemmatiser based on these POS tags to achieve more meaningful output. One-hot encoding was then performed on the gene and variation columns. There was an attempt to identify closely correlated features that could be removed or combined with others, but unfortunately without success as such close correlations could not be found.

For the baseline model, we generated weighted word counts using the scikit-learn TfidfVectorizer, and merged them with the one-hot encoded columns created earlier. We used Synthetic Minority Oversampling Technique (SMOTE) to perform selective oversampling to address the imbalanced classes to some extent, and subsequently scaled the data with MinMaxScaler to facilitate model fitting. Hyperparameter tuning was then performed to find the best classifier, which included a forward neural network, support vector machine, logistic regression, extra trees, ADABoost, K-nearest Neighbours, random forest, decision tree and multinomial Naive Bayes classifiers. The baseline model was chosen to be the Extra Trees Classifier as it had the highest balanced accuracy score on the validation dataset. It also achieved the aim of exceeding the baseline accuracy by at least 10%.

We then explored various static word embeddings (vectors) as a potential alternative model -- these included the Global Vectors for Word Representation (GloVe) and our own word embeddings created by training NLTK's Word2Vec on all the given text in the training dataset. A combination of our own weighted Word2Vec vectors was chosen as it had the highest cross-validated accuracy score. Following the sample process to identify the baseline model, we used SMOTE, MinMaxScaler and hyperparameter tuning on the same set of candidate classifiers. The alternative model was chosen to be the Extra Trees Classifier as it once again had the highest balanced accuracy score on the validation dataset.

The baseline model - while being very large (76k+ features) and requiring significantly more memory and processing power to analyse -- delivered the better overall balanced accuracy score compared to the alternative model. To its credit, the alternative model was 20 times smaller (about 4,400 features) and could still produce a reasonably close score compared to the baseline model.

We made predictions using both the baseline and alternative model and submitted them to Kaggle to obtain the multi-loss function scores.

---

## Directory Structure
```
Capstone: Classifying clinically actionable genetic mutations
|__ code
|   |__ 01_Data_Cleaning_and_EDA.ipynb   
|   |__ 02_Preprocessing_and_EDA.ipynb   
|   |__ 03_Baseline_Model.ipynb
|   |__ 04_Alternative_Model.ipynb
|   |__ 05_Kaggle_Submission.ipynb
|__ assets
|   |__ glove.6B.50d.txt
|   |__ glove.6B.300d.txt
|   |__ sample_submission.csv
|   |__ submission.csv
|   |__ test_clean.csv
|   |__ test_pred.csv
|   |__ test_prep.csv
|   |__ test_text.csv
|   |__ test_variants.csv
|   |__ train_clean.csv
|   |__ train_prep.csv
|   |__ training_text.txt
|   |__ training_variants.txt
|   |__ tree_0.dot
|   |__ tree_0.png
|   |__ tree_50.dot
|   |__ tree_50.png
|   |__ tree_99.dot
|   |__ tree_99.png
|   |__ workflow.jpg
|   |__ tree_0.dot
|__ scores
|   |__ kaggle_score_alternative_20200406.jpg
|   |__ kaggle_score_baseline_20200406.jpg
|__ Capstone_Presentation.pdf
|__ README.md
```
---

## Data Cleaning and EDA

The following are the findings from preliminary EDA:
- The length of the descriptive text (in the training dataset) has a right-skewed distribution with mean of approx. 64,000 characters and a maximum length of approx. 526,000 characters
- The top 3 mentioned genes are:
  - BRCA1: BRCA1 is a human tumor suppressor gene and is responsible for repairing DNA. BRCA mutations increase the risk for breast cancer.
  - TP53: The tumour protein 53 gene prevents cancer formation and functions as a tumour suppressor; there is some evidence (albeit controversial) that links TP53 mutations and cancer.
  - EGFR: Mutations that lead to the overexpression of the Epidermal growth factor receptor (EGFR) protein have been associated iwth a number of cancers.
- The top 3 mentioned variations are:
  - Truncating mutations: a change in DNA that truncates (or shortens) a protein.
  - Deletions: a mutation where a part of a chromosome or a sequence of DNA is left out during DNA replication.
  - Amplification: a mutation that involves an increase in the number of copies of a gene; gene amplification is common in cancer cells.
- WordCloud and Histogram of lemmatised word frequencies reveals that there are many "common" words that may need to be removed (i.e. treated as additional stopwords). The challenge is how to find an existing curated list of such words instead of spotting words in a haphazard manner.

---

## Pre-processing

<to be written>

---

## Baseline Model

<to be written>

---

## Alternative Model

<to be written>

---

## Conclusions and Recommendations

<to be written>

### Limitations

<to be written>

### Areas for further investigation

<to be written>

---

## Data Sources

Kaggle website (https://www.kaggle.com/c/msk-redefining-cancer-treatment/data)

### Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**ID**|*int64*|training_text|The id of the row used to link the clinical evidence to the genetic mutation.|
|**Text**|*object*|training_text|The clinical evidence used to classify the genetic mutation.|
|**ID**|*int64*|training_variants|The id of the row used to link the mutation to the clinical evidence.|
|**Gene**|*object*|training_variants|The gene where this genetic mutation is located.|
|**Variation**|*object*|training_variants|The amino acid change for this mutation.|
|**Class**|*int64*|training_variants|The class (1 to 9) this genetic mutation has been classified on.|
|**ID**|*int64*|test_text|The id of the row used to link the clinical evidence to the genetic mutation.|
|**Text**|*object*|test_text|The clinical evidence used to classify the genetic mutation.|
|**ID**|*int64*|test_variants|The id of the row used to link the mutation to the clinical evidence.|
|**Gene**|*object*|test_variants|The gene where this genetic mutation is located.|
|**Variation**|*object*|test_variants|The amino acid change for this mutation.|
|**Class**|*int64*|test_variants|The class (1 to 9) this genetic mutation has been classified on.|

### Risks & Assumptions of Data Sources

- Risks:
    - The models that are created may be overfitted.

- Assumptions:
    - The 'Class' feature is not treated as an ordinal value, i.e. a class of 1 is not seen as more or less severe than a class of 2, for example.
    - Linearity: The relationship between the independent and dependent features is linear.
    - Independence: The errors are independent of one another
    - Normality:The errors between observed and predicted values (i.e., the residuals of the regression) should be normally distributed.
    - Equality of Variances: The errors should have roughly consistent pattern (i.e. there should be no disceranble relationshop between the independent features and the errors)
    - Independence: The independent features are independent of one another
