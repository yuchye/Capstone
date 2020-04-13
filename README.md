# Capstone Project - Classifying clinically actionable genetic mutations

## Problem Statement

To build and train a classifier to classify genetic variations based on an expert-annotated knowledge base of cancer mutation annotations, so that clinical pathologists can review the medical literature to make the classification, faster and with less effort.

Fast and accurate classification of cancer mutation annotations can help to speed up diagnosis and identification of the correct treatment to deliver to affected patients.

The classifier's performance will be measured by the following:
- Balanced F1 score: this considers both the precision and recall, and is weighted by the number of true instances of each variation class to account for class imbalance.
- Balanced accuracy score: this metric caters for class imbalance and is the average of recall obtained on each variation class. This is especially important in our context, because we seek high recall (i.e. sensitivity) to ensure that we carry out the appropriate interventions in a timely manner, based on the  variation classification. The goal is to achieve a balanced accuracy score that is at least 10% better than the baseline accuracy, which is defined as the proportion of the majority variant class in the given training set.
- Micro-average Area under Curve (AUC): this metric looks at the average area under the Receiver Operating Characteristic (ROC) curve for each of the nine classes. The average is taken by the sum of counts to obtain cumulative metrics (true-positives, false-negatives, true-negatives and false-positives) across all classes, and then calculating the AUC.

---

## Executive Summary

Once sequenced, a cancer tumor can be found to have thousands of genetic mutations (or variations). The challenge is distinguishing the mutations that contribute to tumor growth versus those that do not. This interpretation of genetic mutations is currently done manually and is very time consuming. A clinical pathologist needs to manually review and classify every single genetic mutation based on evidence from text-based clinical literature. The goal of this capstone project is to develop a classifier that can help with automatic classification - this will speed up the classification process and lead to more timely interventions for cancer patients.

We obtained training and testing datasets from Kaggle (https://www.kaggle.com/c/msk-redefining-cancer-treatment/data). These datasets had missing values replaced appropriately and merged so that the clinical text, genes and variations were combined for easier processing. We observed that the training dataset was highly imbalanced with two classes taking up almost 50% of all classes.

Pre-processing was performed by performing parts-of-speech (POS) tagging on the words in the clinical text - time-consuming lemmatisation of the text was done by the NLTK WordNet lemmatiser based on these POS tags to achieve more meaningful output. One-hot encoding was then performed on the gene and variation columns.

For the **baseline model**, we first created (inner) training and validation datasets from Kaggle's training dataset, which left us with three datasets for training, validation and testing. For each of these three datasets, we then generated weighted word counts, performed oversampling, data scaling and feature reduction through the use of Principle Component Analysis (PCA).

Hyperparameter tuning (using a randomised search) was performed to find the best classifier for the training dataset amongst a number of candidate classifiers. The baseline model was chosen to be a Logistic Regression Classifier based on the weighted Tfidf word counts, as it had the highest balanced accuracy score of `0.540`, on the validation dataset. It also achieved the aim of exceeding the baseline accuracy of `0.287` by at least 10%.

We then explored two static word embeddings (vectors) as a potential **alternative model** -- these included the Global Vectors for Word Representation (GloVe) and our own word embeddings created by training NLTK's Word2Vec on all the given text in the training dataset. Our own mean Word2Vec embeddings were chosen as it had the highest cross-validated accuracy score on the validation dataset. Following the same process above that was used to identify the baseline model, we eventually determined that the best alternative model was a Forward Neural Network based on the mean Word2Vec word embeddings. It has a balanced accuracy score of `0.393`.

Our final choice of model was the baseline model due to its better balanced accuracy, balanced F1, and micro-average AUC scores.

For completeness, we made predictions using both the baseline and alternative model and submitted them to Kaggle to obtain the multi-loss function scores.

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
|   |__ workflow.jpg
|   |__ scores
|       |__ kaggle_score_alternative_20200406.jpg
|       |__ kaggle_score_baseline_20200406.jpg
|__ check-ins
|   |__ Part_1_Lightning_Talk.docx
|   |__ Part_2_README.md
|   |__ Part_3_Progress_Report.docx
|__ Capstone_Presentation.pptx
|__ README.md
```
---

## Data Cleaning and EDA

The following are the findings from preliminary EDA:
- The length of the descriptive text (in the training dataset) has a right-skewed distribution with mean of approx. 64,000 characters and a maximum length of approx. 526,000 characters
- The top 3 mentioned genes are:
  - BRCA1: BRCA1 is a human tumor suppressor gene and is responsible for repairing DNA. BRCA mutations increase the risk for breast cancer.
  - TP53: The tumour protein 53 gene prevents cancer formation and functions as a tumour suppressor; there is some evidence (albeit controversial) that links TP53 mutations and cancer.
  - EGFR: Mutations that lead to the overexpression of the Epidermal growth factor receptor (EGFR) protein have been associated with a number of cancers.
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
