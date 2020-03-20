# Capstone Project - Classifying clinically actionable genetic mutations

<div class="alert alert-block alert-warning">
<b>Things to discuss with Conor/TAs:</b>

- Need to deal with imbalanced classes?
- Is it ok to create a validation dataset to check for overfitting?
- What should my baseline model consist of?
- What are the priority improvements I should try out over my baseline model?
</div>

## Problem Statement

To build and train a classifier to propose the correct classification of genetic variations based on an expert-annotated knowledge base of cancer mutation annotations and related biomedical terms, so that clinical pathologists can spend less effort manually reviewing medical literature to make the classification.

The classifier's performance will be measured by the best accuracy and One versus One (OVO) AUC scores, and the classifier accuracy should be at least 10% better than the baseline accuracy - the baseline accuracy being defined as the proportion of the majority variant class in the given training set.

I aim to complete various milestones and deliverables as shown in the table below:

|Week|Target Milestone/Deliverables|
|:-:|:--|
|Week 7 (16 to 20 Mar)|Continue work on baseline model|
|Week 8 (23 to 27 Mar)|Complete baseline model and make 1st submission to Kaggle|
|Week 9 (30 Mar to 3 Apr)|Identify enhancements to be made to baseline model|
|Week 10 (6 to 10 Apr)|Complete alternative model, make additional submissions to Kaggle and complete analyses|
|Week 11 (13 to 17 Apr)|Complete slides and rehearse presentation|
|Week 12 (20 to 24 Apr)|Contingency|

## Proposed Methods and Models

My general approach is as follows:

1. Data Collection, Cleaning and EDA

  - Data Collection: we import the Kaggle training and testing datasets.
  - Data Cleaning: we deal with missing data in the imported datasets.
  - Preliminary EDA: we perform some EDA on the following:
    - Length of descriptive text for each variation
    - Most frequently occurring genes, variations in the training dataset
    - Frequency distribution of the classes of the variations (to detect imbalanced classes, if any)


2. Pre-processing and EDA

  - Lemmatisation: **we stick to lemmatisation** and do not explore stemming, as we wish to use a corpus to match root forms of the words found in the descriptive text.
  - EDA:
    - WordCloud for descriptive text
    - Histogram of 20 most frequent words in descriptive text


3. Modelling

  - Split data into X and y datasets
  - Creation of dummy columns
  - Creation of (inner) training and validation datasets
    - Validation dataset will be used to check for overfitting
  - Tokenisation using CountVectorizer
  - Evaluation of candidates for baseline classifier
    - Evaluate Random Forest Classifer
      - Use RandomizedSearchCV to find optimum parameters
      - Calculate accuracy scores on both training and validation datasets
      - Visualise one of the decision trees (using Graphviz)
      - Calculate AUC score based on one-versus-one algorithm
      - Calculate score on testing dataset
    - Evaluate Multinomial Logistic Regression Classifier
      - **Use StandardScaler on training dataset**
      - Use RandomizedSearchCV to find optimum parameters
      - Calculate accuracy scores on both training and validation datasets
      - Visualise coefficients
      - Calculate AUC score based on one-versus-one algorithm
      - Calculate score on testing dataset
    - Selection of baseline classifier
    - Evaluation of [FastBERT (Bidirectional Encoder Representations for Transformers)](https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384)
    - Evaluation of [BioBERT](https://arxiv.org/abs/1901.08746): a pre-trained biomedical language representation model for biomedical text mining
    - Selection of final classifier


4. Kaggle Submission

  - Using final classifier to classify testing dataset
  - Formatting results for Kaggle submission

## Data Sources

1. [Kaggle training datasets](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data):
    - "training_text": a double pipe (||) delimited file that contains 3,322 rows of clinical evidence (text) used to classify genetic mutations.
    - "training_variants": a comma separated file containing 3,322 rows of descriptions of the genetic mutations used for training.


2. [Kaggle testing datasets](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data):
    - "test_test": a double pipe (||) delimited file that contains 3,322 rows of clinical evidence (text) used to classify genetic mutations.
    - "test_variants": a comma separated file containing 2,954 rows of descriptions of the genetic mutations used for testing.


3. US National Centre for Biotechnology Information (NCBI) ClinVar [Entrez API](https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/#web):
    - Used to retrieve disease and severity information based on a given variant


4. [Biomedical Entity Search Tool (BEST)](http://best.korea.ac.kr/)
    - Used to find related biomedical terms (i.e. diseases, drugs, drug targets, transcription factors and miRNAs) related to specific genes and variants

## Data Dictionary

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

## Risks & Assumptions of Data Sources

- Risks:
    - The models that are created may be overfitted.

- Assumptions:
    - The 'Class' feature is not treated as an ordinal value, i.e. a class of 1 is not seen as more or less severe than a class of 2, for example.
    - Linearity: The relationship between the independent and dependent features is linear.
    - Independence: The errors are independent of one another
    - Normality:The errors between observed and predicted values (i.e., the residuals of the regression) should be normally distributed.
    - Equality of Variances: The errors should have roughly consistent pattern (i.e. there should be no disceranble relationshop between the independent features and the errors)
    - Independence: The independent features are independent of one another

## Summary of Preliminary EDA

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
