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

We obtained training and testing datasets from Kaggle (https://www.kaggle.com/c/msk-redefining-cancer-treatment/data). These datasets had missing values replaced appropriately and merged so that the clinical text, genes and variations were combined for easier processing. Pre-processing was performed by performing parts-of-speech (POS) tagging on the words in the clinical text - time-consuming lemmatisation of the text was done by the NLTK WordNet lemmatiser based on these POS tags to achieve more meaningful output. One-hot encoding was then performed on the gene and variation columns.

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
|       |__ kaggle_score_altmodel_20200413.jpg
|       |__ kaggle_score_basemodel_20200413.jpg
|__ check-ins
|   |__ Part_1_Lightning_Talk.docx
|   |__ Part_2_README.md
|   |__ Part_3_Progress_Report.docx
|   |__ Part_4_Capstone_Checkin.pptx
|__ Capstone_Presentation.pdf
|__ README.md
```
---

## Notebook 1: Data Cleaning and Exploratory Data Analysis (EDA)

This notebook contains code for data cleaning and EDA.

We obtained training and testing datasets from Kaggle (https://www.kaggle.com/c/msk-redefining-cancer-treatment/data). These datasets had missing values which had to be replaced appropriately and merged so that the clinical text, genes and variations were combined for easier processing. We observed that the training dataset was highly imbalanced with two classes taking up almost 50% of all classes.

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
- The baseline accuracy was determined to be 0.287, which is the proportion of the data points having the majority class of '7'. Thus our models would need to have an accuracy minimally perform better than this baseline accuracy.

- Inputs: training_text.txt, training_variants.txt, test_text.csv, test_variants.csv
- Outputs: train_clean.csv, test_clean.csv
---

## Notebook 2: Pre-processing and EDA

This notebook contains code for pre-processing the clean data from Notebook 1 and performing additional EDA.

The pre-processing of the clean data began with converting the clinical text to lower case and removing all punctuation.

Word lemmatisation using the Wordnet lemmatiser was then performed on the clinical text, which is the process of converting each word into its base form. Lemmatisation considers the context and converts the word into its meaningful base form. Wordnet is a publicly available lexical database for the English language.

To improve the results of the lemmatisation, we used NLTK's parts-of-speech (POS) tagging to produce inputs to the lemmatiser. We also specified a list of stopwords that the lemmatiser should ignore as they were found to be very common in the text but have very little impact on the classification of the variations.

The complete lemmatisation of the training and testing datasets took a long time -- approx. 8 h and 15 min.

Following the lemmatisation, one-hot encoding was performed on the combined training and testing datasets to produce more than 4,300 dummy columns.

A deeper examination of the correlations between the dummy columns did not reveal any strong inter-correlations among them. The variation 'class' appeared to have stronger correlations with genes than with variations.

We performed some additional EDA on the pre-processed text in the form of a WordCloud and histogram of the lemmatised word frequencies. These revealed that there are many "common" words that can be removed (i.e. treated as additional stopwords).

- Inputs: train_clean.csv, test_clean.csv
- Outputs: train_prep.csv, test_prep.csv
---

## Notebook 3: Baseline Model

This notebook contains code to establish a baseline model that can address our problem statement of having an accuracy that is at least 10% better than the baseline accuracy of `0.287`.

The pre-processed training dataset was split into predictor (X) and target (y) dataframes. From the predictor dataframe we performed a train-test-split to create a smaller (inner) training and validation dataset based on the default 75% size for the inner training dataset.

We fit the sklearn TfidfVectorizer on the training dataset and used it to produce weighted word counts for the clinical text in the training, validation and testing datasets. The number of columns increased significantly from approx. 4,300 to just over 72,000.

After combining these word counts with the dummy columns created earlier during pre-processing, the number of columns across the datasets has risen to about 77,000.

Given that our datasets were highly imbalanced, we opted to use the adaptive sampling (ADASYN) technique to selectively oversample the 3 least frequent classes such that they had 100 samples each. We took care to generate new samples only in the training dataset to ensure that our eventual model generalises as well as possible to unseen data.

We now dealt with the issue of having too many features, which would most certainly lead to overfitting and long model training times. We scaled the data using StandardScaler, and then applied principle component analysis (PCA) for dimensionality reduction. After less than two minutes, the number of features had been reduced to the number of samples, i.e. just 2,678. Upon closer analysis of the cumulative explained variance as the number of principle components increased, we discovered that just 1,800 or so of the components would account for nearly 100% of all the explained variance. To further mitigate the issue of overfitting, we further reduced the number of features in our datasets to just the first 1,800 of the principle components.

We evaluated a broad range of multi-class classifiers -- decision tree, Gaussian Naive Bayes, Extra Trees, K-nearest Neighbours, a two hidden layer forward neural network, support vector classifier, random forest, ADABoost and multinomial logistic regression. To find the optimal parameters for each of these classifier within a reasonable amount of time (less than 4 hrs), we used the RandomisedSearchCV to find the best parameters for each based on the cross-validated accuracy score on the training dataset.

We then ranked the final classifiers based on their balanced (weighted) accuracy scores for the *validation dataset*, as we sought to find the best classifier that was the least overfitted. This was found to be the logistic regression classifier with a balanced accuracy score of `0.540`, balanced F1 score of `0.618` and a micro-average AUC score of `0.760`. Our baseline model therefore consisted of a Logistic Regression Classifier based on TfidfVectorizer weighted word counts.

We performed additional analysis of the baseline model by obtaining the top 5 more 'predictive' principle components and their respective coefficient values, for each of the 9 classes. As expected, very few principle components impacted more than one class. The ROC curves for each class, and a normalised confusion matrix showed us that our baseline model had done a reasonably good job at making accurate predictions. We also compared the frequency distributions between the actual classes and the predicted ones, showing that the relative differences in class frequency had been mostly preserved.

The baseline model was then used to generate predictions for the testing dataset, so that they could be submitted for Kaggle scoring.

- Inputs: train_prep.csv, test_prep.csv
- Output: test_pred.csv

---

## Notebook 4: Alternative Model

This notebook contains code to find an alternative model based on static word embeddings that outperforms the baseline model in terms of the balanced accuracy, balanced F1 and micro-average AUC scores.

A word embedding is a dense vector representation of words that capture their meaning in some way. Word embeddings are an improvement over simpler word encoding schemes (like TfidfVectorizer) that result in large and sparse vectors that describe documents but not the meaning of the words.

Similar to the baseline model, the pre-processed training dataset was split into predictor (X) and target (y) dataframes.

We then load/generate three static word embeddings to see if we can get better classification results:

- A smaller set of GloVe embeddings (which we call 'glove_small') that are based on based on Wikipedia 2015 and Gigaword 5th Edition (https://catalog.ldc.upenn.edu/LDC2011T07). Global Vectors for Word Representation (GloVe) are pre-trained word embeddings created by the Stanford Natural Language Processing Group and available at https://nlp.stanford.edu/projects/glove/.
- A larger set of GloVe embeddings (which we call 'glove_big') that are based on Common Crawl (https://commoncrawl.org/)
- Our own word embeddings created by training Word2Vec (from nltk) on all the given text in the training dataset, which we call 'w2v'. It is limited to 100 dimensions per sample text.

The word embeddings above represent the superset of all possible embeddings to draw from. To prepare the actual word embeddings that are specifically relevant to our clinical text, we define two 'embedding vectorizers':

- MeanEmbeddingVectorizer: takes the mean of all the 'glove_small' vectors corresponding to individual words
- TfidfEmbeddingVectorizer takes the mean of all the 'glove_small' vectors corresponding to individual words weighted based on each word's inverse document frequency.

The combination of 3 static word embeddings and 2 embedding vectorizers gave us a total of 3 x 2 = 6 possible combinations to choose from. To select the best combination, we defined each of the 6 combinations as a pipeline with an identical Extra Trees Classifier and calculated the 3-fold cross-validated mean accuracy score for each pipeline based on all the given clinical text in the combined training and validation dataset.

The result was that the mean Word2Vec word embeddings approach gave the best performance.

From the predictor dataframe we performed a train-test-split to create a smaller (inner) training and validation dataset based on the default 75% size for the inner training dataset.

We used the mean Word2Vec word embeddings to fit the training dataset, and them to then transform the training, validation and testing datasets accordingly. The end result are word embeddings that have just 100 features for each sample. After combining these word embeddings with the dummy columns created earlier during pre-processing, we have a total of 4,422 features.

We used ADASYN in the same was as the baseline model, to perform selective oversampling. The data was then scaled using StandardScaler, before we performed PCA to reduce the number of features to be the same as the number of samples, i.e. 2,675. An examination of the cumulative explained variance showed us that we could just use the first 2,300 features or so to retain about 100% of the explained variance.

As this stage, we followed the same process as the baseline model by using RandomizedSearchCV to find the optimal values for the same set of candidate multi-class classifiers.

We again ranked the final classifiers based on their balanced (weighted) accuracy scores for the *validation dataset*, and found that the best performing classifier (which we deemed to be our 'alternative' model) was our forward neural network (FNN) with a balanced accuracy score of `0.393`, balanced F1 score of `0.415` and a micro-average AUC score of `0.713`.

To perform additional analysis of our alternative model, we had to first redefine the FNN based on the optimal parameters found by the RandomizedSearchCV. We introduced an early stopping callback to determine if the FNN could be finetuned further in terms of the number of training epochs. We discovered that the training stopped very early -- at the end of epoch 2.

The ROC curves for each class, and a normalised confusion matrix showed us that our alternative model had done a somewhat poorer job at making predictions on the validation dataset, compared to the baseline model. We also compared the frequency distributions between the actual classes and the predicted ones, showing that the relative differences in class frequency did not correspond very well between the actual and predicted values for the validation dataset.

The alternative model was then used to generate predictions for the testing dataset, so that they could be submitted for Kaggle scoring.

- Inputs: train_prep.csv, test_prep.csv, glove.6B.50d.txt, glove.6B.300d.txt
- Output: test_pred.csv

---

## Notebook 5: Kaggle Submission

This notebook contains the code to format the predictions based on the testing dataset, to the format required for Kaggle submission at https://www.kaggle.com/c/msk-redefining-cancer-treatment/submit.

The predictions for the testing dataset created by either the baseline or alternative model was loaded and formatted to meet the submission template requirements for the Kaggle competition.

The baseline and alternative models achieved private KGI scores (representing multi-class loss) of 22.657 and 30.947 respectively.

- Input: test_pred.csv
- Output: submission.csv

---

## Conclusions

We have successfully built and trained a classifier to classify genetic variations based on an expert-annotated knowledge base of cancer mutation annotations.

The classifier is a Logistic Regression Classifier that relies on TfidfVectorizer weighted word counts, and has been trained on 75% of the training data provided by Kaggle. It has achieved a balanced accuracy score of 0.540, balanced F1 score of 0.618 and a micro-average AUC score of 0.760, based on our validation dataset, which is the remaining 25% of the training data provided by Kaggle. Our classifier has better performance compared to an alternative model using Word2Vec and GloVe static word embeddings. The accuracy score of 0.540 is also at least 10% better than the baseline accuracy of 0.287 which was based on the majority class in our training data.

The success of this project means that clinical pathologists have the means to speed up their classification work by using our classifier to come up with the predicted variation classes based on the clinical literature provided. The pathologists can review the predictions and this will help them to make their final classification decisions. The outcome is that patients can receive appropriate follow-up interventions (if needed) more quickly.

---

## Recommendations

Based on a comparison of the balanced accuracy, balanced F1 and micro-average AUC scores, it is clear that our final model ought to be the baseline model comprising a Logistic Regression Classifier trained on TfidfVectorizer weighted word counts.

However, overfitting remains a concern even though we have managed to get a reasonably good model through a significant reduction in features, facilitate by PCA.

We would recommend the following to mitigate the issue of overfitting and potentially improve the performance of our model:

1. **Obtain more samples**, especially for imbalanced classes. Instead of having to rely on techniques such as ADASYN to oversample the existing data, it would be better to obtain more samples for the minority classes.
2. Explore the use of **Long Short-term Memory (LSTM)** units within our neural network classifier. This should intuitively improve the performance of our neural network as there is useful contextual information behind the use of words in each sample's clinical text, especially since the clinical text is very long.
3. Explore the use of **contextual word embeddings**. The static word embeddings we evaluated (i.e. Word2Vec and GloVe) are generated for each word in the vocabulary. For example, the word "express" would have the same context-free representation in in "express delivery" and "gene express[ion]". In contrast, contextual word embeddings for each word are based on the other mentions of the word in the same clinical text. In particular, it would be beneficial to explore the following:

  - *Bidirectional Encoder Representations from Transformers (BERT)* - BERT is based on a multi-layer bidirectional transformer-encoder where a transformer neural network uses parallel attention layers rather than sequential recurrence. It is trained on the BooksCorpus dataset (800M words) and text passages of English Wikipedia. There is a limit of 1,024 words per document/sentence that BERT can analyse and it is unclear how best to apply BERT to our specific scenario given that the non-empty clinical text are all much longer than 1,024 words.
  - *BioBERT* - BioBERT is based on the initial BERT language model and pre-trained on PubMed abstracts and PubMed Central (PMC) full-text articles. It is likely to be a good candidate for our problem statement as  BioBERT would have word embeddings relevant to the biomedical literature we have to classify.
  - *Embedding from Language Models (ELMo)* - ELMo looks at the entire sentence as it assigns each word an embedding. It uses a bidirectional recurrent neural network (RNN) trained on a specific task to create the embeddings. The use of ELMo requires substantial computing and memory resources, which may warrant the use of powerful cloud computing resources from the likes of Amazon Web Services, for example.
4. **Look up related words** based on the genes and variations given in the Kaggle training and testing datasets, and increase the weights manually in our embeddings or word counts. This look-up could be done via the following:
  - *API queries to ClinVar*: ClinVar is a freely accessible, public archive of reports of the relationships among human variations and phenotypes, with supporting evidence. ClinVar facilitates access to and communication about the relationships asserted between human variation and observed health status, and the history of that interpretation. The curator of ClinVar - US National Centre for Biotechnology Information (NCBI)’s - provides an application programming interface (API) ((https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/#web) that can allow us to retrieve disease and severity information based on specific genes or variations.
  - *Web scraping of the online Biomedical Entity Search Tool (BEST)* (http://best.korea.ac.kr/): using BEST, we can obtain  entities (diseases, drugs, targets, transcription factors, miRNAs) related to specific genes and variants that we manually provide as inputs; we then adjust the weights of any of these entities that are found in our training dataset.

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
    - The data sources may contain many words that play little/no role in affecting what classification of the clinical text. At present, we have specified only a relatively small number of stopwords that are ignored during the text lemmatisation. The risk is that we continue to retain redundant words that increase the likelihood of overfitting.


- Assumptions:
    - The 'Class' feature is not treated as an ordinal value, i.e. a class of 1 is not seen as more or less severe than a class of 2, for example.
    - Linearity: The relationship between the independent and dependent features is linear.
    - Independence: The errors are independent of one another.
    - Normality: The errors between observed and predicted values (i.e., the residuals of the regression) should be normally distributed.
    - Equality of Variances: The errors should have roughly consistent pattern (i.e. there should be no disceranble relationshop between the independent features and the errors).
    - Independence: The independent features are independent of one another.
