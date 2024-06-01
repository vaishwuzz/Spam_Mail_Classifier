# Email-Spam-Classifier-Using-Naive-Bayes

Naive Bayes is a supervised classification method based on Bayes' Theorem, assuming independence among predictors. In other words, a Naive Bayes classifier assumes that the occurrence of a specific feature in a class is independent of the presence of any other feature.

This technique is widely used for text categorization, determining whether documents belong to categories like spam or legitimate, sports or politics, etc., using word frequencies as features.

Goal: Accurately assign classes to previously unseen records.

We have a collection of emails categorized as either 'spam' or 'ham' (not spam). The emails are initially read and stored in a dataframe. They are then processed using CountVectorizer. This data trains the model, and its predictions are tested with sample inputs.

Python Libraries Used: pandas, numpy, io, os, CountVectorizer, and MultinomialNB from sklearn.

The spam classifier evaluates the given input and determines whether it is spam or ham. 
