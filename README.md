# About
This repository contains the files used to **classify products** from "NFE" (brazilian eletronic invoice) in order to find the avarage price from a class of products and by that **find overpriced purchases**. 
There were two approaches to the same problem, both using Machine Learning algoritms, an unsupervised and a supervised oriented one (better performing, and chosen to be used in production).

# Details
- **Unsupervised Learning:** Makes use of NLP, Word2Vec, Umap and HDBSCAN to classify products in classes.
  The product implemented on the file is for fuels, which performed very good on classification and prediction, although the algorithm being addaptable to any class of product.
  This approach is very time consuming to make work, if it really works for the class chosen, and hard to use it as a product because of its non deterministic nature.
  This methdology was heavy inspired by [this repository](https://github.com/alexgand/banco-de-precos) which had the same problem and goal. 

- **Supervised Learning:** Makes use of NLP, WordCount & TF-IDF and MultinomialNB.
  The number of classes will keep increasing, and the methodogy from the classifier is still being studied and developed. 
  This file is used to create files used by a PowerBI dashboard, having many elements of the implementation oriented to that.
  Co-Author: [Alysson Casimiro] (https://github.com/Alysonson/)
