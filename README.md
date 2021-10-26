# EPFL Project 1 - The Higgs Boson

## INTRODUCTION

The aim of this project was to use different machine learning methods to predict the presence/absence of the Higgs boson in physics experiments. To do so, 30 features created from decay signatures were provided. Through data exploratory analysis, feature processing, feature engineering and method optimization, we were able to create a binary classifier whose predictions reach 81% accuracy.

## REPOSITORY CONTENT

The main file is ***project1.ipynb***. It contains the code for the optimization of all six machine learning methods used in the project:
- Least-squares
- Ridge-regression
- Least-squares GD
- Least-squares SGD
- Logistic regression
- Regularized logistic regression

An additional notebook called ***data_visualization.ipynb*** shows some early visualization of the parameters and corresponding labels.

All functions used are divided between 4 python files:
- ***Implementations.py*** for the principal methods which can be directly seen in the main file (pre-processing, cross-validation, ML methods).
- ***Secondary.py*** for all methods used by the principal methods but not directly in the main file (gradient/loss computation, feature creation).
- ***Plot.py*** for the functions used for any visualization.
- ***proj1_helpers.py*** for provided functions, used for data loading and submissions creation.
Every function features a commentary on its use, particularly extensive for the more convoluted ones.

The repository contains different folders: 
- ***best_models_perf*** contains lists of accuracies on the optimized versions of each method, used for models comparison using boxplots.
- ***submissions*** contains different submissions that were posted on AIRCROWD.
- ***figures*** contains different figures generated to visualize and help assessing models performance.

A file named ***run.py*** contains the code that was used to generate the submission featuring the best ranking on AIRCROWD.

The data used in this project is stored in compressed format in ***data.zip***. Note that the path used to access the data will have to be adapted.
