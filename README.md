# HorizonMachineLearning
Machine Learning Horizon Guided attributes for reservoir properties prediction

### General Concept
*  Seismic attributes are extracted along interpreted horizons
    * A window of, let's say 20 ms, is used for the extraction of various attributes. 
    * These attributes are then exported to an x y z flat file. Let's say we extract 10 attributes.
*  Well petrophysical data, e.g. porosity, permeability or net to gross:
    *  These are upscaled to the equivalent of the 20 ms window and supplied as a csv file.
    *  A dataframe (a table) with all the attributes is generated:
    *  Each attribute is listed in a column while the rows represent the individual locations, 
    i.e. trace locations
    
### Workflow I - Data Munging:
*  Format the horizon files into one file
*  Scale the horizon data
*  Create a well file with all the attributes back interpolated at the well locations   
with the last column being the petrophysical attribute, e.g. permeability
*  We are ready to apply Machine Learning

### Workflow II - Data Analysis
*  Check data distributions and statistical ranges
*  Check for linearity between various predictors amongst themselves and with the target
    *  Generate a *matrix scatter plot*
*  Check for feature importance using using *__RFE (Recursive Feature Elimination)__*
*  Check for feature contribution using *__PCA (Principle Component Analysis)__*

### Workflow III - Model Fitting and Prediction:
*__swattriblist.py__* has many models to attempt to fit to your data they are all based on sklearn package.  
**CatBoost** is installed and used instead of **XGBOOST**

####  Clustering  
*  *__KMEANS__* is first tested to identify the optimum number of clusters  
*  Once the optimum of clusters are found then *__KMEANS__* is applied to the predictors
*  The resulting clusters are then one hot encoded to be added as predictors for further model fitting
    
####  Regression
    Below are the various regression techniques that can be applied  
*  *__Linear Regression__*
*  *__SGDR__* : Stochastic Gradient Descent with *Lasso*, *Ridge*, and *ElasticNet* options
*  *__KNN__* : K Nearest Neighbors.   
*  *__CatBoostRegression__*      
*  *__NuSVR__*  Support Vector Machines Regression

###  Classification
   _Below are various classification models that can be used:_
*  *__LogisticRegression__*
*  *__GaussianNaiveBayes__*
*  *__CatBoostClassification__*
*  *__NuSVC__*  Support Vector Machines Classification  
*  *__QDA__*   Quadratic Discriminant Analysis
*  *__GMM__*   Gaussian Mixture Model

###  SemiSupervised Learning  

###  Imbalanced Classification  
_Most of our data is imbalanced. These correction techniques apply to all classification models_
*  *__ROS__*     Random oversampling
*  *__SMOTE__*   Synthetic Minority Oversampling
*  *__ADASYN__*  Adaptive Synthetic Sampling



