##  HORIZON GUIDED SEISMIC ATTRIBUTES TO PREDICT WELL PETROPHYSICAL PROPERTIES


##  *__seiswellattrib.py__*  


```
python seiswellattrib.py
usage: seiswellattrib.py [-h]
                         {workflow,sattrib,dropcols,prepfile,listcsvcols,sscalecols,wscalecols,wscaletarget,wattrib,wamerge,seiswellattrib,PCAanalysis,PCAfilter,scattermatrix,EDA,linreg,featureranking,linfitpredict,KNNtest,KNNfitpredict,TuneCatBoostRegressor,NuSVR,SGDR,CatBoostRegressor,ANNRegressor,testCmodels,logisticreg,GaussianNaiveBayes,QuadraticDiscriminantAnalysis,NuSVC,CatBoostClassifier,TuneCatBoostClassifier,clustertest,clustering,GaussianMixtureModel,tSNE,tSNE2,umap,DBSCAN,semisupervised}
                         ...

Seismic and Well Attributes Modeling.

positional arguments:
  {workflow,sattrib,dropcols,prepfile,listcsvcols,sscalecols,wscalecols,wscaletarget,wattrib,wamerge,seiswellattrib,PCAanalysis,PCAfilter,scattermatrix,EDA,linreg,featureranking,linfitpredict,KNNtest,KNNfitpredict,TuneCatBoostRegressor,NuSVR,SGDR,CatBoostRegressor,ANNRegressor,testCmodels,logisticreg,GaussianNaiveBayes,QuadraticDiscriminantAnalysis,NuSVC,CatBoostClassifier,TuneCatBoostClassifier,clustertest,clustering,GaussianMixtureModel,tSNE,tSNE2,umap,DBSCAN,semisupervised}
                        File name listing all attribute grids
    workflow            Workflow file instead of manual steps
    sattrib             Merge Seismic Attributes Grids
    dropcols            csv drop columns
    prepfile            Remove extra rows and columns from csv from Petrel
    listcsvcols         List header row of any csv
    sscalecols          seismic csv scale columns other than xyz
    wscalecols          well csv scale columns other than wxyz
    wscaletarget        well csv scale target column only
    wattrib             Merge Well Petrophysics csv with well xyz csv
    wamerge             Merge Well Attributes in csv format
    seiswellattrib      Back Interpolate wells at all attributes
    PCAanalysis         PCA analysis
    PCAfilter           PCA filter
    scattermatrix       Scatter matrix of all predictors and target
    EDA                 Exploratory Data Analysis
    linreg              Linear Regression Model fit and predict
    featureranking      Ranking of attributes
    linfitpredict       Linear Regression fit on one data set and predicting
                        on another
    KNNtest             Test number of nearest neighbors for KNN
    KNNfitpredict       KNN fit on one data set and predicting on another
    TuneCatBoostRegressor
                        Hyper Parameter Tuning of CatBoost Regression
    NuSVR               Nu Support Vector Machine Regressor
    SGDR                Stochastic Gradient Descent Regressor:
                        OLS/Lasso/Ridge/ElasticNet
    CatBoostRegressor   CatBoost Regressor
    ANNRegressor        Artificial Neural Network
    testCmodels         Test Classification models
    logisticreg         Apply Logistic Regression Classification
    GaussianNaiveBayes  Apply Gaussian Naive Bayes Classification
    QuadraticDiscriminantAnalysis
                        Apply Quadratic Discriminant Analysis Classification
    NuSVC               Apply Nu Support Vector Machine Classification
    CatBoostClassifier  Apply CatBoost Classification - Multi Class
    TuneCatBoostClassifier
                        Hyper Parameter Tuning of CatBoost Classification -
                        Multi Class
    clustertest         Testing of KMeans # of clusters using elbow plot
    clustering          Apply KMeans clustering
    GaussianMixtureModel
                        Gaussian Mixture Model. model well csv apply to
                        seismic csv
    tSNE                Apply tSNE (t distribution Stochastic Neighbor
                        Embedding) clustering to one csv
    tSNE2               Apply tSNE (t distribution Stochastic Neighbor
                        Embedding) clustering to both well and seismic csv
    umap                Clustering using UMAP (Uniform Manifold Approximation
                        & Projection) to one csv
    DBSCAN              Apply DBSCAN (Density Based Spatial Aanalysis with
                        Noise) clustering
    semisupervised      Apply semi supervised Class prediction

optional arguments:
  -h, --help            show this help message and exit
  
```

The program is designed with an outer main shell that has numerous subprogram to run various ML functionalities like exploratory data analysis, regression and classification.  

Classification is a bit restrictive, because it assumes the input target is all numeric and it cuts the set to any number of classes. The default is 3 classes.   

The starting point is generating attributes from the interpretation workstation along an interpreted horizon. A decision has to be made as to either extract at the horizon or even better to extract a window. The output could be any statistical property, e.g. mean, median, standard deviation, absolute value, RMS value, etc...   

The window length should be calculated in accordance with the expected reservoir thickness.  

Once he attributes are generated then each one is exported to a flat ASCII fiel with x y and value columns. The file can have header lines and/or more columns. The assumption is that all exported attributes will have the same dimensions.  

Generate a text file with the list of the exported attributes making sure that the first file is the two way time (or depth) of that horizon. This is needed to be able to import back the results of *__seiswellattrib.py__* into Petrel.

##   **__sattrib__**

```
>python seiswellattrib.py sattrib -h
usage: seiswellattrib.py sattrib [-h] [--gridheader GRIDHEADER]
                                 [--gridcols GRIDCOLS [GRIDCOLS ...]] [--ilxl]
                                 [--outdir OUTDIR]
                                 gridfileslist

positional arguments:
  gridfileslist         grid list file name

optional arguments:
  -h, --help            show this help message and exit
  --gridheader GRIDHEADER
                        grid header lines to skip. default=0
  --gridcols GRIDCOLS [GRIDCOLS ...]
                        grid columns x y z or il xl x y z. default = 2 3 4
  --ilxl                Use IL XL to fillna with mean of column Enter col
                        loctions in gridcols default= False
  --outdir OUTDIR       output directory,default= same dir as input
```  
>   The main input is the file name of the listing of the grids  
>  ``--gridheader`` specify the # of heder lines in each file to skip  
>  ``--gridcols``  column numbers of the x y and z. Please not that counting in Python starts at zero.  
>  ``--outdir``  specify the directory to send the output csv. Default is in the same working dir.  


