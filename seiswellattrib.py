'''
swattriblist.py ver1.01
Program to create :
    -seismic attributes dataframe from many grids
    -well attributes from petrophysical data
    -merge both seismic and well attributes into one dataframe
    -Linear Regression model fitting and prediction
    -Cluster analysis on either seismic or well attributes

    **sattrib option:
        -assume every attribute is in a seperate space delimited xyz file
        -enter file name with list of these xyz files
        -program will merge them and write out a csv file
        >python swattrib.py sattrib  gridlist.txt --gridcols 2 3 4

    **dropcols option:
        -works on csv files only
        -to drop columns by their number location
        >python swattrib.py dropcols gridlist_merged_unscaled.csv --cols2drop 3
    **prepfile option:
        -for csv files from Petrel, remove the top 2 rows and the leftmost column
        -it can be used to clean up outputs from petrel spreadsheets containing either
            back interpolated attributes in petrel or depth or whatever property extracted form PETREL
        -input csv file

    **listcsvcols option:
        -lists enumerated columns of any csv
        -expects on input csv file


    **sscalecols option:
        -works on csv files only
        -drop xyz columns then scale data then add xycolumns
        >python swattrib.py  sscalecols sgrids1_merged_unscaled.csv --xyzcols 0 1 3


    **wscalecols option:
        -works on csv files only
        -drop wxyz columns then scale data then add wxyzcolumns

    **wamerge option:
        -works on csv files
        -need 2 csv files: petrophysical attributes and wells xyz
        -read well name petrophysical parameters, e.g. porosity, density, sw, net/gross, ....
        -automatically removes nan values
        -option to fill nan with either mean or median of values or delete complete row
        >python swattrib.py wamerge i2pp.csv i2well.csv --wattribcols 0 7 --wxyzcols 0 3 4 5

    **wattrib option:
        -assume every attribute is in a seperate csv "well x y z attribute" file
        -enter file name with list of these csv well x y z attribute files
        -program will merge them and write out a csv file
        -this case is for interpolating in Petrel outputting csv file per attribute
        >python swattrib.py wattrib  csvlist.txt --csvcols 2 3 4

    **seiswellattrib option:
        -expect 2 csv files: seismic and wells
        -seismic has many columns representing attributes
        -wells has 4 columns representing well name, x, y, and attribute, e.g. porosity
        -option then back interpolates all seismic attributes and saves a csv well attribute file
        -this should then be used to create a prediction model
        >python swattrib.py seiswellattrib gridlist_scaled.csv  i2pp_wa.csv

    **PCAanalysis option:
        -csv file input with all aatributes
        -a plot is generated showing components and their respective variances
        >python swattrib.py PCAanalysis I2pp_wa_swa.csv --analysiscols 4 5 6 7 8 9 10 11 12 13 14 15 16 --acolsrange 4 16
        >python swattrib.py PCAanalysis I2pp_wa_swa.csv  --acolsrange 4 16
        >python swattrib.py PCAanalysis I2pp_wa_swa.csv --acolsrange 4 16


    **PCAfilter option:
        -csv input with all attributes
        -need number of components to keep
        -enter column range or column numbers to use
        -output a filtered csv file
        >python swattrib.py PCAfilter  I2pp_wa_swa.csv --acolsrange 4 16 --targetcol 17 I2pp_wa_swa.csv
        >python swattrib.py PCAfilter I2pp_wa_swa.csv --acolsrange 4 16 --ncomponents 6
            --cols2addback 0 1 2 3

    **qclin option:
        -expect one csv with all seismic and well attributes
        -supply wellxyz columns to remove before analysis
        -xplots every seismic attribute (predictor) versus the well attribute (target)
        -saves plots to pdf


    **scatterplot option:
        -csv with all attributes
        -specify columns to ignore, e.g. well name, x, y, z

    **linreg option:
        -same input csv file as qclin
        -supply wellxyz column #
        -fits model to all data and prints coefficients, MSE, and R2 score
        -predicts all data

    **linfitpredict option:
        -fit a model to input data set e.g. wells attributes and porosity
        -predict on a seismic attribute data set
        -output can be scaled to range of the input
        -2 csv files are expected: one for wells with target attribute, other of seismic
        -enter cols range for both seismic attributes and well attributes
        >python swattrib.py linfitpredict wattrib.csv sattrib.csv --wcolsrange 4 16 --scolsrange 3 15


    **KNNtest option:
        -same input csv file as qclin
        -supply wellxyz column #
        -splits data to training and testing sets
        -does cross validation
        -fits model to all data and prints coefficients, MSE, and std of errors
        -predicts all data
        -generates a plot of # of clusters vs errors
        >python swattrib.py KNNtest I2pp_wa_swa.csv --wcolsrange 4 16


    **KNNfitpredict option:
        -fit a model to input data set e.g. wells attributes and porosity
        -predict on a seismic attribute data set
        -output can be scaled to range of the input
        -2 csv files are expected: one for wells with target attribute, other of seismic
        >python swattrib.py  KNNfitpredict I2pp_wa_swa.csv i2g_sus_ss.csv
        >python swattrib.py  KNNfitpredict I2pp_wa_swa.csv i2g_sus_ss.csv --minmaxscale
        >python swattrib.py KNNfitpredict I2pp_wa_swa.csv i2g_sus_ss_spred.csv --wtargetcol 17
            --wpredictorcols 4 5 6 7 8 9 10 11 12 13 14 15 16 --spredictorcols 3 4 5 6 7 8 9 10 11 12 13 14 15
            --kneighbors 11
        >python swattrib.py KNNfitpredict I2pp_wa_swa.csv i2g_sus_ss_spred.csv --wpredictorcols 4 16
            --spredictorcols 3 15 --kneighbors 11

    **testCModels
        -test classification model: Logistic regression, linear discriminant analyses
            Naive Bayes, KNN, decision tree classifier (CART), and support vector machines
        -lists accuracy metric
        -performs cross validation but is not exhaustive.


    **logisticreg option:
        -fit a model to input data set e.g. wells attributes and porosity
        -predict on a seismic attribute data set
        -cut range of input data to 3 quantiles -> results in 3 classes
        -output probabilities per class -> should be mapped
        -2 csv files are expected: one for wells with target attribute, other of seismic
        -2 output files are generated one seismic with class predictions and wells with class ranges
        >python swattrib.py logisticreg I2pp_wa_swa.csv i2g_sus_ss_spred.csv  \
            --wcolsrange 4 16 --scolsrange 3 15 --wtargetcol 17

    **GaussianNaiveBayes option
        -fit Gaussian Naive Bayes classification model to well csv
        -input target column. default is last column
        -qcut: quantile breakup of target column, default = 3 i.e. low, mid high
        -input column range of attributes, excluding well, x, y, z and target columns
        -input  seismic attributes csv file
        -input column range of attributes
        -2 output files are generated one seismic with class predictions and wells with class ranges
        >python swattrib.py GaussianNaiveBayes I2pp_wa_swa.csv i2g_sus_ss_spred.csv
            --wcolsrange 4 16 --scolscolsrange 3 15 --wtargetcol 17

    **clustertest option:
        -use all attributes csv file generated above
        -runs up to 20 clusters and check their inertia
        -generates a plot and saves it to a pdf file
        >python swattrib.py clustertest i2pp_wa_swattrib.csv

    **clustering option:
        -use all attributes csv file generated above
        -uses KMeans to perform clustering
        -generates an extra column of clusters to the input csv file
        -default clusters is 5
        >python swattrib.py clustering  i2pp_wa_swattrib.csv  --nclusters 6

    **tSNE option:
        >python swattrib.py tSNE i2_sus_ss.csv --colsrange 3 15  --sample 0.05



    Usage: python swattrib.py sattrib | sscalecols | wscalecols | dataprep | listcsvcols
        drop | wattrib | wamerge | seiswellattrib  | scatterplot | qclin  |  linreg  | testCmodels
         | logisticreg | GaussianNaiveBayes | clustertest | clustering |  tSNE

python swattriblist.py GaussianMixtureModel wattrib.csv  --wcolsrange 4 16 --ncomponents 4

python swattriblist.py ANNRegressor  wattrib.csv sattrib.csv --wcolsrange 4 16 --scolsrange 3 15 --nodes 20 5 --activation relu relu

python swattriblist.py NuSVR  wattrib.csv sattrib.csv --wcolsrange 4 16 --scolsrange 3 15 NuSVR

python swattriblist.py tSNE H763_sus_ss.csv --colsrange 4 9 --targetcol 9

python swattriblist.py CatBoostClassifier  H763_sus_ss_H763_swa.csv H763_sus_ss.csv --wcolsrange 4 9 --scolsrange 3 8 --coded --balancetype ros

python swattriblist.py logisticreg  H763_sus_ss_H763_swa.csv H763_sus_ss.csv --wcolsrange 4 9 --scolsrange 3 8 --coded --balancetype ros

python swattriblist.py GaussianNaiveBayes  H763_sus_ss_H763_swa.csv H763_sus_ss.csv --wcolsrange 4 9 --scolsrange 3 8 --coded --balancetype ros


python swattriblist.py NuSVR  wattrib.csv sattrib.csv --wcolsrange 4 16 --scolsrange 3 15 --generatesamples 10



python swattriblist.py NuSVC H763_sus_ss_H763_swa.csv H763_sus_ss.csv --wcolsrange 4 9 --scolsrange 3 8 --coded --generatesamples 20

python swattriblist.py NuSVC H763_sus_ss_H763_swa.csv H763_sus_ss.csv --wcolsrange 4 9 --scolsrange 3 8 --coded --generatesamples 20 --balancetype ros

python swattriblist.py CatBoostClassifier H763_sus_ss_H763_swa.csv H763_sus_ss.csv --wcolsrange 4 9 --scolsrange 3 8 --coded --generatesamples 20

python swattriblist.py GaussianNaiveBayes H763_sus_ss_H763_swa.csv H763_sus_ss.csv --wcolsrange 4 9 --scolsrange 3 8 --coded --generatesamples 20 --balancetype ros

python swattriblist.py logisticreg H763_sus_ss_H763_swa.csv H763_sus_ss.csv --wcolsrange 4 9 --scolsrange 3 8 --coded --generatesamples 20

python swattriblist.py KNNfitpredict wattrib.csv sattrib.csv --wcolsrange 4 16 --scolsrange 3 15 --kneighbors 4 --generatesamples 20

python swattriblist.py KNNfitpredict wattrib.csv sattrib.csv --wcolsrange 4 16 --scolsrange 3 15 --kneighbors 4

python swattriblist.py tSNE  sattrib.csv --colsrange 3 15 --sample .2 --colorby 0 1 2 3 4

python swattriblist.py tSNE wattrib.csv --colsrange 4 16 --sample 1 --hideplot --xyzcols 0 1 2 3


python swattriblist.py tSNE2 wattrib.csv sattrib.csv  --wcolsrange 4 16 --scolsrange 3 15 --sample 0.005

D:\sekData\OriginalData\sekdata\Tukau Timur\horizons\Mar29>python swattriblist0.py tSNE2 H763_sus_ss_H763_swa.csv H763_sus_ss.csv --scolsrange 3 8 --w
colsrange 4 9 --sample 0.05

'''
import os.path
import argparse
import shlex
# import datetime
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as sts
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import griddata,Rbf,LinearNDInterpolator,CloughTocher2DInterpolator
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import seaborn as sns
from collections import Counter
import itertools


from pandas.tools.plotting import scatter_matrix
# ->deprecated
# from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler,QuantileTransformer
# from sklearn import cross_validation
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC,NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import mixture
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import NuSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import r2_score
# Coefficient of Determination
from sklearn.cluster import KMeans,DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_samples
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.manifold import TSNE
# from sklearn.svm import SVR
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading
try:
    from catboost import CatBoostRegressor
    from catboost import CatBoostClassifier
except ImportError:
    print('***Warning:CatBoost is not installed')

try:
    from imblearn.over_sampling import SMOTE, ADASYN,RandomOverSampler
except ImportError:
    print('***Warning:imblearn is not installed')

try:
    import umap
except ImportError:
    print('***Warning: umap is not installed')

def module_exists(module_name):
    """Check for module installation."""
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True

def plot_classifier(classifier, X, y,xcol0=0,xcol1=1):
    """Plot Classification curves: ROC/AUC."""
    # define ranges to plot the figure
    x_min, x_max = min(X[:, xcol0]) - 1.0, max(X[:, xcol0]) + 1.0
    y_min, y_max = min(X[:, xcol1]) - 1.0, max(X[:, xcol1]) + 1.0

    # denotes the step size that will be used in the mesh grid
    step_size = 0.01

    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot
    plt.figure()

    # choose a color scheme you can find all the options
    # here: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    # Overlay the training points on the plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1]) - 1), int(max(X[:, 1]) + 1), 1.0)))
    plt.show()

def qhull(sample):
    """Compute convex hull."""
    link = lambda a,b: np.concatenate((a,b[1:]))
    edge = lambda a,b: np.concatenate(([a],[b]))

    def dome(sample,base):
        h, t = base
        dists = np.dot(sample - h, np.dot(((0,-1),(1,0)),(t - h)))
        outer = np.repeat(sample, dists > 0, 0)
        if len(outer):
            pivot = sample[np.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                        dome(outer, edge(pivot, t)))
        else:
            return base

    if len(sample) > 2:
        axis = sample[:,0]
        base = np.take(sample, [np.argmin(axis), np.argmax(axis)], 0)
        return link(dome(sample, base),dome(sample, base[::-1]))
    else:
        return sample

def pip(x,y,poly):
    """Point In Polygon."""
    # if (x,y) in poly:
    #     return True
    # check if point is on a boundary
    for i in range(len(poly)):
        p1 = None
        p2 = None
        if i == 0:
            p1 = poly[0]
            p2 = poly[1]
        else:
            p1 = poly[i - 1]
            p2 = poly[i]
        if p1[1] == p2[1] and p1[1] == y and x > min(p1[0], p2[0]) and x < max(p1[0], p2[0]):
            return True

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n + 1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    if inside:
        return True
    else:
        return False

class Invdisttree:
    """inverse-distance-weighted interpolation using KDTree.

    invdisttree = Invdisttree( X, z )  -- data points, values
    interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
    interpolates z from the 3 points nearest each query point q;
    For example, interpol[ a query point q ]
    finds the 3 data points nearest q, at distances d1 d2 d3
    and returns the IDW average of the values z1 z2 z3
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

    q may be one point, or a batch of points.
    eps: approximate nearest, dist <= (1 + eps) * true nearest
    p: use 1 / distance**p
    weights: optional multipliers for 1 / distance**p, of the same shape as q
    stat: accumulate wsum, wn for average weights

    How many nearest neighbors should one take ?
    a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
    b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
    |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
    I find that runtimes don't increase much at all with nnear -- ymmv.

    p=1, p=2 ?
    p=2 weights nearer points more, farther points less.
    In 2d, the circles around query points have areas ~ distance**2,
    so p=2 is inverse-area weighting. For example,
        (z1/area1 + z2/area2 + z3/area3)
        / (1/area1 + 1/area2 + 1/area3)
        = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
    Similarly, in 3d, p=3 is inverse-volume weighting.

    Scaling:
        if different X coordinates measure different things, Euclidean distance
        can be way off.  For example, if X0 is in the range 0 to 1
        but X1 0 to 1000, the X1 distances will swamp X0;
        rescale the data, i.e. make X0.std() ~= X1.std() .

    A nice property of IDW is that it's scale-free around query points:
    if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
    the IDW average
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
    is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
    In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
    is exceedingly sensitive to distance and to h.
    # anykernel( dj / av dj ) is also scale-free
    # error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim
    """

    def __init__(self, X, z, leafsize=10, stat=0):
        """Init inverse distance class."""
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        # build the tree
        self.tree = KDTree(X, leafsize=leafsize)
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None

    def __call__(self, q, nnear=6, eps=0, p=1, weights=None):
        """# nnear nearest neighbours of each query point --."""
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query(q, k=nnear, eps=eps)
        interpol = np.zeros((len(self.distances),) + np.shape(self.z[0]))
        jinterpol = 0
        for dist, ix in zip(self.distances, self.ix):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot(w, self.z[ix])
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1 else interpol[0]

def idw(xy,vr,xyi):
    """Inverse Distance Weighting."""
    # N = vr.size
    # Ndim = 2
    # Nask = N  # N Nask 1e5: 24 sec 2d, 27 sec 3d on mac g4 ppc
    nnear = 8
    # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
    leafsize = 10
    eps = .1
    # approximate nearest, dist <= (1 + eps) * true nearest
    p = 2
    # weights ~ 1 / distance**p
    invdisttree = Invdisttree(xy, vr, leafsize=leafsize, stat=1)
    interpol = invdisttree(xyi, nnear=nnear, eps=eps, p=p)
    return interpol

def gridlistin(fname,xyvcols=[0,1,2],nheader=0):
    """Use for single coef per file."""
    xyv = np.genfromtxt(fname,usecols=xyvcols,skip_header=nheader)
    # filter surfer null values by taking all less than 10000, arbitrary!!
    xyv = xyv[xyv[:,2] < 10000.0]
    # xya = xya[~xya[:,2]==  missing]
    return xyv[:,0],xyv[:,1],xyv[:,2]

def map2ddata(xy,vr,xyi,radius,maptype):
    """Gridd data using various approaches."""
    stats = sts.describe(vr)
    # statsstd=sts.tstd(vr)
    if maptype == 'idw':
        vri = idw(xy,vr,xyi)
    elif maptype == 'nearest':
        vri = griddata(xy,vr,(xyi[:,0],xyi[:,1]),method='nearest')
    elif maptype == 'linear':
        #                vri=griddata(xy,vr,(xyifhull[:,0],xyifhull[:,1]),method='linear')
        vri = griddata(xy,vr,(xyi[:,0],xyi[:,1]),method='linear')
    elif maptype == 'cubic':
        vri = griddata(xy,vr,(xyi[:,0],xyi[:,1]),method='cubic')
    elif maptype == 'rbf':
        rbf = Rbf(xy[:,0],xy[:,1],vr)
        vri = rbf(xyi[:,0],xyi[:,1])
    # elif maptype =='avgmap':
    #     vri=dataavgmap(xy,vr,xyi,radius)
    elif maptype == 'triang':
        linearnd = LinearNDInterpolator(xy,vr,stats[2])
        vri = linearnd(xyi)
    elif maptype == 'ct':
        ct = CloughTocher2DInterpolator(xy,vr,stats[2])
        vri = ct(xyi)
    return vri

def filterhullpolygon(x,y,polygon):
    """Filter data using convex hull."""
    xf = []
    yf = []
    for i in range(x.size):
        if pip(x[i],y[i],polygon):
            xf.append(x[i])
            yf.append(y[i])
    return np.array(xf),np.array(yf)


def filterhullpolygon_wid(x,y,id0,id1,polygon):
    """Filter data set by well id."""
    xf = []
    yf = []
    id0f = []
    # for well name
    id1f = []
    # for z values
    for i in range(x.size):
        if pip(x[i],y[i],polygon):
            xf.append(x[i])
            yf.append(y[i])
            id0f.append(id0[i])
            # well name
            id1f.append(id1[i])
            # well attribute or porosity
    return np.array(xf),np.array(yf),id0f,np.array(id1f)

def filterhullpolygon_mask(x,y,polygon):
    """Filter points inside convex hull polygon."""
    ma = []
    for i in range(x.size):
        if pip(x[i],y[i],polygon):
            ma.append(True)
        else:
            ma.append(False)
    return np.array(ma)

def qcattributes(dfname,pdffname=None,deg=1,dp=3,scattermatrix=False,cmdlsample=None):
    """
    Establish linear fit relationships between each predictor and singular target.

    The second variable changes it from 1=linear to >1=Polynomial Fit

    """
    with PdfPages(pdffname) as pdf:
        for i in range((dfname.shape[1]) - 1):
            xv = dfname.iloc[:,i].values
            yv = dfname.iloc[:,-1].values
            xtitle = dfname.columns[i]
            ytitle = dfname.columns[-1]
            xrngmin,xrngmax = xv.min(),xv.max()
            # print(xrngmin,xrngmax)
            xvi = np.linspace(xrngmin,xrngmax)
            # print(xrng)
            qc = np.polyfit(xv,yv,deg)
            if deg == 1:
                print('Slope: %5.3f, Intercept: %5.3f' % (qc[0],qc[1]))
            else:
                print(qc)
            yvi = np.polyval(qc,xvi)
            plt.scatter(xv,yv,alpha=0.5)
            plt.plot(xvi,yvi,c='red')
            plt.xlabel(xtitle)
            plt.ylabel(ytitle)
            # commenting out annotation : only shows on last plot!!
            if deg == 1:
                plt.annotate('%s = %-.*f   + %-.*f * %s' % (ytitle,dp,qc[0],dp,qc[1],xtitle),
                    xy=(yv[4],xv[4]),xytext=(0.25,0.80),textcoords='figure fraction')

            # plt.show()
            pdf.savefig()
            plt.close()

        if scattermatrix:
            dfnamex = dfname.sample(frac=cmdlsample).copy()
            scatter_matrix(dfnamex)
            pdf.savefig()
            plt.show()
            plt.close()

def savefiles(seisf=None,sdf=None,sxydf=None,
        wellf=None, wdf=None, wxydf=None,
        outdir=None,ssuffix='',wsuffix='',name2merge=None):
    """Generic function to save csv & txt files."""
    if seisf:
        dirsplit,fextsplit = os.path.split(seisf)
        fname1,fextn = os.path.splitext(fextsplit)

        if name2merge:
            dirsplit2,fextsplit2 = os.path.split(name2merge)
            fname2,fextn2 = os.path.splitext(fextsplit2)
            fname = fname1 + '_' + fname2
        else:
            fname = fname1

        if outdir:
            slgrf = os.path.join(outdir,fname) + ssuffix + ".csv"
        else:
            slgrf = os.path.join(dirsplit,fname) + ssuffix + ".csv"
        # if not sdf.empty:
        if isinstance(sdf,pd.DataFrame):
            sdf.to_csv(slgrf,index=False)
            print('Successfully generated %s file' % slgrf)

        if outdir:
            slgrftxt = os.path.join(outdir,fname) + ssuffix + ".txt"
        else:
            slgrftxt = os.path.join(dirsplit,fname) + ssuffix + ".txt"
        if isinstance(sxydf,pd.DataFrame):
            sxydf.to_csv(slgrftxt,sep=' ',index=False)
            print('Successfully generated %s file' % slgrftxt)

    if wellf:
        dirsplit,fextsplit = os.path.split(wellf)
        fname1,fextn = os.path.splitext(fextsplit)

        if name2merge:
            dirsplit2,fextsplit2 = os.path.split(name2merge)
            fname2,fextn2 = os.path.splitext(fextsplit2)
            fname = fname1 + '_' + fname2
        else:
            fname = fname1

        if outdir:
            wlgrf = os.path.join(outdir,fname) + wsuffix + ".csv"
        else:
            wlgrf = os.path.join(dirsplit,fname) + wsuffix + ".csv"
        if isinstance(wdf,pd.DataFrame):
            wdf.to_csv(wlgrf,index=False)
            print('Successfully generated %s file' % wlgrf)

        if outdir:
            wlgrftxt = os.path.join(outdir,fname) + wsuffix + ".txt"
        else:
            wlgrftxt = os.path.join(dirsplit,fname) + wsuffix + ".txt"
        if isinstance(wxydf,pd.DataFrame):
            wxydf.to_csv(wlgrftxt,sep=' ',index=False)
            print('Successfully generated %s file' % wlgrftxt)

def listfiles(flst):
    """Print the list file."""
    for fl in flst:
        print(fl)

def plot_confusion_matrix(cm, classes,
          normalize=False,
          title='Confusion Matrix',
          cmap=plt.cm.Set2,
          hideplot=False):
    """
    Print and plot the confusion matrix.

    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig,ax = plt.subplots()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_roc_curve(y_test,
        preds,
        poslbl,
        hideplot=False):
    """Calculate the fpr and tpr for all thresholds of the classification."""
    fpr, tpr, threshold = roc_curve(y_test, preds,pos_label=poslbl)
    roc_auc = auc(fpr, tpr)
    plt.title('ROC Curve  %1d' % poslbl)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

def gensamples(datain,targetin,
       ncomponents=2,
       nsamples=10,
       kind='r',
       func=None):
    """
    Generate data using GMM.

    Reads in features array and target column
    uses GMM to generate samples after scaling target col
    scales target col back to original values
    saves data to csv file
    returns augmented data features array and target col

    newfeatures,newtarget =gensamples(wf,codecol,kind='c',func='cbr')
    """
    d0 = datain
    t0 = targetin
    d0t0 = np.concatenate((d0,t0.values.reshape(1,-1).T),axis=1)
    sc = StandardScaler()
    t0min,t0max = t0.min(),t0.max()
    t0s = sc.fit_transform(t0.values.reshape(1,-1))
    d0t0s = np.concatenate((d0,t0s.T),axis=1)
    gmm = mixture.GaussianMixture(n_components=ncomponents,covariance_type='spherical', max_iter=500, random_state=0)
    gmm.fit(d0t0s)
    d0sampled = gmm.sample(nsamples)[0]
    d1sampled = d0sampled[:,:-1]
    targetunscaled = d0sampled[:,-1].reshape(1,-1)
    scmm = MinMaxScaler((t0min,t0max))
    if kind == 'c':
        targetscaledback = np.floor(scmm.fit_transform(targetunscaled.T))
    else:
        targetscaledback = scmm.fit_transform(targetunscaled.T)
    d1t1 = np.concatenate((d1sampled,targetscaledback),axis=1)
    d1 = np.concatenate((d0t0,d1t1))
    print(d1.shape)
    fname = 'gensamples_' + func + '.csv'
    np.savetxt(fname,d1,fmt='%.3f',delimiter=',')
    return d1[:,:-1],d1[:,-1]

def process_sattrib(gridfileslist,
        gridcols=None,
        gridheader=None,
        ilxl=False,outdir=None):
    """Merge grids from file list to a columnar csv."""
    grdfiles = []
    with open(gridfileslist, 'r') as f:
        for line in f:
            grdfiles.append(line.rstrip())
    # listfiles(grdfiles)

    dirsplit,fextsplit = os.path.split(grdfiles[0])
    fname,fextn = os.path.splitext(fextsplit)
    if ilxl:
        colnames = ['IL','XL','X','Y',fname]
    else:
        colnames = ['X','Y',fname]
    sag = pd.read_csv(grdfiles[0],delim_whitespace=True,index_col=False,
                    header=None,usecols=gridcols,skiprows=gridheader,names=colnames)
    allattrib = sag.copy()
    allattrib.columns = colnames
    if ilxl:
        sag['IL'] = sag['IL'].astype(int)
        sag['XL'] = sag['XL'].astype(int)
        sag.drop_duplicates(inplace=True)
    for i in range(1,len(grdfiles)):
        dirsplit,fextsplit = os.path.split(grdfiles[i])
        fname,fextn = os.path.splitext(fextsplit)
        sag = pd.read_csv(grdfiles[i],delim_whitespace=True,index_col=False,
                    header=None,usecols=gridcols,skiprows=gridheader,names=colnames)
        if ilxl:
            colnames = ['IL','XL','X','Y',fname]
        else:
            colnames = ['X','Y',fname]
        sag.columns = colnames
        if ilxl:
            sag['IL'] = sag['IL'].astype(int)
            sag['XL'] = sag['XL'].astype(int)
            sag.drop(['X','Y'],inplace=True,axis=1)
            sag.drop_duplicates(inplace=True)
            allattrib = pd.merge(allattrib,sag,on=['IL','XL'],how='left')
            # allattrib.fillna(method='pad',axis=0,inplace=True)
            allattrib.fillna(allattrib.mean(),axis=0,inplace=True)
        else:
            allattrib[fname] = sag.iloc[:,2]
    if ilxl:
        allattrib.drop(['IL','XL'],inplace=True,axis=1)
    savefiles(seisf=gridfileslist,
        sdf=allattrib,
        outdir=outdir,
        ssuffix='_sus')

def process_dropcols(csvfile,cmdlcols2drop=None,cmdloutdir=None):
    """Drop columns."""
    allattrib = pd.read_csv(csvfile)
    # print(allattrib.head(5))
    # cols = allattrib.columns.tolist()
    allattrib.drop(allattrib.columns[[cmdlcols2drop]],axis=1,inplace=True)
    # print(allattrib.head(5))
    # cols = allattrib.columns.tolist()

    savefiles(seisf=csvfile,
        sdf=allattrib,
        outdir=cmdloutdir,
        ssuffix='_drpc')

def process_prepfile(csvfile,cmdloutdir=None):
    """
    Prepare file exported from Petrel.

    For files exported from Petrel as csv
    Top 2 rows are deleted and left most column is deleted
    """
    data = pd.read_csv(csvfile,skiprows=2)
    data1 = data.iloc[:,[1,3,4,5,6,7]]                           # Selects the columns that we need only
    data2 = data1.copy()[data1['Horizon before'] != 'Outside']     # Read only data without 'Outside' #!= negates it
    cols = ['Well', 'X', 'Y', 'Actual', 'Predicted', 'Error']      # Relabelling columns
    data2.columns = cols
    data2['Predicted'] = pd.to_numeric(data2['Predicted'],errors='coerce')

    savefiles(seisf=csvfile,
        sdf=data2,
        outdir=cmdloutdir,
        ssuffix='_w')

def process_sscalecols(csvfile,cmdlxyzcols=None,cmdlincludexyz=False,
                cmdlkind=None,cmdloutdir=None):
    """Scale seismic attributes."""
    allattrib = pd.read_csv(csvfile)
    if cmdlincludexyz:
        allattrib['Xscaled'] = allattrib[allattrib.columns[cmdlxyzcols[0]]]
        allattrib['Yscaled'] = allattrib[allattrib.columns[cmdlxyzcols[1]]]
        allattrib['Zscaled'] = allattrib[allattrib.columns[cmdlxyzcols[2]]]
    if allattrib.isnull().values.any():
        print('Warning: Null Values in the file will be dropped')
        allattrib.dropna(inplace=True)
    xyzcols = allattrib[allattrib.columns[cmdlxyzcols]]
    allattrib.drop(allattrib.columns[[cmdlxyzcols]],axis=1,inplace=True)
    cols = allattrib.columns.tolist()
    if cmdlkind == 'standard':
        allattribs = StandardScaler().fit_transform(allattrib.values)
    elif cmdlkind == 'quniform':
        allattribs = QuantileTransformer(output_distribution='uniform').fit_transform(allattrib.values)
    else:
        allattribs = QuantileTransformer(output_distribution='normal').fit_transform(allattrib.values)

    allattribsdf = pd.DataFrame(allattribs,columns=cols)
    allattribsxy = pd.concat([xyzcols,allattribsdf],axis=1)

    if cmdlincludexyz:
        savefiles(seisf=csvfile,
            sdf=allattribsxy,
            outdir=cmdloutdir,
            ssuffix='_ssxyz')
    else:
        savefiles(seisf=csvfile,
            sdf=allattribsxy,
            outdir=cmdloutdir,
            ssuffix='_ss')

def process_listcsvcols(csvfile):
    """List enumerate csv columns."""
    data = pd.read_csv(csvfile)
    clist = list(enumerate(data.columns))
    print(clist)

def process_wscaletarget(csvfile,cmdlkind=None,
        cmdltargetcol=None,cmdloutdir=None):
    """
    Scale petrophysical parameter.

    process well df by scaling only the target column
    new scaled column is added to df and saved as csv
    file
    """
    allattrib = pd.read_csv(csvfile)
    if allattrib.iloc[:,cmdltargetcol].isnull().any():
        print('Warning: Null Values in the file will be dropped')
        allattrib[cmdltargetcol].dropna(inplace=True)
    targetc = allattrib.iloc[:,cmdltargetcol].values.reshape(-1,1)
    targetcn = allattrib.columns[cmdltargetcol] + '_scaled'
    if cmdlkind == 'standard':
        allattrib[targetcn] = StandardScaler().fit_transform(targetc)
    elif cmdlkind == 'quniform':
        allattrib[targetcn] = QuantileTransformer(output_distribution='uniform').fit_transform(targetc)
    else:
        # quantile distribution = normal
        allattrib[targetcn] = QuantileTransformer(output_distribution='normal').fit_transform(targetc)

    # ts for target scaled
    savefiles(seisf=csvfile,
        sdf=allattrib,
        outdir=cmdloutdir,
        ssuffix='_ts')

def process_wscalecols(cmdlcsvfile,cmdlwxyzcols=None,cmdloutdir=None):
    """
    Scale attribute columns of well csv.

    If Petrel csv files of wells were used then you need to scale
    inividual attributes. Recommeded to use seiswellmerge option
    """
    allattrib = pd.read_csv(cmdlcsvfile)
    if allattrib.isnull().values.any():
        print('Warning: Data has Null values. Remove first before scaling')
        return

    dirsplit,fextsplit = os.path.split(cmdlcsvfile)
    fname,fextn = os.path.splitext(fextsplit)
    # scaled_fname = os.path.join(dirsplit,fname) + "_ws.csv"
    wxyzcols = allattrib[allattrib.columns[cmdlwxyzcols]]
    allattrib.drop(allattrib.columns[[cmdlwxyzcols]],axis=1,inplace=True)
    cols = allattrib.columns.tolist()
    SSg = StandardScaler().fit(allattrib.values)
    allattribs = SSg.transform(allattrib.values)
    allattribsdf = pd.DataFrame(allattribs,columns=cols)
    allattribsxy = pd.concat([wxyzcols,allattribsdf],axis=1)

    savefiles(seisf=cmdlcsvfile,
        sdf=allattribsxy,
        outdir=cmdloutdir,
        ssuffix='_wscaled')

def process_wellattrib(cmdlwattribfile,
        cmdlwxyfile,
        cmdlwattribcols=None,
        cmdlwxyzcols=None,
        cmdlfillna=None,
        cmdloutdir=None):
    """Read in petrophysical data."""
    waf = pd.read_csv(cmdlwattribfile)
    wafx = waf[waf.columns[[cmdlwattribcols]]]
    ppattribname = wafx.columns[1]
    if cmdlfillna == 'delete':
        wafxx = wafx.dropna()
    elif cmdlfillna == 'mean':
        wafxx = wafx.fillna(wafx.mean())
    elif cmdlfillna == 'median':
        wafxx = wafx.fillna(wafx.median())
    print(wafxx.head())

    wxy = pd.read_csv(cmdlwxyfile)
    wxy0 = wxy[wxy.columns[cmdlwxyzcols]]
    print(wxy0.head())
    wxyzwellcol = wxy0.columns[0]
    print('wxyzwell col name: ',wxyzwellcol)
    wxy0.rename(columns={wxyzwellcol:'Well'},inplace=True)
    wawellcol = wafxx.columns[0]
    print('Well PP col name:',wawellcol)
    wafxx.rename(columns={wawellcol:'Well'},inplace=True)

    ppxy = wxy0.merge(wafxx, left_on=wxy0.columns[0], right_on=wafxx.columns[0],how='inner')
    print(ppxy.head())
    '''
    ppxy['Well'] = ppxy.iloc[:,0]
    print(ppxy.columns)
    ppxyordered = ppxy[ppxy.columns[[-1,3,4,5,1]]]
    print(ppxyordered.columns)
    #results in well x y z attribute
    '''
    savefiles(seisf=cmdlwattribfile,
        sdf=ppxy,
        outdir=cmdloutdir,
        ssuffix='_' + ppattribname)

def process_wamerge(cmdlcsvfileslist=None,
        cmdlcsvcols=None,
        cmdlcsvskiprows=None,
        cmdloutdir=None):
    """
    Only for Petrel output.

    This is designed to read csv files generated from Petrel for each attribute
    back interpolation is from Petrel done for each attribute.
    This merges all those csv files to one csv file with each column representing
    at attribute.

    You should use wellseisattrib option instead
    """
    csvfiles = []
    with open(cmdlcsvfileslist, 'r') as f:
        for line in f:
            csvfiles.append(line.rstrip())
    # listfiles(csvfiles)

    dirsplit,fextsplit = os.path.split(csvfiles[0])
    fname,fextn = os.path.splitext(fextsplit)
    colnames = ['Well','X','Y','Z',fname]
    wag = pd.read_csv(csvfiles[0],skiprows=cmdlcsvskiprows,usecols=cmdlcsvcols,names=colnames)
    wag.iloc[:,4] = pd.to_numeric(wag.iloc[:,4],errors='coerce')
    allattrib = wag.copy()
    # print(allattrib.info())
    for i in range(1,len(csvfiles)):
        dirsplit,fextsplit = os.path.split(csvfiles[i])
        fname,fextn = os.path.splitext(fextsplit)
        wag = pd.read_csv(csvfiles[i],skiprows=cmdlcsvskiprows,usecols=cmdlcsvcols,names=colnames)
        wag.iloc[:,4] = pd.to_numeric(wag.iloc[:,4],errors='coerce')
        allattrib[fname] = wag.iloc[:,4]
        # print(allattrib.info())
        # print()

    allattrib.dropna(inplace=True)
    savefiles(seisf=cmdlcsvfileslist,
        sdf=allattrib,
        sxydf=allattrib,
        outdir=cmdloutdir,
        ssuffix='_wm')

def process_seiswellattrib(cmdlseiscsv,cmdlwellcsv,
                    cmdlwellcsvcols=None,cmdlradius=None,
                    cmdlinterpolate=None,cmdloutdir=None):
    """Merge seismic csv with well petrophysifcal csv."""
    sa = pd.read_csv(cmdlseiscsv)
    wa = pd.read_csv(cmdlwellcsv)
    print(sa.head(5))
    print(wa.head(5))
    xs = sa.iloc[:,0]
    ys = sa.iloc[:,1]
    xys = np.transpose(np.vstack((xs,ys)))
    xyhull = qhull(xys)

    xw = wa.iloc[:,cmdlwellcsvcols[1]].values
    yw = wa.iloc[:,cmdlwellcsvcols[2]].values
    # xyw = np.transpose(np.vstack((xw,yw)))
    wz = wa.iloc[:,cmdlwellcsvcols[3]]  # z value
    wid = wa.iloc[:,cmdlwellcsvcols[0]]  # wellname
    wida = wa.iloc[:,cmdlwellcsvcols[4]]   # well porosity or any other attribute

    ma = filterhullpolygon_mask(xw,yw,xyhull)
    print('Remaining Wells after convex hull: ',len(ma))
    xwf = xw[ma]
    ywf = yw[ma]
    wzf = wz[ma]
    widf = wid[ma]
    wattribf = wida[ma]

    xywf = np.transpose(np.vstack((xwf,ywf)))
    welldfcols = ['Well','X','Y','Z']
    # wellsin_df = pd.DataFrame([widf,xwf,ywf,wattrib],columns=welldfcols)
    wellsin_df = pd.DataFrame(widf,columns=[welldfcols[0]])
    wellsin_df[welldfcols[1]] = xwf
    wellsin_df[welldfcols[2]] = ywf
    wellsin_df[welldfcols[3]] = wzf
    print('wellsin df shape:',wellsin_df.shape)
    print(wellsin_df.head(10))

    nattrib = sa.shape[1]
    print('nattrib:',nattrib)
    # welldflist =[]
    for i in range(3,nattrib):
        vs = sa.iloc[:,i].values
        zwsa = map2ddata(xys,vs,xywf,cmdlradius,cmdlinterpolate)
        print('i:',i,'zwsa:',zwsa.size)
        # welldflist.append(zwsa)
        colname = sa.columns[i]
        wellsin_df[colname] = zwsa
    wa_col = wa.columns[cmdlwellcsvcols[4]]
    wellsin_df[wa_col] = wattribf

    savefiles(seisf=cmdlseiscsv,
        sdf=wellsin_df,
        outdir=cmdloutdir,
        ssuffix='_swa',name2merge=cmdlwellcsv)

def process_PCAanalysis(cmdlallattribcsv,cmdlacolsrange=None,
            cmdlanalysiscols=None,cmdlhideplot=False):
    """Filter out components."""
    swa = pd.read_csv(cmdlallattribcsv)
    if cmdlacolsrange:
        print('From col# %d to col %d' % (cmdlacolsrange[0],cmdlacolsrange[1]))
        X = swa[swa.columns[cmdlacolsrange[0]: cmdlacolsrange[1] + 1]].values
    else:
        print('analysis cols',cmdlanalysiscols)
        X = swa[swa.columns[cmdlanalysiscols]].values
    # Create scaler: scaler
    scaler = StandardScaler()

    # Create a PCA instance: pca
    pca = PCA()

    # Create pipeline: pipeline
    pipeline = make_pipeline(scaler,pca)

    # Fit the pipeline to well data
    pipeline.fit(X)
    dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    pdfsave = os.path.join(dirsplit,fname) + "_pca.pdf"

    # Plot the explained variances
    features = range(pca.n_components_)
    with PdfPages(pdfsave) as pdf:
        plt.figure(figsize=(8,8))
        plt.bar(features, pca.explained_variance_)
        plt.xlabel('PCA feature')
        plt.ylabel('variance')
        plt.xticks(features)
        plt.title('Elbow Plot')
        pdf.savefig()
        if not cmdlhideplot:
            plt.show()
        plt.close()

def process_PCAfilter(cmdlallattribcsv,
        cmdlacolsrange=None,
        cmdlanalysiscols=None,
        cmdltargetcol=None,
        cmdlncomponents=None,
        cmdloutdir=None,
        cmdlcols2addback=None):
    """PCA keep selected components only."""
    swa = pd.read_csv(cmdlallattribcsv)
    if cmdlacolsrange:
        print('From col# %d to col %d' % (cmdlacolsrange[0],cmdlacolsrange[1]))
        X = swa[swa.columns[cmdlacolsrange[0]: cmdlacolsrange[1] + 1]].values
        colnames = swa[swa.columns[cmdlacolsrange[0]: cmdlacolsrange[1] + 1]].columns
    else:
        print('analysis cols',cmdlanalysiscols)
        X = swa[swa.columns[cmdlanalysiscols]].values
        colnames = swa[swa.columns[cmdlanalysiscols]].columns
    if cmdltargetcol:
        targetname = swa.columns[cmdltargetcol]
        print(colnames,targetname)

    # Create scaler: scaler
    # scaler = StandardScaler()

    # Create a PCA instance: pca
    if not cmdlncomponents:
        pca = PCA(X.shape[1])
        colnames = list()
        # [colnames.append('PCA%d'%i) for i in range(X.shape[1] -1)]
        [colnames.append('PCA%d' % i) for i in range(X.shape[1])]
    else:
        pca = PCA(cmdlncomponents)
        colnames = list()
        # [colnames.append('PCA%d'%i) for i in range(cmdl.ncomponents -1)]
        [colnames.append('PCA%d' % i) for i in range(cmdlncomponents)]

    if cmdltargetcol:
        colnames.append(targetname)
    # Create pipeline: pipeline
    # pipeline = make_pipeline(scaler,pca)

    # Fit the pipeline to well data
    # CX = pipeline.fit_transform(X)
    CX = pca.fit_transform(X)
    print('cx shape',CX.shape,'ncolumns ',len(colnames))
    swa0 = swa[swa.columns[cmdlcols2addback]]
    cxdf = pd.DataFrame(CX,columns=colnames)
    if cmdltargetcol:
        cxdf[targetname] = swa[swa.columns[cmdltargetcol]]
    cxdfall = pd.concat([swa0,cxdf],axis=1)

    savefiles(seisf=cmdlallattribcsv,
        sdf=cxdfall, sxydf=cxdfall,
        outdir=cmdloutdir,
        ssuffix='_pca')

def process_scattermatrix(cmdlallattribcsv,cmdlwellxyzcols=None,cmdlsample=None):
    """Plot scatter matrix for all attributes."""
    swa = pd.read_csv(cmdlallattribcsv)
    # print(swa.sample(5))
    swax = swa.drop(swa.columns[cmdlwellxyzcols],axis=1)
    swaxx = swax.sample(frac=cmdlsample).copy()
    scatter_matrix(swaxx)
    plt.show()

def process_eda(cmdlallattribcsv,
        cmdlxyzcols=None,
        cmdlpolydeg=None,
        cmdlsample=None,
        cmdlhideplot=False,
        cmdlplotoption=None,
        cmdloutdir=None):
    """Generate Exploratroy Data Analyses plots."""
    plt.style.use('seaborn-whitegrid')
    swa = pd.read_csv(cmdlallattribcsv)
    swax = swa.drop(swa.columns[cmdlxyzcols],axis=1)
    if cmdlsample:
        swax = swax.sample(frac=cmdlsample).copy()
        print('**********Data has been resampled by %.2f' % cmdlsample)
    dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    def pltundefined():
        pass

    def pltheatmap():
        if cmdloutdir:
            pdfheat = os.path.join(cmdloutdir,fname) + "_heat.pdf"
        else:
            pdfheat = os.path.join(dirsplit,fname) + "_heat.pdf"

        plt.figure(figsize=(8,8))
        mask = np.zeros_like(swax.corr(), dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        ht = sns.heatmap(swax.corr(),
                    vmin=-1, vmax=1,
                    square=True,
                    cmap='RdBu_r',
                    mask=mask,
                    linewidths=.5)
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)

        if not cmdlhideplot:
            plt.show()
        fig = ht.get_figure()
        fig.savefig(pdfheat)

    def pltxplots():
        # decimal places for display
        dp = 3
        ytitle = swax.columns[-1]
        # assume the xplots are for attributes vs target
        # that assumption is not valid for other plots
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_xplots.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_xplots.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range((swax.shape[1]) - 1):

                xtitle = swax.columns[i]

                xv = swax.iloc[:,i].values
                yv = swax.iloc[:,-1].values
                xrngmin,xrngmax = xv.min(),xv.max()
                # print(xrngmin,xrngmax)
                xvi = np.linspace(xrngmin,xrngmax)
                # print(xrng)
                qc = np.polyfit(xv,yv,cmdlpolydeg)
                if cmdlpolydeg == 1:
                    print('%s  vs %s  Slope: %5.3f, Intercept: %5.3f'% (xtitle,ytitle,qc[0],qc[1]))
                else:
                    print('%s  vs %s ' % (xtitle,ytitle),qc)
                yvi = np.polyval(qc,xvi)
                plt.scatter(xv,yv,alpha=0.5)
                plt.plot(xvi,yvi,c='red')
                plt.xlabel(xtitle)
                plt.ylabel(ytitle)
                # commenting out annotation : only shows on last plot!!
                if cmdlpolydeg == 1:
                    plt.annotate('%s = %-.*f   + %-.*f * %s' % (ytitle,dp,qc[0],dp,qc[1],xtitle),
                        xy=(yv[4],xv[4]),xytext=(0.25,0.80),textcoords='figure fraction')

                # if not cmdlhideplot:
                    # plt.show()
                # fig = p0.get_figure()
                # fig.savefig(pdfcl)
                # plt.close()
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

    def pltbox():
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_box.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_box.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range((swax.shape[1])):
                # xtitle = swax.columns[i]
                sns.boxplot(x=swax.iloc[:,i])
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()
            ax = sns.boxplot(data=swax)
            labels = ax.get_xticklabels()
            plt.setp(labels, rotation=30, fontsize=10)
            # ax.set_xticklabels(xticklabels, rotation = 45)
            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

    def pltdistribution():
        plt.style.use('seaborn-whitegrid')
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_dist.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_dist.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range(swax.shape[1]):
                sns.distplot(swax.iloc[:,i])
                # title = swax.columns[i]

                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

    def pltscattermatrix():
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_scatter.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_scatter.pdf"
        scatter_matrix(swax, alpha=0.2, figsize=(8, 8), diagonal='kde')
        if not cmdlhideplot:
            plt.show()
        plt.savefig(pdfcl)
        plt.close()

    plotoptlist = {'xplots': pltxplots,'heatmap': pltheatmap,
                'box':pltbox,
                'distribution':pltdistribution,
                # 'distribution':lambda: pltdistribution(swax),
                'scattermatrix':pltscattermatrix}
    plotoptlist.get(cmdlplotoption,pltundefined)()

def process_linreg(cmdlallattribcsv,
        cmdlwtargetcol=None,
        cmdlwcolsrange=None,
        cmdlwpredictorcols=None,
        cmdlwellxyzcols=None,
        cmdlminmaxscale=None,
        cmdloutdir=None):
    """Multi Linear Regression."""
    swa = pd.read_csv(cmdlallattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1]]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    swxyz = swa[swa.columns[cmdlwellxyzcols]].copy()
    lm = LinearRegression()
    lm.fit(X, y)  # Fitting all predictors 'X' to the target 'y' using linear fit model

    # Print intercept and coefficients
    print('Intercept: ',lm.intercept_)
    print('Coefficients: ',lm.coef_)
    print('R2 Score:',lm.score(X, y))

    # Calculating coefficients
    cflst = lm.coef_.tolist()
    # cflst.append(lm.intercept_)
    cflst.insert(0,lm.intercept_)
    cnameslst = colnames.tolist()
    # cnameslst.append('Intercept')
    cnameslst.insert(0,'Intercept')
    coeff = pd.DataFrame(cnameslst,columns=['Attribute'])
    coeff['Coefficient Estimate'] = pd.Series(cflst)
    pred = lm.predict(X)

    if cmdlminmaxscale:
        ymin,ymax = y.min(), y.max()
        mmscale = MinMaxScaler((ymin,ymax))
        # mmscale.fit(pred)
        pred1 = pred.reshape(-1,1)
        predscaled = mmscale.fit_transform(pred1)
        swa['LRPred'] = predscaled
        swxyz['LRPred'] = predscaled
    else:
        swa['LRPred'] = pred
        swxyz['LRPred'] = pred

    # Calculating Mean Squared Error
    mse = np.mean((pred - y)**2)
    print('MSE: ',mse)
    print('R2 Score:',lm.score(X,y))
    swa['Predict'] = pred
    swa['Prederr'] = pred - y

    savefiles(seisf=cmdlallattribcsv,
        sdf=swa, sxydf=swxyz,
        wellf=cmdlallattribcsv,
        wdf=coeff, wxydf=coeff,
        outdir=cmdloutdir,
        ssuffix='_lr',
        wsuffix='_lrcf')

def process_linfitpredict(cmdlwellattribcsv,
        cmdlseisattribcsv,
        cmdlwcolsrange=None,
        cmdlwpredictorcols=None,
        cmdlwtargetcol=None,
        cmdlsaxyzcols=None,
        cmdlscolsrange=None,
        cmdlspredictorcols=None,
        cmdlminmaxscale=None,
        cmdloutdir=None,
        cmdlhideplot=False):
    """Perform linear fitting and prediction."""
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1]]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    ssa = pd.read_csv(cmdlseisattribcsv)
    # New code
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    # plt.style.use('seaborn-whitegrid')

    if cmdlscolsrange:
        print('Seismic Predictors From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        colnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1]]].columns
    else:
        print('Seismic Predictor cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        colnames = swa[swa.columns[cmdlspredictorcols]].columns

    lm = LinearRegression()
    lm.fit(X, y)
    # Fitting all predictors 'X' to the target 'y' using linear fit model
    ypred = lm.predict(X)
    # Print intercept and coefficients
    print('Intercept: ',lm.intercept_)
    print('Coefficients: ',lm.coef_)
    print('R2 Score:',lm.score(X, y))

    # Calculating coefficients
    cflst = lm.coef_.tolist()
    # cflst.append(lm.intercept_)
    cflst.insert(0,lm.intercept_)
    cnameslst = colnames.tolist()
    # cnameslst.append('Intercept')
    cnameslst.insert(0,'Intercept')
    coeff = pd.DataFrame(cnameslst,columns=['Attribute'])
    coeff['Coefficient Estimate'] = pd.Series(cflst)

    pred = lm.predict(Xpred)
    if cmdlminmaxscale:
        ymin,ymax = y.min(), y.max()
        mmscale = MinMaxScaler((ymin,ymax))
        # mmscale.fit(pred)
        pred1 = pred.reshape(-1,1)
        predscaled = mmscale.fit_transform(pred1)
        ssa['LRPred'] = predscaled
        ssxyz['LRPred'] = predscaled
    else:
        ssa['LRPred'] = pred
        ssxyz['LRPred'] = pred

    # ax =plt.scatter(y,ypred)
    ax = sns.regplot(x=y,y=ypred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Linear Regressor %s' % swa.columns[cmdlwtargetcol])
    if not cmdlhideplot:
        plt.show()
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + "_lreg.pdf"
        xyplt = os.path.join(cmdloutdir,fname) + "_lregxplt.csv"
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_lreg.pdf"
        xyplt = os.path.join(dirsplit,fname) + "_lregxplt.csv"
    fig = ax.get_figure()
    fig.savefig(pdfcl)
    # xpltcols = ['Actual','Predicted']
    xpltdf = swa.iloc[:,:3].copy()  # copy well x y
    xpltdf['Actual'] = y
    xpltdf['Predicted'] = ypred
    xpltdf.to_csv(xyplt,index=False)
    print('Sucessfully generated xplot file %s' % xyplt)

    savefiles(seisf=cmdlseisattribcsv,
        sdf=ssa, sxydf=ssxyz,
        wellf=cmdlseisattribcsv,
        wdf=coeff,
        wxydf=coeff,
        outdir=cmdloutdir,
        ssuffix='_lr',
        wsuffix='_lrcf',
        name2merge=cmdlwellattribcsv)


def process_featureranking(cmdlallattribcsv,
        cmdlwtargetcol=None,
        cmdlwcolsrange=None,
        cmdlwpredictorcols=None,
        cmdltestfeatures=None,
        cmdllassoalpha=None,
        cmdlfeatures2keep=None,
        cmdlcv=None,
        cmdltraintestsplit=None):
    """Rank features with different approaches."""
    swa = pd.read_csv(cmdlallattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    if cmdltestfeatures == 'rlasso':
        rlasso = RandomizedLasso(alpha=cmdllassoalpha)
        rlasso.fit(X, y)
        print("Features sorted by their score:")
        print(sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),colnames), reverse=True))
        # print (sorted(zip(rlasso.scores_,colnames), reverse=True))

    elif cmdltestfeatures == 'rfe':
        # rank all features, i.e continue the elimination until the last one
        lm = LinearRegression()
        rfe = RFE(lm, n_features_to_select=cmdlfeatures2keep)
        rfe.fit(X,y)
        # print ("Features sorted by their rank:")
        # print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), colnames)))

        scores = []
        for i in range(X.shape[1]):
            score = cross_val_score(lm, X[:, i:i + 1], y, scoring="r2",
                cv=ShuffleSplit(len(X), cmdlcv, cmdltraintestsplit))
        # scores.append(round(np.mean(score), 3))
        scores.append(np.mean(score))
        # print (sorted(scores, reverse=True))
        r2fr = pd.DataFrame(sorted(zip(scores, colnames),reverse=True),columns=['R2 Score ','Attribute'])
        print('Feature Ranking by R2 scoring: ')
        print(r2fr)

    elif cmdltestfeatures == 'svrcv':
        # rank all features, i.e continue the elimination until the last one

        estimator = SVR(kernel="linear")
        selector = RFECV(estimator, step=1, cv=cmdlcv)
        selector = selector.fit(X, y)
        fr = pd.DataFrame(sorted(zip(selector.ranking_, colnames)),columns=['Importance','Attribute'])
        print('Feature Ranking with Cross Validated Recursive Feature Elimination Using SVR: ')
        print(fr)

    elif cmdltestfeatures == 'svr':
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, cmdlfeatures2keep, step=1)
        selector = selector.fit(X, y)
        fr = pd.DataFrame(sorted(zip(selector.ranking_, colnames)),columns=['Importance','Attribute'])
        print('Feature Ranking with Recursive Feature Elimination Using SVR: ')
        print(fr)

    elif cmdltestfeatures == 'rfregressor':
        rf = RandomForestRegressor(n_estimators=20, max_depth=4)
        rf.fit(X,y)
        fi = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), colnames),
            reverse=True),columns=['Importance','Attribute'])
        print(fi)
        # print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), colnames)))
        # print(rf.feature_importances_)
        scores = []

        for i in range(X.shape[1]):
            score = cross_val_score(rf, X[:, i:i + 1], y, scoring="r2",
                  cv=ShuffleSplit(len(X), cmdlcv, cmdltraintestsplit))
            scores.append((round(np.mean(score), 3), colnames[i]))
        cvscoredf = pd.DataFrame(sorted(scores,reverse=True),columns=['Partial R2','Attribute'])
        print('\nCross Validation:')
        print(cvscoredf)

    elif cmdltestfeatures == 'decisiontree':
        regressor = DecisionTreeRegressor(random_state=0)
        # cross_val_score(regressor, X, y, cv=3)
        regressor.fit(X,y)
        # print(regressor.feature_importances_)
        fr = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), regressor.feature_importances_),
            colnames), reverse=True),columns=['Importance','Attribute'])
        print('Feature Ranking with Decision Tree Regressor: ')
        print(fr)

def process_KNNtest(cmdlallattribcsv,
        cmdlsample=1.0,
        cmdlwcolsrange=None,
        cmdlwpredictorcols=None,
        cmdlwtargetcol=None,
        cmdlcv=None,
        cmdlhideplot=None,
        cmdloutdir=None):
    """Test for # of neighbors for KNN regression."""
    swa = pd.read_csv(cmdlallattribcsv)
    swa = swa.sample(frac=cmdlsample).copy()

    if cmdlwcolsrange:
        print('Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        # colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    k_values = np.array([n for n in range(1,21)])
    # print('kvalues:',k_values)
    mselist = []
    stdlist = []
    for k in k_values:
        # kfold = KFold(n_splits=10, random_state=7)
        kfold = KFold(n_splits=cmdlcv, random_state=7)
        KNNmodel = KNeighborsRegressor(n_neighbors=k)
        scoring = 'neg_mean_squared_error'
        results = cross_val_score(KNNmodel, X, y, cv=kfold, scoring=scoring)
        print("K value: %2d  MSE: %.3f (%.3f)" % (k,results.mean(), results.std()))
        mselist.append(results.mean())
        stdlist.append(results.std())

    dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + "_knn.pdf"
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_knn.pdf"
    plt.plot(k_values,mselist)
    plt.xlabel('# of clusters')
    plt.ylabel('Neg Mean Sqr Error')
    plt.savefig(pdfcl)
    if not cmdlhideplot:
        plt.show()

def process_KNNfitpredict(cmdlwellattribcsv,cmdlseisattribcsv,
        cmdlwcolsrange=None,
        cmdlwpredictorcols=None,
        cmdlwtargetcol=None,
        cmdlsaxyzcols=None,
        cmdlscolsrange=None,
        cmdlspredictorcols=None,
        cmdlkneighbors=None,
        cmdlminmaxscale=None,
        cmdloutdir=None,
        cmdlgeneratesamples=None,
        cmdlradius=None,
        cmdlinterpolate='idw',  # hard coded interpolation method
        cmdlhideplot=False):
    """Use K Nearest Neigbors to fit regression."""
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1]]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        # colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='r',func='svr')

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    plt.style.use('seaborn-whitegrid')

    if cmdlscolsrange:
        print('Seismic Predictors From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        # colnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1]]].columns
    else:
        print('Seismic Predictor cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # colnames = swa[swa.columns[cmdlspredictorcols]].columns

    KNNmodel = KNeighborsRegressor(n_neighbors=cmdlkneighbors)
    KNNmodel.fit(X, y)

    ypred = KNNmodel.predict(X)

    # Calculating Mean Squared Error
    mse = np.mean((ypred - y)**2)
    print('Metrics on input data: ')
    print('MSE: %.4f' % (mse))
    print('R2 Score: %.4f' % (KNNmodel.score(X,y)))
    ccmdl = sts.pearsonr(y,ypred)
    print('Pearsonr : %10.3f  %10.4f ' % (ccmdl[0],ccmdl[1]))

    pred = KNNmodel.predict(Xpred)

    qc0 = np.polyfit(y,ypred,1)
    xrngmin,xrngmax = y.min(),y.max()
    xvi = np.linspace(xrngmin,xrngmax)
    if cmdlminmaxscale:
        ymin,ymax = y.min(), y.max()
        mmscale = MinMaxScaler((ymin,ymax))
        # mmscale.fit(pred)
        pred1 = pred.reshape(-1,1)
        predscaled = mmscale.fit_transform(pred1)
        ssa['KNNPred'] = predscaled
        ssxyz['KNNPred'] = predscaled
    else:
        ssa['KNNPred'] = pred
        ssxyz['KNNPred'] = pred

    # ax =plt.scatter(y,ypred)
    # sns.set(color_codes=True)
    # ax =sns.regplot(x=y,y=ypred)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    yvi0 = np.polyval(qc0,xvi)
    plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
    plt.plot(xvi,yvi0,c='k',lw=2)

    ax.annotate('Model = %-.*f * Actual + %-.*f' %
        (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
        textcoords='figure fraction', fontsize=10)
    ax.annotate('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
        (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.81),
        textcoords='figure fraction', fontsize=10)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('KNN Regressor %s %-d' % (swa.columns[cmdlwtargetcol],cmdlkneighbors))

    if not cmdlhideplot:
        plt.show()
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + "_knnr.pdf"
        xyplt = os.path.join(cmdloutdir,fname) + "_knnrxplt.csv"
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_knnr.pdf"
        xyplt = os.path.join(dirsplit,fname) + "_knnrxplt.csv"
    fig = ax.get_figure()
    fig.savefig(pdfcl)

    if not cmdlgeneratesamples:
        # xpltcols = ['Actual','Predicted']
        xpltdf = swa.iloc[:,:3].copy()
        # copy well x y
        xpltdf['Actual'] = y
        xpltdf['Predicted'] = ypred
        xpltdf.to_csv(xyplt,index=False)
        print('Sucessfully generated xplot file %s' % xyplt)

        # map back interpolate
        xs = ssxyz.iloc[:,0].values
        ys = ssxyz.iloc[:,1].values
        xys = np.transpose(np.vstack((xs,ys)))
        xw = swa.iloc[:,cmdlsaxyzcols[1]].values
        yw = swa.iloc[:,cmdlsaxyzcols[2]].values
        xyw = np.transpose(np.vstack((xw,yw)))

        print('******Map Back Interpolation')
        zw = map2ddata(xys,pred,xyw,cmdlradius,cmdlinterpolate)
        ccmap = sts.pearsonr(y,zw)
        print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (ccmap[0],ccmap[1]))

        xrngmin,xrngmax = y.min(),y.max()
        xvi = np.linspace(xrngmin,xrngmax)

        # plotting model predicted values
        fig = plt.figure()
        # fig.suptitle( ' Model vs Map Prediction ' )
        fig.suptitle(' KNNR Model vs Map Prediction ')
        ax = fig.add_subplot(111)
        qc0 = np.polyfit(y,ypred,1)
        yvi0 = np.polyval(qc0,xvi)
        plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
        plt.plot(xvi,yvi0,c='b',lw=2)

        # plot map predicted values
        fig.subplots_adjust(top=0.9)
        qc1 = np.polyfit(y,zw,1)
        yvi1 = np.polyval(qc1,xvi)
        plt.scatter(y,zw,alpha=0.5,c='r',label='Map Predicted')
        plt.plot(xvi,yvi1,c='r',lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        # ytitle = 'Predicted'
        # dp = 3

        ax.annotate('Model = %-.*f * Actual + %-.*f' %
            (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
            textcoords='figure fraction', fontsize=10)
        ax.annotate('Map = %-.*f * Actual + %-.*f' %
            (2,qc1[0],2,qc1[1]),xy=(xvi[0],yvi1[0]),xytext=(0.14,0.81),
            textcoords='figure fraction', fontsize=10)

        ax.annotate('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
            (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.77),
            textcoords='figure fraction', fontsize=10)
        ax.annotate('Map Pearson cc = %-.*f   Pearson p = %-.*f' %
            (2,ccmap[0],3,ccmap[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.73),
            textcoords='figure fraction', fontsize=10)

        plt.legend(loc='lower right')
        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl2 = os.path.join(cmdloutdir,fname) + "K%-d" % (cmdlkneighbors) + "_knnr.pdf"
            xyplt2 = os.path.join(cmdloutdir,fname) + "K%-d" % (cmdlkneighbors) + "_knnr.csv"
        else:
            pdfcl2 = os.path.join(dirsplit,fname) + "K%-d" % (cmdlkneighbors) + "_knnr.pdf"
            xyplt2 = os.path.join(dirsplit,fname) + "K%-d" % (cmdlkneighbors) + "_knnr.csv"
        fig = ax.get_figure()
        fig.savefig(pdfcl2)
        print('Model Predicted Line:',qc0)
        print('Map Predicted Line:',qc1)
        # xpltcols1 =['Actual','ModelPredicted','MapPredicted']
        xpltdf1 = swa.iloc[:,:3].copy()
        # copy well x y
        xpltdf1['Actual'] = y
        xpltdf1['ModelPredicted'] = ypred
        xpltdf1['MapPredicted'] = zw
        xpltdf1.to_csv(xyplt2,index=False)
        print('Sucessfully generated xplot file %s' % xyplt2)

    else:
        print('******No map prediction plot because of generate samples option')

    savefiles(seisf=cmdlseisattribcsv,
        sdf=ssa, sxydf=ssxyz,
        outdir=cmdloutdir,
        ssuffix='_KNN',name2merge=cmdlwellattribcsv)

def process_TuneCatBoostRegressor(cmdlwellattribcsv,cmdlseisattribcsv,
        cmdlwcolsrange=None,
        cmdlwpredictorcols=None,
        cmdlwtargetcol=None,
        cmdlsaxyzcols=None,
        cmdlscolsrange=None,
        cmdlspredictorcols=None,
        cmdlwellsxyzcols=None,
        cmdlminmaxscale=None,
        cmdloutdir=None,
        cmdliterations=None,
        cmdllearningrate=None,
        cmdldepth=None,
        cmdlcv=None,
        cmdlhideplot=False):
    """Tuning hyper parameters of CatBoostRegression."""
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        # colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic Predictors From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        # colnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic Predictor cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # colnames = swa[swa.columns[cmdlspredictorcols]].columns

    params = {'iterations': cmdliterations,
        'learning_rate': cmdllearningrate,
        'depth': cmdldepth}
    grdcv = GridSearchCV(CatBoostRegressor(loss_function='RMSE'),params,cv=cmdlcv)

    # Fit model
    grdcv.fit(X, y)
    print(grdcv.best_params_)
    clf = grdcv.best_estimator_
    print(grdcv.best_estimator_)
    # Get predictions
    ypred = clf.predict(X)

    msev = np.mean((ypred - y)**2)
    print('Metrics on Well data: ')
    print('Well Data Best Estimator MSE: %.4f' % (msev))
    r2v = r2_score(y,ypred)
    print('Well Data Best Estimator R2 : %10.3f' % r2v)

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    pred = clf.predict(Xpred)
    # all seismic using optimum params

    if cmdlminmaxscale:
        ymin,ymax = y.min(), y.max()
        mmscale = MinMaxScaler((ymin,ymax))
        # mmscale.fit(pred)
        pred1 = pred.reshape(-1,1)
        predscaled = mmscale.fit_transform(pred1)
        ssa['CatBoostPred'] = predscaled
        ssxyz['CatBoostPred'] = predscaled
    else:
        ssa['CatBoostPred'] = pred
        ssxyz['CatBoostPred'] = pred

    # ax =plt.scatter(y,ypred)
    sns.set(color_codes=True)
    ax = sns.regplot(x=y,y=ypred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('CatBoostRegressor %s' % swa.columns[cmdlwtargetcol])
    if not cmdlhideplot:
        plt.show()
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + "_cbreg.pdf"
        xyplt = os.path.join(cmdloutdir,fname) + "_cbrxplt.csv"
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_cbreg.pdf"
        xyplt = os.path.join(dirsplit,fname) + "_cbrxplt.csv"
    fig = ax.get_figure()
    fig.savefig(pdfcl)

    # xpltcols =['Actual','Predicted']
    xpltdf = swa.iloc[:,:3].copy()
    # copy well x y
    xpltdf['Actual'] = y
    xpltdf['Predicted'] = ypred
    xpltdf.to_csv(xyplt,index=False)
    print('Sucessfully generated xplot file %s' % xyplt)

    savefiles(seisf=cmdlseisattribcsv,
        sdf=ssa, sxydf=ssxyz,
        outdir=cmdloutdir,
        ssuffix='_stcbr',name2merge=cmdlwellattribcsv)

# **********NuSVR support vector regresssion: uses nusvr
def process_NuSVR(cmdlwellattribcsv,
        cmdlseisattribcsv,
        cmdlwcolsrange=None,
        cmdlwpredictorcols=None,
        cmdlwtargetcol=None,
        cmdlsaxyzcols=None,
        cmdlscolsrange=None,
        cmdlspredictorcols=None,
        cmdlminmaxscale=None,
        cmdloutdir=None,
        cmdlnu=None,
        cmdlerrpenalty=None,
        cmdlcv=None,
        cmdlscaleminmaxvalues=None,
        cmdlhideplot=False,
        cmdlvalsize=0.3,
        cmdlradius=None,
        cmdlinterpolate='idw',
        # hard coded interpolation method
        cmdlgeneratesamples=None):
    """Nu SVR Support Vector Machine Regression."""
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        # colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='r',func='svr')
    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    if cmdlscolsrange:
        print('Seismic Predictors From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        # colnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic Predictor cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # colnames = swa[swa.columns[cmdlspredictorcols]].columns

    if cmdlcv:
        model = NuSVR(C=cmdlerrpenalty, nu=cmdlnu)
        cvscore = cross_val_score(model,X,y,cv=cmdlcv,scoring='r2')
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        print("Mean Score R2: %10.4f" % (np.mean(cvscore[0])))
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        cvscore = cross_val_score(model,X,y,cv=cmdlcv,scoring='neg_mean_squared_error')
        print("Mean MSE: %10.4f" % (np.mean(np.abs(cvscore[1]))))
        print('No files will be generated. Re-run without cross validation')

    else:
        model = NuSVR(C=cmdlerrpenalty, nu=cmdlnu)
        # Fit model
        model.fit(X, y)
        # Get predictions
        ypred = model.predict(X)
        # Calculating Mean Squared Error
        mse = np.mean((ypred - y)**2)
        print('Metrics on input data: ')
        print('MSE: %.4f' % (mse))
        r2 = r2_score(y,ypred)
        print('R2 : %10.3f' % r2)
        ccmdl = sts.pearsonr(y,ypred)
        print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (ccmdl[0],ccmdl[1]))
        pred = model.predict(Xpred)

        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        model.fit(Xtrain, ytrain)
        print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
        qc0 = np.polyfit(y,ypred,1)
        xrngmin,xrngmax = y.min(),y.max()
        xvi = np.linspace(xrngmin,xrngmax)

        yvalpred = model.predict(Xval)
        # Calculating Mean Squared Error
        msev = np.mean((yvalpred - yval)**2)
        print('Metrics on Train-Test-Split data: ')
        print('Train-Test-Split MSE: %.4f' % (msev))
        r2v = r2_score(yval,yvalpred)
        print('Train-Test-Split R2 : %10.3f' % r2v)

        if cmdlminmaxscale:
            if cmdlscaleminmaxvalues:
                ymin,ymax = cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]
                print('****Manual Scaling of output data to min: %10.4f,  max: %10.4f'
                    % (cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]))
            else:
                ymin,ymax = y.min(), y.max()
            mmscale = MinMaxScaler((ymin,ymax))
            # mmscale.fit(pred)
            pred1 = pred.reshape(-1,1)
            predscaled = mmscale.fit_transform(pred1)
            ssa['NUSVRPred'] = predscaled
            ssxyz['NUSVRPred'] = predscaled
        else:
            ssa['NUSVRPred'] = pred
            ssxyz['NUSVRPred'] = pred

        # ax =plt.scatter(y,ypred)
        # sns.set(color_codes=True)
        # ax =sns.regplot(x=y,y=ypred)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        yvi0 = np.polyval(qc0,xvi)
        plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
        plt.plot(xvi,yvi0,c='k',lw=2)

        ax.annotate('Model = %-.*f * Actual + %-.*f' %
            (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
            textcoords='figure fraction', fontsize=10)
        ax.annotate('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
            (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.81),
            textcoords='figure fraction', fontsize=10)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('NuSVR %s  C=%.1f  nu=%.1f' % (swa.columns[cmdlwtargetcol],cmdlerrpenalty,cmdlnu))

        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "C%.1fnu%.1f" % (cmdlerrpenalty,cmdlnu) + "_nusvr.pdf"
            xyplt = os.path.join(cmdloutdir,fname) + "_nusvrxplt.csv"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "C%.1fnu%.1f" % (cmdlerrpenalty,cmdlnu) + "_nusvr.pdf"
            xyplt = os.path.join(dirsplit,fname) + "_nusvrxplt.csv"
        fig = ax.get_figure()
        fig.savefig(pdfcl)

        if not cmdlgeneratesamples:
            # xpltcols = ['Actual','Predicted']
            xpltdf = swa.iloc[:,:3].copy()
            # copy well x y
            xpltdf['Actual'] = y
            xpltdf['Predicted'] = ypred
            xpltdf.to_csv(xyplt,index=False)
            print('Sucessfully generated xplot file %s' % xyplt)

            # map back interpolate
            xs = ssxyz.iloc[:,0].values
            ys = ssxyz.iloc[:,1].values
            xys = np.transpose(np.vstack((xs,ys)))
            xw = swa.iloc[:,cmdlsaxyzcols[1]].values
            yw = swa.iloc[:,cmdlsaxyzcols[2]].values
            xyw = np.transpose(np.vstack((xw,yw)))

            print('******Map Back Interpolation')
            zw = map2ddata(xys,pred,xyw,cmdlradius,cmdlinterpolate)
            ccmap = sts.pearsonr(y,zw)
            print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (ccmap[0],ccmap[1]))

            xrngmin,xrngmax = y.min(),y.max()
            xvi = np.linspace(xrngmin,xrngmax)

            # plotting model predicted values
            fig = plt.figure()
            # fig.suptitle( ' Model vs Map Prediction ' )
            # fig.suptitle( ' NuSVR Model vs Map Prediction ')
            plt.title('NuSVR Model vs Map Prediction %s  C=%.1f  nu=%.1f' % (swa.columns[cmdlwtargetcol],cmdlerrpenalty,cmdlnu))
            ax = fig.add_subplot(111)
            qc0 = np.polyfit(y,ypred,1)
            yvi0 = np.polyval(qc0,xvi)
            plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
            plt.plot(xvi,yvi0,c='b',lw=2)

            # plot map predicted values
            fig.subplots_adjust(top=0.9)
            qc1 = np.polyfit(y,zw,1)
            yvi1 = np.polyval(qc1,xvi)
            plt.scatter(y,zw,alpha=0.5,c='r',label='Map Predicted')
            plt.plot(xvi,yvi1,c='r',lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            # ytitle = 'Predicted'
            dp = 3

            ax.annotate('Model = %-.*f * Actual + %-.*f' %
                (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
                textcoords='figure fraction', fontsize=10)
            ax.annotate('Map = %-.*f * Actual + %-.*f' %
                (2,qc1[0],2,qc1[1]),xy=(xvi[0],yvi1[0]),xytext=(0.14,0.81),
                textcoords='figure fraction', fontsize=10)

            ax.annotate('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
                (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.77),
                textcoords='figure fraction', fontsize=10)
            ax.annotate('Map Pearson cc = %-.*f   Pearson p = %-.*f' %
                (2,ccmap[0],3,ccmap[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.73),
                textcoords='figure fraction', fontsize=10)

            plt.legend(loc='lower right')
            if not cmdlhideplot:
                plt.show()
            if cmdloutdir:
                pdfcl2 = os.path.join(cmdloutdir,fname) + "C%.1fnu%.1f" % (cmdlerrpenalty,cmdlnu) + "_nusvr2.pdf"
                xyplt2 = os.path.join(cmdloutdir,fname) + "C%.1fnu%.1f" % (cmdlerrpenalty,cmdlnu) + "_nusvrxplt2.csv"
            else:
                pdfcl2 = os.path.join(dirsplit,fname) + "C%.1fnu%.1f" % (cmdlerrpenalty,cmdlnu) + "_nusvr2.pdf"
                xyplt2 = os.path.join(dirsplit,fname) + "C%.1fnu%.1f" % (cmdlerrpenalty,cmdlnu) + "_nusvrxplt2.csv"
            fig = ax.get_figure()
            fig.savefig(pdfcl2)
            print('Model Predicted Line:',qc0)
            print('Map Predicted Line:',qc1)
            # xpltcols1 = ['Actual','ModelPredicted','MapPredicted']
            xpltdf1 = swa.iloc[:,:3].copy()
            # copy well x y
            xpltdf1['Actual'] = y
            xpltdf1['ModelPredicted'] = ypred
            xpltdf1['MapPredicted'] = zw
            xpltdf1.to_csv(xyplt2,index=False)
            print('Sucessfully generated xplot file %s' % xyplt2)

            # box plot of actual vs model vs map predictions
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # plt.boxplot([y,ypred,zw],labels=['Actual','Model','Map'],showmeans=True,notch=True)
            ax.boxplot([y,ypred,zw],labels=['Actual','Model','Map'],showmeans=True,notch=True)
            plt.title('NuSVR for %s C%.1f nu%.1f' % (swa.columns[cmdlwtargetcol],cmdlerrpenalty,cmdlnu))
            if not cmdlhideplot:
                plt.show()
            if cmdloutdir:
                pdfcl3 = os.path.join(cmdloutdir,fname) +"C%.1fnu%.1f" % (cmdlerrpenalty,cmdlnu) + "_nusvr2bx.pdf"
            else:
                pdfcl3 = os.path.join(dirsplit,fname) + "C%.1fnu%.1f" % (cmdlerrpenalty,cmdlnu) + "_nusvr2bx.pdf"
            fig = ax.get_figure()
            fig.savefig(pdfcl3)

        else:
            print('******No map prediction plot because of generate samples option')

        savefiles(seisf=cmdlseisattribcsv,
            sdf=ssa, sxydf=ssxyz,
            outdir=cmdloutdir,
            ssuffix='_NUSVR',name2merge=cmdlwellattribcsv)

def process_SGDR(cmdlwellattribcsv,cmdlseisattribcsv,
                cmdlwcolsrange=None,
                cmdlwpredictorcols=None,
                cmdlwtargetcol=None,
                cmdlsaxyzcols=None,
                cmdlscolsrange=None,
                cmdlspredictorcols=None,
                cmdlminmaxscale=None,
                cmdloutdir=None,
                cmdlloss=None,
                # ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’
                cmdlpenalty='l2',
                # options: l1,l2,elasticnet,none
                cmdll1ratio=0.15,
                # elastic net mixing: 0 (l2)to 1 (l1)
                cmdlcv=None,
                cmdlscaleminmaxvalues=None,
                cmdlhideplot=False,
                cmdlvalsize=0.3,
                cmdlradius=None,
                cmdlinterpolate='idw',
                # hard coded interpolation method
                cmdlgeneratesamples=None):
    """Stochastic Gradient Descent Regressor OLS/L1/L2 regresssion."""
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        # colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='r',func='svr')
    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    if cmdlscolsrange:
        print('Seismic Predictors From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        # colnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic Predictor cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # colnames = swa[swa.columns[cmdlspredictorcols]].columns

    if cmdlcv:
        model = SGDRegressor(loss=cmdlloss,penalty=cmdlpenalty,l1_ratio=cmdll1ratio)
        cvscore = cross_val_score(model,X,y,cv=cmdlcv,scoring='r2')
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        print("Mean Score R2: %10.4f" % (np.mean(cvscore[0])))
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        cvscore = cross_val_score(model,X,y,cv=cmdlcv,scoring='neg_mean_squared_error')
        print("Mean MSE: %10.4f" % (np.mean(np.abs(cvscore[1]))))
        print('No files will be generated. Re-run without cross validation')

    else:
        model = SGDRegressor(loss=cmdlloss,penalty=cmdlpenalty,l1_ratio=cmdll1ratio)
        # Fit model
        model.fit(X, y)
        # Get predictions
        ypred = model.predict(X)
        # Calculating Mean Squared Error
        mse = np.mean((ypred - y)**2)
        print('Metrics on input data: ')
        print('MSE: %.4f' % (mse))
        r2 = r2_score(y,ypred)
        print('R2 : %10.3f' % r2)
        ccmdl = sts.pearsonr(y,ypred)
        print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (ccmdl[0],ccmdl[1]))
        pred = model.predict(Xpred)

        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        model.fit(Xtrain, ytrain)
        print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
        qc0 = np.polyfit(y,ypred,1)
        xrngmin,xrngmax = y.min(),y.max()
        xvi = np.linspace(xrngmin,xrngmax)

        yvalpred = model.predict(Xval)
        # Calculating Mean Squared Error
        msev = np.mean((yvalpred - yval)**2)
        print('Metrics on Train-Test-Split data: ')
        print('Train-Test-Split MSE: %.4f' % (msev))
        r2v = r2_score(yval,yvalpred)
        print('Train-Test-Split R2 : %10.3f' % r2v)

        if cmdlminmaxscale:
            if cmdlscaleminmaxvalues:
                ymin,ymax = cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]
                print('****Manual Scaling of output data to min: %10.4f,  max: %10.4f'
                    % (cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]))
            else:
                ymin,ymax = y.min(), y.max()
            mmscale = MinMaxScaler((ymin,ymax))
            # mmscale.fit(pred)
            pred1 = pred.reshape(-1,1)
            predscaled = mmscale.fit_transform(pred1)
            ssa['SGDRPred'] = predscaled
            ssxyz['SGDRPred'] = predscaled
        else:
            ssa['SGDRPred'] = pred
            ssxyz['SGDRPred'] = pred

        # ax =plt.scatter(y,ypred)
        # sns.set(color_codes=True)
        # ax =sns.regplot(x=y,y=ypred)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        yvi0 = np.polyval(qc0,xvi)
        plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
        plt.plot(xvi,yvi0,c='k',lw=2)

        ax.annotate('Model = %-.*f * Actual + %-.*f' %
            (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
            textcoords='figure fraction', fontsize=10)
        ax.annotate('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
            (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.81),
            textcoords='figure fraction', fontsize=10)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('SGDR %s  Loss=%s  Penalty=%s l1ratio=%.1f' % (swa.columns[cmdlwtargetcol],cmdlloss,cmdlpenalty,cmdll1ratio))

        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "L%sP%sR%.1f" % (cmdlloss,cmdlpenalty,cmdll1ratio) + "_sgdr.pdf"
            xyplt = os.path.join(cmdloutdir,fname) + "_sgdrrxplt.csv"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "L%sP%sR%.1f" % (cmdlloss,cmdlpenalty,cmdll1ratio) + "_sgdrr.pdf"
            xyplt = os.path.join(dirsplit,fname) + "_sgdrrxplt.csv"
        fig = ax.get_figure()
        fig.savefig(pdfcl)

        if not cmdlgeneratesamples:
            # xpltcols = ['Actual','Predicted']
            xpltdf = swa.iloc[:,:3].copy()
            # copy well x y
            xpltdf['Actual'] = y
            xpltdf['Predicted'] = ypred
            xpltdf.to_csv(xyplt,index=False)
            print('Sucessfully generated xplot file %s' % xyplt)

            # map back interpolate
            xs = ssxyz.iloc[:,0].values
            ys = ssxyz.iloc[:,1].values
            xys = np.transpose(np.vstack((xs,ys)))
            xw = swa.iloc[:,cmdlsaxyzcols[1]].values
            yw = swa.iloc[:,cmdlsaxyzcols[2]].values
            xyw = np.transpose(np.vstack((xw,yw)))

            print('******Map Back Interpolation')
            zw = map2ddata(xys,pred,xyw,cmdlradius,cmdlinterpolate)
            ccmap = sts.pearsonr(y,zw)
            print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (ccmap[0],ccmap[1]))

            xrngmin,xrngmax = y.min(),y.max()
            xvi = np.linspace(xrngmin,xrngmax)

            # plotting model predicted values
            fig = plt.figure()
            # fig.suptitle( ' Model vs Map Prediction ' )
            fig.suptitle(' SGDR Model vs Map Prediction ')
            ax = fig.add_subplot(111)
            qc0 = np.polyfit(y,ypred,1)
            yvi0 = np.polyval(qc0,xvi)
            plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
            plt.plot(xvi,yvi0,c='b',lw=2)

            # plot map predicted values
            fig.subplots_adjust(top=0.9)
            qc1 = np.polyfit(y,zw,1)
            yvi1 = np.polyval(qc1,xvi)
            plt.scatter(y,zw,alpha=0.5,c='r',label='Map Predicted')
            plt.plot(xvi,yvi1,c='r',lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            # ytitle = 'Predicted'
            dp = 3

            ax.annotate('Model = %-.*f * Actual + %-.*f' %
                (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
                textcoords='figure fraction', fontsize=10)
            ax.annotate('Map = %-.*f * Actual + %-.*f' %
                (2,qc1[0],2,qc1[1]),xy=(xvi[0],yvi1[0]),xytext=(0.14,0.81),
                textcoords='figure fraction', fontsize=10)

            ax.annotate('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
                (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.77),
                textcoords='figure fraction', fontsize=10)
            ax.annotate('Map Pearson cc = %-.*f   Pearson p = %-.*f' %
                (2,ccmap[0],3,ccmap[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.73),
                textcoords='figure fraction', fontsize=10)

            plt.legend(loc='lower right')
            if not cmdlhideplot:
                plt.show()
            if cmdloutdir:
                pdfcl2 = os.path.join(cmdloutdir,fname) + "L%sP%sR%.1f" % (cmdlloss,cmdlpenalty,cmdll1ratio) + "_sgdr2.pdf"
                xyplt2 = os.path.join(cmdloutdir,fname) + "L%sP%sR%.1f" % (cmdlloss,cmdlpenalty,cmdll1ratio) + "_sgdrxplt2.csv"
            else:
                pdfcl2 = os.path.join(dirsplit,fname) + "L%sP%sR%.1f" % (cmdlloss,cmdlpenalty,cmdll1ratio) + "_sgdr2.pdf"
                xyplt2 = os.path.join(dirsplit,fname) + "L%sP%sR%.1f" % (cmdlloss,cmdlpenalty,cmdll1ratio) + "_sgdrxplt2.csv"
            fig = ax.get_figure()
            fig.savefig(pdfcl2)
            print('Model Predicted Line:',qc0)
            print('Map Predicted Line:',qc1)
            # xpltcols1 = ['Actual','ModelPredicted','MapPredicted']
            xpltdf1 = swa.iloc[:,:3].copy()
            # copy well x y
            xpltdf1['Actual'] = y
            xpltdf1['ModelPredicted'] = ypred
            xpltdf1['MapPredicted'] = zw
            xpltdf1.to_csv(xyplt2,index=False)
            print('Sucessfully generated xplot file %s' % xyplt2)

        else:
            print('******No map prediction plot because of generate samples option')

        savefiles(seisf=cmdlseisattribcsv,
            sdf=ssa, sxydf=ssxyz,
            outdir=cmdloutdir,
            ssuffix='_SGDR',name2merge=cmdlwellattribcsv)

# **********CatBoostRegressor
def process_CatBoostRegressor(cmdlwellattribcsv,cmdlseisattribcsv,
                cmdlwcolsrange=None,
                cmdlwpredictorcols=None,
                cmdlwtargetcol=None,
                cmdlsaxyzcols=None,
                cmdlscolsrange=None,
                cmdlspredictorcols=None,
                cmdlminmaxscale=None,
                cmdloutdir=None,
                cmdliterations=None,
                cmdllearningrate=None,
                cmdldepth=None,
                cmdlcv=None,
                cmdlscaleminmaxvalues=None,
                cmdlfeatureimportance=None,
                cmdlgeneratesamples=None,
                cmdlhideplot=False,
                cmdlvalsize=0.3,
                cmdlnofilesout=False,
                cmdlradius=None,
                cmdlinterpolate='idw',
                # hard coded interpolation method
                cmdloverfittingdetection=False,
                cmdlodpval=None):
    """CatBoostRegression."""
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='r',func='cbr')

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    if cmdlscolsrange:
        print('Seismic Predictors From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        colnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic Predictor cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        colnames = swa[swa.columns[cmdlspredictorcols]].columns

    if cmdlfeatureimportance:
        model = CatBoostRegressor(iterations=cmdliterations, learning_rate=cmdllearningrate,
                    depth=cmdldepth,loss_function='RMSE',calc_feature_importance=True,random_seed=42)
        model.fit(X, y)
        fr = pd.DataFrame(sorted(zip(model.get_feature_importance(X,y), colnames),reverse=True),columns=['Importance','Attribute'])
        print('Feature Ranking with CatBoostRegressor: ')
        print(fr)

        plt.style.use('seaborn-whitegrid')
        ax = fr['Importance'].plot(kind='bar', figsize=(12,8))
        ax.set_xticklabels(fr['Attribute'],rotation=45)
        ax.set_ylabel('Feature Importance')
        ax.set_title('CatBoostRegressor Feature Importance of %s' % cmdlwellattribcsv)
        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_cbrfi.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_cbrfi.pdf"
        fig = ax.get_figure()
        fig.savefig(pdfcl)

    elif cmdlcv:
        model = CatBoostRegressor(iterations=cmdliterations, learning_rate=cmdllearningrate,
                    depth=cmdldepth,loss_function='RMSE',random_seed=42)
        cvscore = cross_val_score(model,X,y,cv=cmdlcv,scoring='r2')
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        print("Mean Score R2: %10.4f" % (np.mean(cvscore[0])))
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        cvscore = cross_val_score(model,X,y,cv=cmdlcv,scoring='neg_mean_squared_error')
        print("Mean MSE: %10.4f" % (np.mean(np.abs(cvscore[1]))))
        print('No files will be generated. Re-run without cross validation')

    # does not work: problem with eval_set variable Mar 1 2018
    elif cmdloverfittingdetection:
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
        evalset = (Xtest,ytest)
        model = CatBoostRegressor(iterations=cmdliterations,
                    learning_rate=cmdllearningrate,
                    depth=cmdldepth,
                    loss_function='RMSE',
                    use_best_model=True,
                    od_type='IncToDec',
                    od_pval=cmdlodpval,
                    eval_metric='RMSE',
                    random_seed=42)

        # Fit model
        model.fit(X, y,eval_set=evalset)
        # Get predictions
        ypred = model.predict(X)
        # Calculating Mean Squared Error
        mse = np.mean((ypred - y)**2)
        print('Metrics on input data: ')
        print('MSE: %.4f' % (mse))
        r2 = r2_score(y,ypred)
        print('R2 : %10.3f' % r2)
        print(model.get_params())
        # model.save_model('CBRmodel.mdl')
        print('No files will be generated. Re-run without overfittingdetection')
    else:
        model = CatBoostRegressor(iterations=cmdliterations, learning_rate=cmdllearningrate,
                depth=cmdldepth,loss_function='RMSE',random_seed=42)
        # Fit model
        model.fit(X, y)
        # Get predictions
        ypred = model.predict(X)
        # Calculating Mean Squared Error
        mse = np.mean((ypred - y)**2)
        print('Metrics on input data: ')
        print('MSE: %.4f' % (mse))
        r2 = r2_score(y,ypred)
        print('R2 : %10.3f' % r2)
        ccmdl = sts.pearsonr(y,ypred)
        print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (ccmdl[0],ccmdl[1]))
        qc0 = np.polyfit(y,ypred,1)
        xrngmin,xrngmax = y.min(),y.max()
        xvi = np.linspace(xrngmin,xrngmax)

        pred = model.predict(Xpred)

        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        model.fit(Xtrain, ytrain)
        print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))

        yvalpred = model.predict(Xval)

        # Calculating Mean Squared Error
        msev = np.mean((yvalpred - yval)**2)
        print('Metrics on Train-Test-Split data: ')
        print('Train-Test-Split MSE: %.4f' % (msev))
        r2v = r2_score(yval,yvalpred)
        print('Train-Test-Split R2 : %10.3f' % r2v)
        ccxv = sts.pearsonr(yval,yvalpred)
        print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (ccxv[0],ccxv[1]))

        if cmdlminmaxscale:
            if cmdlscaleminmaxvalues:
                ymin,ymax = cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]
                print('****Manual Scaling of output data to min: %10.4f,  max: %10.4f'
                    % (cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]))
            else:
                ymin,ymax = y.min(), y.max()
            mmscale = MinMaxScaler((ymin,ymax))
            # mmscale.fit(pred)
            pred1 = pred.reshape(-1,1)
            predscaled = mmscale.fit_transform(pred1)
            ssa['CatBoostPred'] = predscaled
            ssxyz['CatBoostPred'] = predscaled
        else:
            ssa['CatBoostPred'] = pred
            ssxyz['CatBoostPred'] = pred

        # ax =plt.scatter(y,ypred)
        # sns.set(color_codes=True)
        # ax =sns.regplot(x=y,y=ypred)
        # fig.suptitle( ' CBR Model I %-.*f  LR %-.*f  D %-.*f ' %
        #     (0,cmdliterations,2,cmdllearningrate,0,cmdldepth))
        fig = plt.figure()

        ax = fig.add_subplot(111)
        # qc0 = np.polyfit(y,ypred,1) #has already been calculated above
        yvi0 = np.polyval(qc0,xvi)
        plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
        plt.plot(xvi,yvi0,c='k',lw=2)

        ax.annotate('Model = %-.*f * Actual + %-.*f' %
            (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
            textcoords='figure fraction', fontsize=10)
        ax.annotate ('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
            (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.81),
            textcoords='figure fraction', fontsize=10)
        plt.title('CatBoostRegressor %s' % swa.columns[cmdlwtargetcol])
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "I%-.*fLR%-.*fD%-.*f" % (0,cmdliterations,2,cmdllearningrate,0,cmdldepth) + "_cbreg.pdf"
            xyplt = os.path.join(cmdloutdir,fname) + "I%-.*fLR%-.*fD%-.*f" % (0,cmdliterations,2,cmdllearningrate,0,cmdldepth) + "_cbrxplt.csv"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "I%-.*fLR%-.*fD%-.*f" % (0,cmdliterations,2,cmdllearningrate,0,cmdldepth) + "_cbreg.pdf"
            xyplt = os.path.join(dirsplit,fname) + "I%-.*fLR%-.*fD%-.*f" % (0,cmdliterations,2,cmdllearningrate,0,cmdldepth) + "_cbrxplt.csv"
        fig = ax.get_figure()
        fig.savefig(pdfcl)

        if not cmdlgeneratesamples:
            # xpltcols = ['Actual','Predicted']
            xpltdf = swa.iloc[:,:3].copy()
            # copy well x y
            xpltdf['Actual'] = y
            xpltdf['Predicted'] = ypred
            xpltdf.to_csv(xyplt,index=False)
            print('Sucessfully generated xplot file %s' % xyplt)

            # map back interpolate
            xs = ssxyz.iloc[:,0].values
            ys = ssxyz.iloc[:,1].values
            xys = np.transpose(np.vstack((xs,ys)))
            xw = swa.iloc[:,cmdlsaxyzcols[1]].values
            yw = swa.iloc[:,cmdlsaxyzcols[2]].values
            xyw = np.transpose(np.vstack((xw,yw)))

            print('******Map Back Interpolation')
            zw = map2ddata(xys,pred,xyw,cmdlradius,cmdlinterpolate)
            ccmap = sts.pearsonr(y,zw)
            print('Train-Test-Split Pearsonr : %10.3f  %10.4f ' % (ccmap[0],ccmap[1]))

            # xrngmin,xrngmax = y.min(),y.max()
            # xvi = np.linspace(xrngmin,xrngmax)

            # plotting model predicted values
            fig = plt.figure()
            # fig.suptitle( ' Model vs Map Prediction ' )
            fig.suptitle(' Model vs Map Prediction I %-.*f  LR %-.*f  D %-.*f ' %
                    (0,cmdliterations,2,cmdllearningrate,0,cmdldepth))
            ax = fig.add_subplot(111)
            # qc0 = np.polyfit(y,ypred,1) #has already been calculated above
            yvi0 = np.polyval(qc0,xvi)
            plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
            plt.plot(xvi,yvi0,c='b',lw=2)

            # plot map predicted values
            fig.subplots_adjust(top=0.9)
            qc1 = np.polyfit(y,zw,1)
            yvi1 = np.polyval(qc1,xvi)
            plt.scatter(y,zw,alpha=0.5,c='r',label='Map Predicted')
            plt.plot(xvi,yvi1,c='r',lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            # ytitle = 'Predicted'
            # dp = 3
            # commenting out annotation : only shows on last plot!!
            # plt.annotate('%s = %-.*f   + %-.*f * %s' % (ytitle,dp,qc[0],dp,qc[1],xtitle),xy=(yvi[4],xvi[4]),xytext=(0.25,0.80),textcoords='figure fraction')

            ax.annotate('Model = %-.*f * Actual + %-.*f' %
                (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
                textcoords='figure fraction', fontsize=10)
            ax.annotate('Map = %-.*f * Actual + %-.*f' %
                (2,qc1[0],2,qc1[1]),xy=(xvi[0],yvi1[0]),xytext=(0.14,0.81),
                textcoords='figure fraction', fontsize=10)

            ax.annotate('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
                (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.77),
                textcoords='figure fraction', fontsize=10)
            ax.annotate('Map Pearson cc = %-.*f   Pearson p = %-.*f' %
                (2,ccmap[0],3,ccmap[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.73),
                textcoords='figure fraction', fontsize=10)

            plt.legend(loc='lower right')
            if not cmdlhideplot:
                plt.show()
            if cmdloutdir:
                pdfcl2 = os.path.join(cmdloutdir,fname) + "_cbreg2.pdf"
                xyplt2 = os.path.join(cmdloutdir,fname) + "_cbrxplt2.csv"
            else:
                pdfcl2 = os.path.join(dirsplit,fname) + "_cbreg2.pdf"
                xyplt2 = os.path.join(dirsplit,fname) + "_cbrxplt2.csv"
            fig = ax.get_figure()
            fig.savefig(pdfcl2)
            print('Model Predicted Line:',qc0)
            print('Pearson CC : %-.*f  %-.*f' % (2,ccmdl[0],2,ccmdl[1]))

            print('Map Predicted Line:',qc1)
            print('Pearson CC : %-.*f  %-.*f' % (2,ccmap[0],2,ccmap[1]))
            # xpltcols1 = ['Actual','ModelPredicted','MapPredicted']
            xpltdf1 = swa.iloc[:,:3].copy()
            # copy well x y
            xpltdf1['Actual'] = y
            xpltdf1['ModelPredicted'] = ypred
            xpltdf1['MapPredicted'] =zw
            xpltdf1.to_csv(xyplt2,index=False)
            print('Sucessfully generated xplot file %s'  % xyplt2)

            # box plot of actual vs model vs map predictions
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.boxplot([y,ypred,zw],labels=['Actual','Model','Map'],showmeans=True,notch=True)
            plt.title('CatBoostRegressor for %s I %-.*f  LR %-.*f  D %-.*f' % (swa.columns[cmdlwtargetcol],0,cmdliterations,2,cmdllearningrate,0,cmdldepth))
            if not cmdlhideplot:
                plt.show()
            if cmdloutdir:
                pdfcl3 = os.path.join(cmdloutdir,fname) + "I%-.*fLR%-.*fD%-.*f" % (0,cmdliterations,2,cmdllearningrate,0,cmdldepth) + "_cbreg2bx.pdf"
            else:
                pdfcl3 = os.path.join(dirsplit,fname) + "I%-.*fLR%-.*fD%-.*f" % (0,cmdliterations,2,cmdllearningrate,0,cmdldepth) + "_cbreg2bx.pdf"
            fig = ax.get_figure()
            fig.savefig(pdfcl3)

        else:
            print('******No map prediction plot because of generate samples option')

        if cmdlnofilesout:
            print('No data files will be saved')
        else:
            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                outdir=cmdloutdir,
                ssuffix='_CBR',name2merge=cmdlwellattribcsv)

def process_ANNRegressor(cmdlwellattribcsv,cmdlseisattribcsv,
        cmdlwcolsrange=None,
        cmdlwpredictorcols=None,
        cmdlwtargetcol=None,
        cmdlsaxyzcols=None,
        cmdlscolsrange=None,
        cmdlspredictorcols=None,
        cmdlminmaxscale=None,
        cmdloutdir=None,
        cmdliterations=None,
        cmdllearningrate=None,
        cmdlcv=None,
        cmdlscaleminmaxvalues=None,
        # same number as num layers
        cmdlnodes=None,
        # same numberof codes as num layers
        cmdlactivation=None,
        # one number
        cmdlepochs=None,
        # one number
        cmdlbatch=None,
        cmdlhideplot=False,
        cmdlvalsize=0.3,
        cmdlnofilesout=False,
        cmdlgeneratesamples=None,
        cmdlradius=None,
        # hard coded interpolation method
        cmdlinterpolate='idw'):
    """**********ANNRegressor."""
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor

    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1]+1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        # colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='r',func='ann')

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    print('cmdlactivation',cmdlactivation)
    cmdllayers = len(cmdlnodes)

    def build_model():
        indim = cmdlwcolsrange[1] - cmdlwcolsrange[0] + 1
        model = Sequential()
        model.add(Dense(cmdlnodes[0], input_dim=indim, kernel_initializer='normal', activation=cmdlactivation[0]))
        for i in range(1,cmdllayers):
            model.add(Dense(cmdlnodes[i], kernel_initializer='normal', activation=cmdlactivation[i]))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    if cmdlscolsrange:
        print('Seismic Predictors From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        # colnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic Predictor cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # colnames = swa[swa.columns[cmdlspredictorcols]].columns

    if cmdlcv:
        kfold = KFold(n_splits=cmdlcv, random_state=42)
        estimator = KerasRegressor(build_fn=build_model,
                    epochs=cmdlepochs,
                    batch_size=cmdlbatch,
                    verbose=0)
        cvscore = cross_val_score(estimator,X,y,cv=cmdlcv,scoring='r2')
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        print("Mean Score R2: %10.4f" % (np.mean(cvscore[0])))
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        cvscore = cross_val_score(estimator,X,y,cv=cmdlcv,scoring='neg_mean_squared_error')
        print("Mean MSE: %10.4f" % (np.mean(np.abs(cvscore[1]))))
        print('No files will be generated. Re-run without cross validation')

    else:
        estimator = KerasRegressor(build_fn=build_model,
                    epochs=cmdlepochs,
                    batch_size=cmdlbatch,
                    verbose=0)
        estimator.fit(X, y)
        # Get predictions
        ypred = estimator.predict(X)
        # Calculating Mean Squared Error
        mse = np.mean((ypred - y)**2)
        print('Metrics on input data: ')
        print('MSE: %.4f' % (mse))
        r2 = r2_score(y,ypred)
        print('R2 : %10.3f' % r2)
        pred = estimator.predict(Xpred)

        Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
            random_state=42)
        estimator.fit(Xtrain, ytrain)
        print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
        ccmdl = sts.pearsonr(y,ypred)
        qc0 = np.polyfit(y,ypred,1)
        xrngmin,xrngmax = y.min(),y.max()
        xvi = np.linspace(xrngmin,xrngmax)

        yvalpred = estimator.predict(Xval)
        # Calculating Mean Squared Error
        msev = np.mean((yvalpred - yval)**2)
        print('Metrics on Train-Test-Split data: ')
        print('Train-Test-Split MSE: %.4f' % (msev))
        r2v = r2_score(yval,yvalpred)
        print('Train-Test-Split R2 : %10.3f' % r2v)

        if cmdlminmaxscale:
            if cmdlscaleminmaxvalues:
                ymin,ymax = cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]
                print('****Manual Scaling of output data to min: %10.4f,  max: %10.4f'
                    % (cmdlscaleminmaxvalues[0],cmdlscaleminmaxvalues[1]))
            else:
                ymin,ymax = y.min(), y.max()
            mmscale = MinMaxScaler((ymin,ymax))
            # mmscale.fit(pred)
            pred1 = pred.reshape(-1,1)
            predscaled = mmscale.fit_transform(pred1)
            ssa['ANNPred'] = predscaled
            ssxyz['ANNPred'] = predscaled
        else:
            ssa['ANNPred'] = pred
            ssxyz['ANNPred'] = pred

        # ax =plt.scatter(y,ypred)
        # sns.set(color_codes=True)
        # ax =sns.regplot(x=y,y=ypred)
        # plt.xlabel('Actual')
        # plt.ylabel('Predicted')

        fig = plt.figure()
        ax = fig.add_subplot(111)

        yvi0 = np.polyval(qc0,xvi)
        plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
        plt.plot(xvi,yvi0,c='k',lw=2)

        ax.annotate('Model = %-.*f * Actual + %-.*f' %
            (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
            textcoords='figure fraction', fontsize=10)
        ax.annotate('Model Pearson cc = %-.*f   Pearson p = %-.*f' %
            (2,ccmdl[0],3,ccmdl[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.81),
            textcoords='figure fraction', fontsize=10)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('ANNRegressor %s' % swa.columns[cmdlwtargetcol])

        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_annreg.pdf"
            xyplt = os.path.join(cmdloutdir,fname) + "_annrxplt.csv"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_annreg.pdf"
            xyplt = os.path.join(dirsplit,fname) + "_annrxplt.csv"
        fig = ax.get_figure()
        fig.savefig(pdfcl)

        if not cmdlgeneratesamples:
            # xpltcols =['Actual','Predicted']
            xpltdf = swa.iloc[:,:3].copy()
            # copy well x y
            xpltdf['Actual'] = y
            xpltdf['Predicted'] = ypred
            xpltdf.to_csv(xyplt,index=False)
            print('Sucessfully generated xplot file %s' % xyplt)

            # map back interpolate
            xs = ssxyz.iloc[:,0].values
            ys = ssxyz.iloc[:,1].values
            xys = np.transpose(np.vstack((xs,ys)))
            xw = swa.iloc[:,cmdlsaxyzcols[1]].values
            yw = swa.iloc[:,cmdlsaxyzcols[2]].values
            xyw = np.transpose(np.vstack((xw,yw)))

            print('******Map Back Interpolation')
            zw = map2ddata(xys,pred,xyw,cmdlradius,cmdlinterpolate)

            xrngmin,xrngmax = y.min(),y.max()
            xvi = np.linspace(xrngmin,xrngmax)

            # plotting model predicted values
            fig = plt.figure()
            # fig.suptitle( ' Model vs Map Prediction ' )
            fig.suptitle(' Model vs Map Prediction ')
            ax = fig.add_subplot(111)
            qc0 = np.polyfit(y,ypred,1)
            yvi0 = np.polyval(qc0,xvi)
            plt.scatter(y,ypred,alpha=0.5,c='b',label='Model Predicted')
            plt.plot(xvi,yvi0,c='b',lw=2)

            # plot map predicted values
            fig.subplots_adjust(top=0.9)
            qc1 = np.polyfit(y,zw,1)
            yvi1 = np.polyval(qc1,xvi)
            plt.scatter(y,zw,alpha=0.5,c='r',label='Map Predicted')
            plt.plot(xvi,yvi1,c='r',lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            # ytitle = 'Predicted'
            # dp = 3

            ax.annotate('Model = %-.*f * Actual + %-.*f' %
                (2,qc0[0],2,qc0[1]),xy=(xvi[0],yvi0[0]),xytext=(0.14,0.85),
                textcoords='figure fraction', fontsize=10)
            ax.annotate('Map = %-.*f * Actual + %-.*f' %
                (2,qc1[0],2,qc1[1]),xy=(xvi[0],yvi1[0]),xytext=(0.14,0.81),
                textcoords='figure fraction', fontsize=10)

            plt.legend(loc='lower right')
            if not cmdlhideplot:
                plt.show()
            if cmdloutdir:
                pdfcl2 = os.path.join(cmdloutdir,fname) + "_annreg2.pdf"
                xyplt2 = os.path.join(cmdloutdir,fname) + "_annrxplt2.csv"
            else:
                pdfcl2 = os.path.join(dirsplit,fname) + "_annreg2.pdf"
                xyplt2 = os.path.join(dirsplit,fname) + "_annrxplt2.csv"
            fig = ax.get_figure()
            fig.savefig(pdfcl2)
            print('Model Predicted Line:',qc0)
            print('Map Predicted Line:',qc1)
            # xpltcols1 =['Actual','ModelPredicted','MapPredicted']
            xpltdf1 = swa.iloc[:,:3].copy()
            # copy well x y
            xpltdf1['Actual'] = y
            xpltdf1['ModelPredicted'] = ypred
            xpltdf1['MapPredicted'] = zw
            xpltdf1.to_csv(xyplt2,index=False)
            print('Sucessfully generated xplot file %s' % xyplt2)
        else:
            print('******No map prediction plot because of generate samples option')

        savefiles(seisf=cmdlseisattribcsv,
            sdf=ssa, sxydf=ssxyz,
            outdir=cmdloutdir,
            ssuffix='_ANNR',name2merge=cmdlwellattribcsv)

def process_testCmodels(cmdlwellattribcsv,
    cmdlwcolsrange=None,
    cmdlwanalysiscols=None,
    cmdlwtargetcol=None,
    cmdlqcut=None,
    cmdlcv=None,
    cmdloutdir=None,
    cmdlhideplot=None):
    """test various classification models."""
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlwcolsrange:
        print('Well From col# %d to col %d' %(cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1]]].values
        colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1]]].columns
    else:
        print('Well analysis cols',cmdlwanalysiscols)
        X = swa[swa.columns[cmdlwanalysiscols]].values
        colnames = swa[swa.columns[cmdlwanalysiscols]].columns

    probacolnames = ['Class%d' % i for i in range(cmdlqcut)]
    swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
    swa['qcodes'] = swa['qa'].cat.codes
    y = swa['qcodes'].values
    print('Quantile bins: ',qbins)
    qcount = Counter(y)
    print(qcount)

    models = []
    models.append(( ' LR ' , LogisticRegression()))
    models.append(( ' LDA ' , LinearDiscriminantAnalysis()))
    models.append(( ' KNN ' , KNeighborsClassifier()))
    models.append(( ' CART ' , DecisionTreeClassifier()))
    models.append(( ' NB ' , GaussianNB()))
    models.append(( ' SVM ' , SVC()))
    # evaluate each model in turn
    results = []
    names = []
    resultsmean = []
    resultsstd = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = KFold(n_splits=cmdlcv, random_state=7)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "Model: %s: Mean Accuracy: %0.4f Std: (%0.4f)" % (name, cv_results.mean(), cv_results.std())
        #print (msg)
        resultsmean.append(cv_results.mean())
        resultsstd.append(cv_results.std())
    modeltest = pd.DataFrame(list(zip(names,resultsmean,resultsstd)),columns=['Model','Model Mean Accuracy','Accuracy STD'])
    print(modeltest)

    dirsplit,fextsplit = os.path.split(cmdl.wellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + "_testcm.pdf"
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_testcm.pdf"


    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle( ' Classification Algorithm Comparison ' )
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig(pdfcl)
    if not cmdlhideplot:
        plt.show()




def process_CatBoostClassifier(cmdlwellattribcsv,cmdlseisattribcsv,
                cmdlwcolsrange=None,
                cmdlwpredictorcols=None,
                cmdlwtargetcol=None,
                cmdlsaxyzcols=None,
                cmdlscolsrange=None,
                cmdlspredictorcols=None,
                cmdlwellsxyzcols=None,
                cmdlcoded=None,
                cmdlminmaxscale=None,
                cmdloutdir=None,
                cmdliterations=None,
                cmdllearningrate=None,
                cmdldepth=None,cmdlqcut=None,
                cmdlcv=None,
                cmdlfeatureimportance=False,
                cmdlgeneratesamples=None,
                cmdlbalancetype=None,
                cmdlnneighbors=None,
                cmdlvalsize=0.3,cmdlhideplot=False):
    """***************CatBoostClassifier."""
    cvalmin = 20
    # skip cross validation if data is less than cvalmin
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    if cmdlcoded:
        y = swa[swa.columns[cmdlwtargetcol]]
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qcodes'] = swa[swa.columns[cmdlwtargetcol]]
        qcount = len(Counter(y).items())
        probacolnames = ['Class%d' % i for i in range(qcount)]
        nclasses = qcount
        print('In coded:',probacolnames,nclasses)

    else:
        # use cmdlqcut
        probacolnames = ['Class%d' % i for i in range(cmdlqcut)]
        nclasses = cmdlqcut
        print('In qcut:',probacolnames,nclasses)
        swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
        swa['qcodes'] = swa['qa'].cat.codes
        y = swa['qcodes'].values
        print(qbins)

        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qa'] = swa['qa']
        swxyz['qcodes'] = swa['qcodes']

    if cmdlgeneratesamples:
        # yoriginal = y.copy()
        # Xoriginal = X.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='c',func='svc')

    resampled = False
    if cmdlbalancetype == 'ros':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'smote':
        X_resampled, y_resampled = SMOTE(k_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'adasyn':
        X_resampled, y_resampled = ADASYN(n_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic Predictors From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        colnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic Predictor cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        colnames = swa[swa.columns[cmdlspredictorcols]].columns

    if cmdlfeatureimportance:
        clf = CatBoostClassifier(iterations=cmdliterations, learning_rate=cmdllearningrate,
                    depth=cmdldepth,loss_function='MultiClass',calc_feature_importance=True,
                    random_seed=42)
        clf.fit(X, y)
        fr = pd.DataFrame(sorted(zip(clf.get_feature_importance(X,y), colnames)),columns=['Importance','Attribute'])
        print('Feature Ranking with CatBoostClassifier: ')
        print(fr)

    elif cmdlcv:
        clf = CatBoostClassifier(iterations=cmdliterations, learning_rate=cmdllearningrate,
                    depth=cmdldepth,loss_function='MultiClass',random_seed=42)
        cvscore = cross_val_score(clf,X,y,cv=cmdlcv)
        print("Accuracy: %.3f%% (%.3f%%)" % (cvscore.mean() * 100.0, cvscore.std() * 100.0))
        # print("Mean Score: {:10.4f}".format(np.mean(cvscore)))
        # print("Mean Score: %10.4f" %(np.mean(cvscore)))
        print('No files will be generated. Re-run without cross validation')

    else:
        clf = CatBoostClassifier(iterations=cmdliterations, learning_rate=cmdllearningrate,
                    depth=cmdldepth,loss_function='MultiClass',random_seed=42)
        if resampled:
            clf.fit(X_resampled, y_resampled)
        else:
            clf.fit(X, y)
        # Get predictions
        y_clf = clf.predict(Xpred, prediction_type='Class')
        # ypred = clf.predict(Xpred, prediction_type='RawFormulaVal')
        allprob = clf.predict_proba(Xpred)
        if y.size >= cvalmin:
            Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
                random_state=42)
            clf.fit(Xtrain, ytrain)

            yvalpred = clf.predict(Xval)
            yvalproba = clf.predict_proba(Xval)
            print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
            print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
            print('Validation Data Log Loss: %10.4f' % log_loss(yval,yvalproba))
            ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
            print(pd.crosstab(ydf['A'],ydf['P'],rownames=['Actuall'], colnames=['Predicted']))
            # print(confusion_matrix(yval.ravel(),yvalpred.ravel()))
            print(classification_report(yval.ravel(),yvalpred.ravel()))
            cnfmat = confusion_matrix(yval.ravel(),yvalpred.ravel())
            if cmdloutdir:
                pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_cbccvcnfmtx.pdf"
            else:
                pdfcnfmtx = os.path.join(dirsplit,fname) + "_cbccvcnfmtx.pdf"
            with PdfPages(pdfcnfmtx) as pdf:
                plot_confusion_matrix(cnfmat,probacolnames,title='CatBoost Train Confusion Matrix',
                    hideplot=cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

                plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='CatBoost Train Confusion Matrix Normalized',
                    hideplot=cmdlhideplot)

                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        else:
            print('********No cross validation will be done due to low # of data points************')
        """
        if cmdlcoded:
            nclasses = qcount
        else:
            nclasses = cmdlqcut
        """
        # [(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
        # add class column before probabilities
        ssa['CatBoost'] = y_clf
        ssxyz['CatBoost'] = y_clf
        for i in range(cmdlqcut):
            ssa[probacolnames[i]] = allprob[:,i]
            ssxyz[probacolnames[i]] = allprob[:,i]

        # for i in range(cmdlqcut):
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_cbcroc.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_cbcroc.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range(nclasses):
                yw_prob = clf.predict_proba(X)[:,i]
                plot_roc_curve(y,yw_prob,i,cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d' % len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' % log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames=['Actuall'], colnames=['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))
        cnfmat = confusion_matrix(y.ravel(),yw.ravel())
        if cmdloutdir:
            pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_cbccnfmtx.pdf"
        else:
            pdfcnfmtx = os.path.join(dirsplit,fname) + "_cbccnfmtx.pdf"
        with PdfPages(pdfcnfmtx) as pdf:
            plot_confusion_matrix(cnfmat,probacolnames,title='CatBoost Confusion Matrix',
                hideplot=cmdlhideplot)
            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

            plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='CatBoost Confusion Matrix Normalized',
                hideplot=cmdlhideplot)

            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

        if not cmdlgeneratesamples:
            swxyz['predqcodes'] = yw
            swa['predqcodes'] = yw
            # pdfbar = os.path.join(dirsplit,fname) +"_gnbbar.pdf"

            if cmdlcoded:
                swxyz1 = swxyz[~swxyz['Well'].str.contains('PW')]
            else:
                swxyz1 = swxyz.copy()

            swxyz1.set_index('Well',inplace=True)
            # plot 20 wells per bar graph
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) + "_cbcbar.pdf"
            else:
                pdfcl = os.path.join(dirsplit,fname) + "_cbcbar.pdf"
            with PdfPages(pdfcl) as pdf:
                for i in range(0,swxyz1.shape[0],20):
                    swtemp = swxyz1.iloc[i:i + 20,:]
                    swtemp[['qcodes','predqcodes']].plot(kind='bar',rot=45)
                    pdf.savefig()
                    if not cmdlhideplot:
                        plt.show()
                    plt.close()

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                wellf=cmdlwellattribcsv,
                wdf=swa, wxydf=swxyz,
                outdir=cmdloutdir,
                ssuffix='_scbc',
                wsuffix='_wcbc',name2merge=cmdlwellattribcsv)

        else:
            print('******Will not generate well bar plots because of generate samples option')

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                outdir=cmdloutdir,
                ssuffix='_scbc',
                name2merge=cmdlwellattribcsv)

def process_TuneCatBoostClassifier(cmdlwellattribcsv,cmdlseisattribcsv,
                cmdlwcolsrange=None,cmdlwpredictorcols=None,
                cmdlwtargetcol=None,cmdlsaxyzcols=None,
                cmdlscolsrange=None,cmdlspredictorcols=None,
                cmdlwellsxyzcols=None,cmdlcoded=None,
                cmdlminmaxscale=None,cmdloutdir=None,cmdliterations=None,
                cmdllearningrate=None,cmdldepth=None,cmdlqcut=None,cmdlcv=None,
                cmdlhideplot=False):
    """Tuning hyperparameters for CatBoost Classifier."""
    swa = pd.read_csv(cmdlwellattribcsv)
    # print(swa.head(5))
    print('Well Target: %d ' % cmdlwtargetcol)
    if cmdlwcolsrange:
        print('Well Predictors From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well Predictor cols',cmdlwpredictorcols)
        X = swa[swa.columns[cmdlwpredictorcols]].values
        # colnames = swa[swa.columns[cmdlwpredictorcols]].columns
    y = swa[swa.columns[cmdlwtargetcol]]

    probacolnames = ['Class%d' % i for i in range(cmdlqcut)]

    if not cmdlcoded:
        swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
        swa['qcodes'] = swa['qa'].cat.codes
        y = swa['qcodes'].values
        print(qbins)

        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qa'] = swa['qa']
        swxyz['qcodes'] = swa['qcodes']

    else:
        y = swa[swa.columns[cmdlwtargetcol]]
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qcodes'] = swa[swa.columns[cmdlwtargetcol]]
    qcount = Counter(y)
    print(qcount)

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic Predictors From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1] + 1))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        # colnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic Predictor cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # colnames = swa[swa.columns[cmdlspredictorcols]].columns

    params = {'iterations': cmdliterations,
        'learning_rate': cmdllearningrate,
        'depth': cmdldepth}
    grdcv = GridSearchCV(CatBoostClassifier(loss_function='MultiClass'),params,cv=cmdlcv)

    # Fit model
    grdcv.fit(X, y)
    print(grdcv.best_params_)
    clf = grdcv.best_estimator_
    # Get predictions
    wpred = clf.predict(X)
    y_clf = clf.predict(Xpred, prediction_type='Class')
    # ypred = clf.predict(Xpred, prediction_type='RawFormulaVal')
    allprob = clf.predict_proba(Xpred)
    wproba = clf.predict_proba(X)
    print('All Data Accuracy Score: %10.4f' % accuracy_score(y,wpred))
    print('Log Loss: %10.4f' % log_loss(y,wproba))

    # [(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
    # add class column before probabilities
    ssa['TunedCatBoost'] = y_clf
    ssxyz['TunedCatBoost'] = y_clf
    for i in range(cmdlqcut):
        ssa[probacolnames[i]] = allprob[:,i]
        ssxyz[probacolnames[i]] = allprob[:,i]

    yw = clf.predict(X)

    ywproba = clf.predict_proba(X)
    print('Full Data size: %5d' % len(yw))
    print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
    print('Full Data Log Loss: %10.4f' % log_loss(y,ywproba))
    ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
    print(pd.crosstab(ywdf['A'],ywdf['P'],rownames=['Actuall'], colnames=['Predicted']))
    print(classification_report(y.ravel(),yw.ravel()))

    swxyz['predqcodes'] = yw
    swa['predqcodes'] = yw
    # pdfbar = os.path.join(dirsplit,fname) +"_gnbbar.pdf"

    if cmdlcoded:
        swxyz1 = swxyz[~swxyz['Well'].str.contains('PW')]
    else:
        swxyz1 = swxyz.copy()

    swxyz1.set_index('Well',inplace=True)

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    # plot 20 wells per bar graph
    for i in range(0,swxyz1.shape[0],20):
        swtemp = swxyz1.iloc[i:i + 20,:]
        ax = swtemp[['qcodes','predqcodes']].plot(kind='bar',rot=45,figsize=(15,10))
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_tcbccode%1d.pdf" % i
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_tcbccode%1d.pdf" % i
        if not cmdlhideplot:
            plt.show()
        fig = ax.get_figure()
        fig.savefig(pdfcl)

    savefiles(seisf=cmdlseisattribcsv,
        sdf=ssa, sxydf=ssxyz,
        wellf=cmdlwellattribcsv,
        wdf=swa, wxydf=swxyz,
        outdir=cmdloutdir,
        ssuffix='_stcbc',
        wsuffix='_wtcbc',name2merge=cmdlwellattribcsv)

def process_logisticreg(cmdlwellattribcsv,cmdlseisattribcsv,
        cmdlwcolsrange=None,cmdlwanalysiscols=None,
        cmdlwtargetcol=None,cmdlwellsxyzcols=None,
        cmdlsaxyzcols=None,cmdlscolsrange=None,
        cmdlspredictorcols=None,cmdlqcut=None,
        cmdlcoded=None,cmdlclassweight=False,
        cmdloutdir=None,cmdlcv=None,cmdlvalsize=0.3,
        cmdlgeneratesamples=None,
        cmdlbalancetype=None,
        cmdlnneighbors=None,
        cmdlhideplot=False):
    """Logistic Regression -> Classification."""
    cvalmin = 20
    # skip cross validation if data is less than cvalmin
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlwcolsrange:
        print('Well From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well analysis cols',cmdlwanalysiscols)
        X = swa[swa.columns[cmdlwanalysiscols]].values
        # colnames = swa[swa.columns[cmdlwanalysiscols]].columns

    if cmdlcoded:
        y = swa[swa.columns[cmdlwtargetcol]]
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qcodes'] = swa[swa.columns[cmdlwtargetcol]]
        qcount = len(Counter(y).items())
        print('qcount:',qcount)
        probacolnames = ['Class%d' % i for i in range(qcount)]
        nclasses = qcount
        print('In coded:',probacolnames,nclasses)
        # print(qbins)
    else:
        # use cmdlqcut
        probacolnames = ['Class%d' % i for i in range(cmdlqcut)]
        nclasses = cmdlqcut
        print('In qcut:',probacolnames,nclasses)
        swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
        swa['qcodes'] = swa['qa'].cat.codes
        y = swa['qcodes'].values
        print(qbins)

        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qa'] = swa['qa']
        swxyz['qcodes'] = swa['qcodes']

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='c',func='svc')

    resampled = False
    if cmdlbalancetype == 'ros':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'smote':
        X_resampled, y_resampled = SMOTE(k_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'adasyn':
        X_resampled, y_resampled = ADASYN(n_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        predcolnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic analysis cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # predcolnames = ssa[ssa.columns[cmdlspredictorcols]].columns

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    if cmdlcv:
        seed = 42
        kfold = KFold(n_splits=cmdlcv, random_state=seed)
        if cmdlclassweight:
            clf = LogisticRegression(class_weight='balanced')
            print('Class weight balanced')
        else:
            clf = LogisticRegression()
        results = cross_val_score(clf, X, y, cv=kfold)
        print("Logistic Regression Accuracy: %.3f%% (%.3f%%)"  % (results.mean() * 100.0, results.std() * 100.0))
        print('No files will be generated. Re-run without cv option')
    else:
        if cmdlclassweight:
            clf = LogisticRegression(class_weight='balanced')
            print('Class weight balanced')
        else:
            clf = LogisticRegression()
        if resampled:
            clf.fit(X_resampled, y_resampled)
        else:
            clf.fit(X, y)
        y_clf = clf.predict(Xpred)
        allprob = clf.predict_proba(Xpred)

        if y.size >= cvalmin:
            Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
                random_state=42)
            clf.fit(Xtrain, ytrain)

            yvalpred = clf.predict(Xval)
            yvalproba = clf.predict_proba(Xval)
            print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
            print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
            print('Validation Data Log Loss: %10.4f' % log_loss(yval,yvalproba))
            ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
            print(pd.crosstab(ydf['A'],ydf['P'],rownames=['Actuall'], colnames=['Predicted']))
            print(classification_report(yval.ravel(),yvalpred.ravel()))

            cnfmat = confusion_matrix(yval.ravel(),yvalpred.ravel())
            if cmdloutdir:
                pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_lgrcvcnfmtx.pdf"
            else:
                pdfcnfmtx = os.path.join(dirsplit,fname) + "_lgrcvcnfmtx.pdf"
            with PdfPages(pdfcnfmtx) as pdf:
                plot_confusion_matrix(cnfmat,probacolnames,title='Logistic Reg Train Confusion Matrix',
                    hideplot=cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

                plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='Logistic Reg Train Confusion Matrix Normalized',
                    hideplot=cmdlhideplot)

                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()
        else:
            print('********No cross validation will be done due to low # of data points************')

        ssa['LRClass'] = y_clf
        ssxyz['LRClass'] = y_clf
        for i in range(nclasses):
            ssa[probacolnames[i]] = allprob[:,i]
            ssxyz[probacolnames[i]] = allprob[:,i]

        # for i in range(cmdlqcut):
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_lgrroc.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_lgrroc.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range(nclasses):
                yw_prob = clf.predict_proba(X)[:,i]
                plot_roc_curve(y,yw_prob,i,cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d' % len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' % log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames=['Actuall'], colnames=['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))
        cnfmat = confusion_matrix(y.ravel(),yw.ravel())
        if cmdloutdir:
            pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_lgrcnfmtx.pdf"
        else:
            pdfcnfmtx = os.path.join(dirsplit,fname) + "_lgrcnfmtx.pdf"
        with PdfPages(pdfcnfmtx) as pdf:
            plot_confusion_matrix(cnfmat,probacolnames,title='Logistic Regression Confusion Matrix',
                hideplot=cmdlhideplot)
            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

            plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='Logistic Regression Confusion Matrix Normalized',
                hideplot=cmdlhideplot)

            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

        if not cmdlgeneratesamples:
            swxyz['predqcodes'] = yw
            swa['predqcodes'] = yw
            # pdfbar = os.path.join(dirsplit,fname) +"_lgrbar.pdf"

            if cmdlcoded:
                swxyz1 = swxyz[~swxyz['Well'].str.contains('PW')]
            else:
                swxyz1 = swxyz.copy()

            swxyz1.set_index('Well',inplace=True)
            # plot 20 wells per bar graph
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) + "_lgrbar.pdf"
            else:
                pdfcl = os.path.join(dirsplit,fname) + "_lgrbar.pdf"
            with PdfPages(pdfcl) as pdf:
                for i in range(0,swxyz1.shape[0],20):
                    swtemp = swxyz1.iloc[i:i + 20,:]
                    # ax =swtemp[['qcodes','predqcodes']].plot(kind='bar',rot=45,figsize=(15,10))
                    swtemp[['qcodes','predqcodes']].plot(kind='bar',rot=45)
                    pdf.savefig()
                    if not cmdlhideplot:
                        plt.show()
                    plt.close()

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                wellf=cmdlwellattribcsv,
                wdf=swa, wxydf=swxyz,
                outdir=cmdloutdir,
                ssuffix='_slgrg',
                wsuffix='_wlgrg',name2merge=cmdlwellattribcsv)
        else:
            print('******Will not generate well bar plots because of generate samples option')

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                outdir=cmdloutdir,
                ssuffix='_slgrg',
                name2merge=cmdlwellattribcsv)

def process_GaussianNaiveBayes(cmdlwellattribcsv,cmdlseisattribcsv,
        cmdlwcolsrange=None,
        cmdlwanalysiscols=None,
        cmdlqcut=None,
        cmdlwellsxyzcols=None,
        cmdlsaxyzcols=None,
        cmdlscolsrange=None,
        cmdlspredictorcols=None,
        cmdlwtargetcol=None,
        cmdlcoded=None,
        cmdloutdir=None,
        cmdlcv=None,
        cmdlgeneratesamples=None,
        cmdlbalancetype=None,
        cmdlnneighbors=None,
        cmdlvalsize=0.3,
        cmdlhideplot=False):
    """Gaussian Naive Bayes Classification."""
    cvalmin = 20
    # skip cross validation if data is less than cvalmin
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlwcolsrange:
        print('Well From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well analysis cols',cmdlwanalysiscols)
        X = swa[swa.columns[cmdlwanalysiscols]].values
        # colnames = swa[swa.columns[cmdlwanalysiscols]].columns

    if cmdlcoded:
        y = swa[swa.columns[cmdlwtargetcol]]
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qcodes'] = swa[swa.columns[cmdlwtargetcol]]
        qcount = len(Counter(y).items())
        print('qcount:',qcount)
        probacolnames = ['Class%d' % i for i in range(qcount)]
        nclasses = qcount
        print('In coded:',probacolnames,nclasses)

    else:
        probacolnames = ['Class%d' % i for i in range(cmdlqcut)]
        nclasses = cmdlqcut
        print('In qcut:',probacolnames,nclasses)
        swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
        swa['qcodes'] = swa['qa'].cat.codes
        y = swa['qcodes'].values
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qa'] = swa['qa']
        swxyz['qcodes'] = swa['qcodes']
        print(qbins)

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='c',func='svc')

    resampled = False
    if cmdlbalancetype == 'ros':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'smote':
        X_resampled, y_resampled = SMOTE(k_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'adasyn':
        X_resampled, y_resampled = ADASYN(n_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        # predcolnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic analysis cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # predcolnames = ssa[ssa.columns[cmdlspredictorcols]].columns

    if cmdlcv:
        seed = 42
        kfold = KFold(n_splits=cmdlcv, random_state=seed)
        clf = GaussianNB()
        results = cross_val_score(clf, X, y, cv=kfold)
        print("Gaussian Naive Bayes Accuracy: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))
        print('No files will be generated. Re-run without cv option')
    else:
        clf = GaussianNB()
        if resampled:
            clf.fit(X_resampled, y_resampled)
        else:
            clf.fit(X, y)
        y_clf = clf.predict(Xpred)
        allprob = clf.predict_proba(Xpred)

        if y.size >= cvalmin:
            Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
                random_state=42)
            clf.fit(Xtrain, ytrain)

            yvalpred = clf.predict(Xval)
            yvalproba = clf.predict_proba(Xval)
            print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
            print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
            print('Validation Data Log Loss: %10.4f' % log_loss(yval,yvalproba))
            ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
            print(pd.crosstab(ydf['A'],ydf['P'],rownames=['Actuall'], colnames=['Predicted']))
            # print(confusion_matrix(yval.ravel(),yvalpred.ravel()))
            print(classification_report(yval.ravel(),yvalpred.ravel()))
            cnfmat =confusion_matrix(yval.ravel(),yvalpred.ravel())
            if cmdloutdir:
                pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_gnbcvcnfmtx.pdf"
            else:
                pdfcnfmtx = os.path.join(dirsplit,fname) + "_gnbcvcnfmtx.pdf"
            with PdfPages(pdfcnfmtx) as pdf:
                plot_confusion_matrix(cnfmat,probacolnames,title='Gaussian NB Train Confusion Matrix',
                    hideplot=cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

                plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='Gaussian NB Train Confusion Matrix Normalized',
                    hideplot=cmdlhideplot)

                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()
        else:
            print('********No cross validation will be done due to low # of data points************')

        # add class column before probabilities
        ssa['GNBClass'] = y_clf
        ssxyz['GNBClass'] = y_clf

        # [(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
        for i in range(cmdlqcut):
            ssa[probacolnames[i]] = allprob[:,i]
            ssxyz[probacolnames[i]] = allprob[:,i]

        # for i in range(cmdlqcut):
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_gnbroc.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_gnbroc.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range(nclasses):
                yw_prob = clf.predict_proba(X)[:,i]
                plot_roc_curve(y,yw_prob,i,cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d' % len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' % log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames=['Actuall'], colnames=['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))
        cnfmat = confusion_matrix(y.ravel(),yw.ravel())
        if cmdloutdir:
            pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_gnbcnfmtx.pdf"
        else:
            pdfcnfmtx = os.path.join(dirsplit,fname) + "_gnbcnfmtx.pdf"
        with PdfPages(pdfcnfmtx) as pdf:
            plot_confusion_matrix(cnfmat,probacolnames,title='Gaussian NB Confusion Matrix',
                hideplot=cmdlhideplot)
            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

            plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='Gaussian NB Confusion Matrix Normalized',
                hideplot=cmdlhideplot)

            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

        if not cmdlgeneratesamples:
            swxyz['predqcodes'] = yw
            swa['predqcodes'] = yw

            if cmdlcoded:
                swxyz1 = swxyz[~swxyz['Well'].str.contains('PW')]
            else:
                swxyz1 = swxyz.copy()

            swxyz1.set_index('Well',inplace=True)
            # plot 20 wells per bar graph
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) + "_gnbbar.pdf"
            else:
                pdfcl = os.path.join(dirsplit,fname) + "_gnbbar.pdf"
            with PdfPages(pdfcl) as pdf:
                for i in range(0,swxyz1.shape[0],20):
                    swtemp = swxyz1.iloc[i:i + 20,:]
                    swtemp[['qcodes','predqcodes']].plot(kind='bar',rot=45)
                    pdf.savefig()
                    if not cmdlhideplot:
                        plt.show()
                    plt.close()

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                wellf=cmdlwellattribcsv,
                wdf=swa, wxydf=swxyz,
                outdir=cmdloutdir,
                ssuffix='_sgnb',
                wsuffix='_wgnb',name2merge=cmdlwellattribcsv)

        else:
            print('******Will not generate well bar plots because of generate samples option')

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                outdir=cmdloutdir,
                ssuffix='_sgnb',
                name2merge=cmdlwellattribcsv)

def process_QuadraticDiscriminantAnalysis(cmdlwellattribcsv,cmdlseisattribcsv,
        cmdlwcolsrange=None,
        cmdlwanalysiscols=None,
        cmdlqcut=None,
        cmdlwellsxyzcols=None,
        cmdlsaxyzcols=None,
        cmdlscolsrange=None,
        cmdlspredictorcols=None,
        cmdlwtargetcol=None,
        cmdlcoded=None,
        cmdloutdir=None,
        cmdlcv=None,
        cmdlgeneratesamples=None,
        cmdlbalancetype=None,
        cmdlnneighbors=None,
        cmdlvalsize=0.3,
        cmdlhideplot=False):
    """Quadratic Discriminant Anlalysis Classification."""
    cvalmin = 20
    # skip cross validation if data is less than cvalmin
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlwcolsrange:
        print('Well From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well analysis cols',cmdlwanalysiscols)
        X = swa[swa.columns[cmdlwanalysiscols]].values
        # colnames = swa[swa.columns[cmdlwanalysiscols]].columns

    if cmdlcoded:
        y = swa[swa.columns[cmdlwtargetcol]]
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qcodes'] = swa[swa.columns[cmdlwtargetcol]]
        qcount = len(Counter(y).items())
        print('qcount:',qcount)
        probacolnames = ['Class%d' % i for i in range(qcount)]
        nclasses = qcount
        print('In coded:',probacolnames,nclasses)

    else:
        probacolnames = ['Class%d' % i for i in range(cmdlqcut)]
        nclasses = cmdlqcut
        print('In qcut:',probacolnames,nclasses)
        swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
        swa['qcodes'] = swa['qa'].cat.codes
        y = swa['qcodes'].values
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qa'] = swa['qa']
        swxyz['qcodes'] = swa['qcodes']
        print(qbins)

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='c',func='svc')

    resampled = False
    if cmdlbalancetype == 'ros':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'smote':
        X_resampled, y_resampled = SMOTE(k_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'adasyn':
        X_resampled, y_resampled = ADASYN(n_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        # Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        # predcolnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic analysis cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # predcolnames = ssa[ssa.columns[cmdlspredictorcols]].columns

    if cmdlcv:
        seed = 42
        kfold = KFold(n_splits=cmdlcv, random_state=seed)
        clf = QuadraticDiscriminantAnalysis()
        results = cross_val_score(clf, X, y, cv=kfold)
        print("Quadratic Discriminant Analysis Accuracy: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))
        print('No files will be generated. Re-run without cv option')
    else:
        clf = QuadraticDiscriminantAnalysis()
        if resampled:
            clf.fit(X_resampled, y_resampled)
        else:
            clf.fit(X, y)
        y_clf = clf.predict(Xpred)
        allprob = clf.predict_proba(Xpred)

        if y.size >= cvalmin:
            Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
                random_state=42)
            clf.fit(Xtrain, ytrain)

            yvalpred = clf.predict(Xval)
            yvalproba = clf.predict_proba(Xval)
            print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
            print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
            print('Validation Data Log Loss: %10.4f' % log_loss(yval,yvalproba))
            ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
            print(pd.crosstab(ydf['A'],ydf['P'],rownames=['Actuall'], colnames=['Predicted']))
            # print(confusion_matrix(yval.ravel(),yvalpred.ravel()))
            print(classification_report(yval.ravel(),yvalpred.ravel()))

            cnfmat = confusion_matrix(yval.ravel(),yvalpred.ravel())
            if cmdloutdir:
                pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_qdacvcnfmtx.pdf"
            else:
                pdfcnfmtx = os.path.join(dirsplit,fname) + "_qdacvcnfmtx.pdf"
            with PdfPages(pdfcnfmtx) as pdf:
                plot_confusion_matrix(cnfmat,probacolnames,title='QDA Train Confusion Matrix',
                    hideplot=cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

                plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='QDA Train Confusion Matrix Normalized',
                    hideplot=cmdlhideplot)

                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        else:
            print('********No cross validation will be done due to low # of data points************')

        # add class column before probabilities
        ssa['QDAClass'] = y_clf
        ssxyz['QDAClass'] = y_clf

        # [(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
        for i in range(cmdlqcut):
            ssa[probacolnames[i]] = allprob[:,i]
            ssxyz[probacolnames[i]] = allprob[:,i]

        # for i in range(cmdlqcut):
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_qdaroc.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_qdaroc.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range(nclasses):
                yw_prob = clf.predict_proba(X)[:,i]
                plot_roc_curve(y,yw_prob,i,cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d' % len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' % log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames=['Actuall'], colnames=['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))

        cnfmat = confusion_matrix(y.ravel(),yw.ravel())
        if cmdloutdir:
            pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_qdacnfmtx.pdf"
        else:
            pdfcnfmtx = os.path.join(dirsplit,fname) + "_qdacnfmtx.pdf"
        with PdfPages(pdfcnfmtx) as pdf:
            plot_confusion_matrix(cnfmat,probacolnames,title='QDA Confusion Matrix',
                hideplot=cmdlhideplot)
            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

            plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='QDA Confusion Matrix Normalized',
                hideplot=cmdlhideplot)

            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

        if not cmdlgeneratesamples:
            swxyz['predqcodes'] = yw
            swa['predqcodes'] = yw
            # pdfbar = os.path.join(dirsplit,fname) +"_gnbbar.pdf"

            if cmdlcoded:
                swxyz1 = swxyz[~swxyz['Well'].str.contains('PW')]
            else:
                swxyz1 = swxyz.copy()

            swxyz1.set_index('Well',inplace=True)
            # plot 20 wells per bar graph
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) + "_qdabar.pdf"
            else:
                pdfcl = os.path.join(dirsplit,fname) + "_qdabar.pdf"
            with PdfPages(pdfcl) as pdf:
                for i in range(0,swxyz1.shape[0],20):
                    swtemp = swxyz1.iloc[i:i + 20,:]
                    swtemp[['qcodes','predqcodes']].plot(kind='bar',rot=45)
                    # ax.set_ylim(-1.0,2.0)
                    pdf.savefig()
                    if not cmdlhideplot:
                        plt.show()
                    plt.close()

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                wellf=cmdlwellattribcsv,
                wdf=swa, wxydf=swxyz,
                outdir=cmdloutdir,
                ssuffix='_sqda',
                wsuffix='_wqda',name2merge=cmdlwellattribcsv)

        else:
            print('******Will not generate well bar plots because of generate samples option')

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                outdir=cmdloutdir,
                ssuffix='_sqda',
                name2merge=cmdlwellattribcsv)

def process_NuSVC(cmdlwellattribcsv,cmdlseisattribcsv,
        cmdlwcolsrange=None,
        cmdlwanalysiscols=None,
        cmdlqcut=None,
        cmdlwellsxyzcols=None,
        cmdlsaxyzcols=None,
        cmdlscolsrange=None,
        cmdlspredictorcols=None,
        cmdlwtargetcol=None,
        cmdlcoded=None,
        cmdloutdir=None,cmdlcv=None,
        cmdlvalsize=0.3,
        cmdlnu=None,
        cmdlbalancetype=None,
        cmdlnneighbors=None,
        cmdlgeneratesamples=None,
        cmdlhideplot=False):
    """Support Vector Machine Classification."""
    cvalmin = 20
    # skip cross validation if data is less than cvalmin
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlwcolsrange:
        print('Well From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        print('Well analysis cols',cmdlwanalysiscols)
        X = swa[swa.columns[cmdlwanalysiscols]].values
        # colnames = swa[swa.columns[cmdlwanalysiscols]].columns

    if cmdlcoded:
        y = swa[swa.columns[cmdlwtargetcol]]
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qcodes'] = swa[swa.columns[cmdlwtargetcol]]
        qcount = len(Counter(y).items())
        print('qcount:',qcount)
        probacolnames = ['Class%d' % i for i in range(qcount)]
        nclasses = qcount
        # print('In coded:',probacolnames,nclasses)
    else:
        probacolnames = ['Class%d' % i for i in range(cmdlqcut)]
        nclasses = cmdlqcut
        # print('In qcut:',probacolnames,nclasses)
        swa['qa'],qbins = pd.qcut(swa[swa.columns[cmdlwtargetcol]],cmdlqcut,labels=probacolnames,retbins=True)
        swa['qcodes'] = swa['qa'].cat.codes
        y = swa['qcodes'].values
        swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()
        swxyz['qa'] = swa['qa']
        swxyz['qcodes'] = swa['qcodes']
        print(qbins)

    if cmdlgeneratesamples:
        # Xoriginal = X.copy()
        # yoriginal = y.copy()
        X,y = gensamples(X,y,nsamples=cmdlgeneratesamples,kind='c',func='svc')

    resampled = False
    if cmdlbalancetype == 'ros':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'smote':
        X_resampled, y_resampled = SMOTE(k_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True
    elif cmdlbalancetype == 'adasyn':
        X_resampled, y_resampled = ADASYN(n_neighbors=cmdlnneighbors).fit_sample(X, y)
        print(sorted(Counter(y_resampled).items()))
        resampled = True

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
        # predcolnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
    else:
        print('Seismic analysis cols',cmdlspredictorcols)
        Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
        # predcolnames = ssa[ssa.columns[cmdlspredictorcols]].columns

    if cmdlcv:
        seed = 42
        kfold = KFold(n_splits=cmdlcv, random_state=seed)
        clf = NuSVC(nu=cmdlnu,probability=True)
        results = cross_val_score(clf, X, y, cv=kfold)
        print("NuSVC Accuracy: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))
        print('No files will be generated. Re-run without cv option')
    else:
        clf = NuSVC(nu=cmdlnu,probability=True)
        if resampled:
            clf.fit(X_resampled, y_resampled)
        else:
            clf.fit(X, y)
        y_clf = clf.predict(Xpred)
        allprob = clf.predict_proba(Xpred)

        if y.size >= cvalmin:
            Xtrain,Xval,ytrain,yval = train_test_split(X,y,test_size=cmdlvalsize,
                random_state=42)
            clf.fit(Xtrain, ytrain)

            yvalpred = clf.predict(Xval)
            yvalproba = clf.predict_proba(Xval)
            print('Train Data size: %5d, Validation Data size: %5d' % (len(ytrain),len(yval)))
            print('Validation Data Accuracy Score: %10.4f' % accuracy_score(yval,yvalpred))
            print('Validation Data Log Loss: %10.4f' % log_loss(yval,yvalproba))
            ydf = pd.DataFrame({'A':yval.ravel(),'P':yvalpred.ravel()})
            print(pd.crosstab(ydf['A'],ydf['P'],rownames=['Actuall'], colnames=['Predicted']))
            # print(confusion_matrix(yval.ravel(),yvalpred.ravel()))
            print(classification_report(yval.ravel(),yvalpred.ravel()))

            cnfmat = confusion_matrix(yval.ravel(),yvalpred.ravel())
            if cmdloutdir:
                pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_nsvccvcnfmtx.pdf"
            else:
                pdfcnfmtx = os.path.join(dirsplit,fname) + "_nsvccvcnfmtx.pdf"
            with PdfPages(pdfcnfmtx) as pdf:
                plot_confusion_matrix(cnfmat,probacolnames,title='NuSVC Train Confusion Matrix',
                    hideplot=cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

                plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='NuSVC Train Confusion Matrix Normalized',
                    hideplot=cmdlhideplot)

                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        else:
            print('********No cross validation will be done due to low # of data points************')

        # add class column before probabilities
        ssa['NuSVCClass'] = y_clf
        ssxyz['NuSVCClass'] = y_clf

        # [(ssa[probacolnames[i]] = allprob[:,i]) for i in range(cmd.qcut)]
        # for i in range(cmdlqcut):

        # ************need to adjust this for all other classifications
        for i in range(nclasses):
            ssa[probacolnames[i]] = allprob[:,i]
            ssxyz[probacolnames[i]] = allprob[:,i]

        # for i in range(cmdlqcut):
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_nusvcroc.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_nusvcroc.pdf"
        with PdfPages(pdfcl) as pdf:
            for i in range(nclasses):
                yw_prob = clf.predict_proba(X)[:,i]
                plot_roc_curve(y,yw_prob,i,cmdlhideplot)
                pdf.savefig()
                if not cmdlhideplot:
                    plt.show()
                plt.close()

        yw = clf.predict(X)
        ywproba = clf.predict_proba(X)
        print('Full Data size: %5d' % len(yw))
        print('Full Data Accuracy Score: %10.4f' % accuracy_score(y,yw))
        print('Full Data Log Loss: %10.4f' % log_loss(y,ywproba))
        ywdf = pd.DataFrame({'A':y.ravel(),'P':yw.ravel()})
        print(pd.crosstab(ywdf['A'],ywdf['P'],rownames=['Actuall'], colnames=['Predicted']))
        print(classification_report(y.ravel(),yw.ravel()))

        cnfmat = confusion_matrix(y.ravel(),yw.ravel())
        if cmdloutdir:
            pdfcnfmtx = os.path.join(cmdloutdir,fname) + "_nsvccnfmtx.pdf"
        else:
            pdfcnfmtx = os.path.join(dirsplit,fname) + "_nsvccnfmtx.pdf"
        with PdfPages(pdfcnfmtx) as pdf:
            plot_confusion_matrix(cnfmat,probacolnames,title='NuSVC Confusion Matrix',
                hideplot=cmdlhideplot)
            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

            plot_confusion_matrix(cnfmat,probacolnames,normalize=True,title='NuSVC Confusion Matrix Normalized',
                hideplot=cmdlhideplot)

            pdf.savefig()
            if not cmdlhideplot:
                plt.show()
            plt.close()

        if not cmdlgeneratesamples:
            swxyz['predqcodes'] = yw
            swa['predqcodes'] = yw
            # pdfbar = os.path.join(dirsplit,fname) +"_gnbbar.pdf"

            if cmdlcoded:
                swxyz1 = swxyz[~swxyz['Well'].str.contains('PW')]
            else:
                swxyz1 = swxyz.copy()

            swxyz1.set_index('Well',inplace=True)
            # plot 20 wells per bar graph
            if cmdloutdir:
                pdfcl = os.path.join(cmdloutdir,fname) + "_nusvcbar.pdf"
            else:
                pdfcl = os.path.join(dirsplit,fname) + "_nusvcbar.pdf"
            with PdfPages(pdfcl) as pdf:
                for i in range(0,swxyz1.shape[0],20):
                    swtemp = swxyz1.iloc[i:i + 20,:]
                    swtemp[['qcodes','predqcodes']].plot(kind='bar',rot=45)
                    # ax.set_ylim(-1.0,2.0)
                    pdf.savefig()
                    if not cmdlhideplot:
                        plt.show()
                    plt.close()

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                wellf=cmdlwellattribcsv,
                wdf=swa, wxydf=swxyz,
                outdir=cmdloutdir,
                ssuffix='_snusvc',
                wsuffix='_wnusvc',name2merge=cmdlwellattribcsv)
        else:
            print('******Will not generate well bar plots because of generate samples option')

            savefiles(seisf=cmdlseisattribcsv,
                sdf=ssa, sxydf=ssxyz,
                outdir=cmdloutdir,
                ssuffix='_snusvc',
                name2merge=cmdlwellattribcsv)

def process_GaussianMixtureModel(cmdlwellattribcsv,cmdlseisattribcsv,
        cmdlbayesian=False,
        cmdlwcolsrange=None,
        cmdlwanalysiscols=None,
        cmdlwtargetcol=None,
        cmdlwellsxyzcols=None,
        cmdlsaxyzcols=None,
        cmdlscolsrange=None,
        cmdlcatcol=None,
        cmdlspredictorcols=None,
        cmdlncomponents=None,
        cmdloutdir=None,
        cmdlhideplot=False):
    """Gaussian Mixture Model Classification.

    This can be used as a clustering process by supplying only the welld data.
    The saved file will have the probabilities of the specified classes

    If seismic attributes are supplied the model generated from the well data
    will be used to predict the probabilities of the seismic.
    """
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlcatcol:
        collabels = pd.get_dummies(swa.iloc[:,cmdlcatcol],drop_first=True)
        swa.drop(swa.columns[cmdlcatcol],axis=1,inplace=True)
        swa = pd.concat([swa,collabels],axis=1)
        cmdlwcolsrange[1] += collabels.shape[1]
    if cmdlwcolsrange:
        print('Well From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        X = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
        # colnames = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].columns
    else:
        if cmdlcatcol:
            # need to find a way to add to list of columns the list of dummy cols
            pass
        print('Well analysis cols',cmdlwanalysiscols)
        X = swa[swa.columns[cmdlwanalysiscols]].values
        # colnames = swa[swa.columns[cmdlwanalysiscols]].columns
    # if cmdlwtargetcol:
        # inclasses = swa[swa.columns[cmdlwtargetcol]].unique().tolist()
        # probacolnames = ['Class%d' % i for i in inclasses]

    probaclassnames = ['GMMClass%d' % i for i in range(cmdlncomponents)]
    swxyz = swa[swa.columns[cmdlwellsxyzcols]].copy()

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    pltname = ''
    if cmdlbayesian:
        gmm = mixture.BayesianGaussianMixture(n_components=cmdlncomponents,
                covariance_type='spherical',
                max_iter=500,
                random_state=0).fit(X)
        pltname = 'bayes'
    else:
        gmm = mixture.GaussianMixture(n_components=cmdlncomponents,
                covariance_type='spherical',
                max_iter=500,
                random_state=0).fit(X)
    xpdf = np.linspace(-4, 3, 1000)
    _,ax = plt.subplots()
    for i in range(gmm.n_components):
        pdf = gmm.weights_[i] * sts.norm(gmm.means_[i, 0],np.sqrt(gmm.covariances_[i])).pdf(xpdf)
        ax.fill(xpdf, pdf, edgecolor='none', alpha=0.3,label='%s' % probaclassnames[i])
    ax.legend()
    if not cmdlhideplot:
        plt.show()
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + "_gmm%d%s.pdf" % (cmdlncomponents,pltname)
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_gmm%d%s.pdf" % (cmdlncomponents,pltname)
    fig = ax.get_figure()
    fig.savefig(pdfcl)

    # yw_gmm = gmm.predict(X)
    allwprob = gmm.predict_proba(X)
    for i in range(cmdlncomponents):
        swa[probaclassnames[i]] = allwprob[:,i]
        swxyz[probaclassnames[i]] = allwprob[:,i]

    if cmdlseisattribcsv:
        ssa = pd.read_csv(cmdlseisattribcsv)
        ssxyz = ssa[ssa.columns[cmdlsaxyzcols]].copy()

        if cmdlscolsrange:
            print('Seismic From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
            Xpred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].values
            # predcolnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]].columns
        else:
            print('Seismic analysis cols',cmdlspredictorcols)
            Xpred = ssa[ssa.columns[cmdlspredictorcols]].values
            # predcolnames = ssa[ssa.columns[cmdlspredictorcols]].columns

        ys_gmm = gmm.predict(Xpred)
        allsprob = gmm.predict_proba(Xpred)
        ssa['GMMclass'] = ys_gmm
        ssxyz['GMMclass'] = ys_gmm

        for i in range(cmdlncomponents):
            ssa[probaclassnames[i]] = np.around(allsprob[:,i],decimals=3)
            ssxyz[probaclassnames[i]] = np.around(allsprob[:,i],decimals=3)

        savefiles(seisf=cmdlseisattribcsv,
            sdf=ssa, sxydf=ssxyz,
            wellf=cmdlwellattribcsv,
            wdf=swa, wxydf=swxyz,
            outdir=cmdloutdir,
            ssuffix='_gmm%d' % cmdlncomponents,
            wsuffix='_gmm%d' % cmdlncomponents,name2merge=cmdlwellattribcsv)

    else:
        savefiles(seisf=cmdlwellattribcsv,
            sdf=swa, sxydf=swxyz,
            outdir=cmdloutdir,
            ssuffix='_gmm%d%s' % (cmdlncomponents,pltname))

def process_clustertest(cmdlallattribcsv,
        cmdlcolsrange=None,
        cmdlcols2cluster=None,
        cmdlsample=None,
        cmdloutdir=None,
        cmdlhideplot=False):
    """Test for optimum # of clusters for KMeans Clustering."""
    swa = pd.read_csv(cmdlallattribcsv)
    #print(swa.sample(5))

    if cmdlcolsrange:
        print('Well From col# %d to col %d' %(cmdlcolsrange[0],cmdlcolsrange[1]))
        swax = swa[swa.columns[cmdlcolsrange[0]: cmdlcolsrange[1] + 1]]
    else:
        swax = swa[swa.columns[cmdlcols2cluster]]
    swax = swax.sample(frac=cmdlsample).copy()


    dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    if cmdloutdir:
        pdfcl = os.path.join(cmdl.outdir,fname) + "_cla.pdf"
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_cla.pdf"
    inertia = list()
    delta_inertia = list()
    for k in range(1,21):
        clustering = KMeans(n_clusters=k, n_init=10,random_state= 1)
        clustering.fit(swax)
        if inertia:
            delta_inertia.append(inertia[-1] - clustering.inertia_)
        inertia.append(clustering.inertia_)
    with PdfPages(pdfcl) as pdf:
        plt.figure(figsize=(8,8))
        plt.plot([k for k in range(2,21)], delta_inertia,'ko-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Rate of Change of Intertia')
        plt.title('KMeans Cluster Analysis')
        pdf.savefig()
        if not cmdlhideplot:
            plt.show()
        plt.close()
        print('Successfully generated %s file'  % pdfcl)

def process_clustering(cmdlallattribcsv,cmdlcolsrange=None,
        cmdlcols2cluster=None,
        cmdlnclusters=None,
        cmdlplotsilhouette=False,
        cmdlsample=None,cmdlxyzcols=None,
        cmdladdclass=None,
        cmdloutdir=None,
        cmdlhideplot=False):
    """Cluster once csv."""
    swa = pd.read_csv(cmdlallattribcsv)
    if cmdlcolsrange:
        print('Well From col# %d to col %d' % (cmdlcolsrange[0],cmdlcolsrange[1]))
        swax = swa[swa.columns[cmdlcolsrange[0]: cmdlcolsrange[1] + 1]]
    else:
        swax = swa[swa.columns[cmdlcols2cluster]]
    clustering = KMeans(n_clusters=cmdlnclusters,
        n_init=5,
        max_iter=300,
        tol=1e-04,
        random_state=1)
    ylabels = clustering.fit_predict(swax)
    nlabels = np.unique(ylabels)
    print('nlabels',nlabels)

    if cmdladdclass == 'labels':
        swa['Class'] = ylabels
    elif cmdladdclass == 'dummies':
        classlabels = pd.Series(ylabels)
        classdummies = pd.get_dummies(classlabels,prefix='Class')
        swa = pd.concat([swa,classdummies],axis=1)
    print(swa.shape)

    swatxt = swa[swa.columns[cmdlxyzcols]].copy()
    if cmdladdclass == 'labels':
        swatxt['Class'] = ylabels
    elif cmdladdclass == 'dummies':
        classlabels = pd.Series(ylabels)
        classdummies = pd.get_dummies(classlabels,prefix='Cluster')
        swatxt = pd.concat([swatxt,classdummies],axis=1)

    if cmdladdclass == 'labels':
        savefiles(seisf=cmdlallattribcsv,
            sdf=swa, sxydf=swatxt,
            outdir=cmdloutdir,
            ssuffix='_cl')
    else:
        savefiles(seisf=cmdlallattribcsv,
            sdf=swa, sxydf=swatxt,
            outdir=cmdloutdir,
            ssuffix='_cld')

    '''
    Warning: Do not use sample to enable plot silhouette and add labels or dummies
    Better make a seperate run for silhouette plot on sampled data then use full data
    to add labels
    '''

    if cmdlplotsilhouette:
        dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
        fname,fextn = os.path.splitext(fextsplit)
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "_silcl.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "_silcl.pdf"
        # only resample data if plotting silhouette
        swax = swax.sample(frac=cmdlsample).copy()
        ylabels = clustering.fit_predict(swax)
        n_clusters = ylabels.shape[0]
        silhouette_vals = silhouette_samples(swax, ylabels, metric='euclidean')
        y_ax_lower, y_ax_upper = 0, 0
        yticks = []
        for i, c in enumerate(nlabels):
            c_silhouette_vals = silhouette_vals[ylabels == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            color = cm.jet(i / n_clusters)
            plt.barh(range(y_ax_lower, y_ax_upper),c_silhouette_vals,height=1.0,edgecolor='none',color=color)
            yticks.append((y_ax_lower + y_ax_upper) / 2)
            y_ax_lower += len(c_silhouette_vals)
            silhouette_avg = np.mean(silhouette_vals)
            plt.axvline(silhouette_avg,
                color="red",
                linestyle="--")
            plt.yticks(yticks, ylabels + 1)
            plt.ylabel('Cluster')
            plt.xlabel('Silhouette coefficient')
        plt.savefig(pdfcl)
        if not cmdlhideplot:
            plt.show()

def process_dbscan(cmdlallattribcsv,
    cmdlcolsrange=None,
                cmdlcols2cluster=None,
                cmdlsample=None,
                cmdlxyzcols=None,
                cmdlminsamples=None,
                cmdladdclass=None,
                cmdleps=None,
                cmdloutdir=None):
    """DBSCAN CLUSTERING."""
    swa = pd.read_csv(cmdlallattribcsv)
    if cmdlcolsrange:
        print('Well From col# %d to col %d' % (cmdlcolsrange[0],cmdlcolsrange[1]))
        swax = swa[swa.columns[cmdlcolsrange[0]: cmdlcolsrange[1] + 1]]
    else:
        swax = swa[swa.columns[cmdlcols2cluster]]

    dbscan = DBSCAN(eps=cmdleps, metric='euclidean', min_samples=cmdlminsamples)
    ylabels = dbscan.fit_predict(swax)
    print('Labels count per class:',list(Counter(ylabels).items()))

    # n_clusters = len(set(ylabels)) - (1 if -1 in ylabels else 0)
    n_clusters = len(set(ylabels))
    print('Estimated number of clusters: %d' % n_clusters)

    if cmdladdclass == 'labels':
        swa['Class'] = ylabels
    elif cmdladdclass == 'dummies':
        classlabels = pd.Series(ylabels)
        classdummies = pd.get_dummies(classlabels,prefix='Class')
        swa = pd.concat([swa,classdummies],axis=1)
    print(swa.shape)

    swatxt = swa[swa.columns[cmdlxyzcols]].copy()
    if cmdladdclass == 'labels':
        swatxt['Class'] = ylabels
    elif cmdladdclass == 'dummies':
        classlabels = pd.Series(ylabels)
        classdummies = pd.get_dummies(classlabels,prefix='Class')
        swatxt = pd.concat([swatxt,classdummies],axis=1)

    if cmdladdclass == 'labels':
        savefiles(seisf=cmdlallattribcsv,
            sdf=swa, sxydf=swatxt,
            outdir=cmdloutdir,
            ssuffix='_dbscn')
    else:
        savefiles(seisf=cmdlallattribcsv,
            sdf=swa, sxydf=swatxt,
            outdir=cmdloutdir,
            ssuffix='_dbscnd')

def process_tSNE(cmdlallattribcsv,
        cmdlcolsrange=None,
        cmdlcols2cluster=None,
        cmdlsample=None,
        cmdlxyzcols=None,
        cmdllearningrate=None,
        cmdlscalefeatures=True,
        cmdloutdir=None,
        cmdlhideplot=None):
    """Student t stochastic neighborhood embedding."""
    swa = pd.read_csv(cmdlallattribcsv)
    swaxx = swa.sample(frac=cmdlsample).copy()
    if cmdlcolsrange:
        print('Attrib From col# %d to col %d' % (cmdlcolsrange[0],cmdlcolsrange[1]))
        swax = swaxx[swaxx.columns[cmdlcolsrange[0]: cmdlcolsrange[1] + 1]]
    else:
        swax = swaxx[swaxx.columns[cmdlcols2cluster]]

    xyzc = swaxx[swaxx.columns[cmdlxyzcols]].copy()

    clustering = TSNE(n_components=2,
                learning_rate=cmdllearningrate)
    start_time = datetime.now()
    tsne_features = clustering.fit_transform(swax)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    xs = tsne_features[:,0]
    ys = tsne_features[:,1]

    # colvals = [dt.hour for dt in datashuf[MIN:MAX].index]
    # for i in range(len(cmdlcolorby)):
    for i in range(swax.shape[1]):
        colvals = swax.iloc[:,i].values
        minima = min(colvals)
        maxima = max(colvals)
        matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        # mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues_r)
        # mapper = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)
        # mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
        # mycolors = [mapper.to_rgba(v) for v in colvals]
        # clmp=cm.get_cmap('rainbow_r')
        clmp = cm.get_cmap('hsv')

        fig, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(xs,ys,
                 # c=mycolors,
                 # cmap=plt.cm.hsv,
                 c=colvals,
                 cmap=clmp,
                 s=10,
                 alpha=0.5)
        # cbar = plt.colorbar(scatter,mapper)
        plt.colorbar(scatter)
        plt.title('tSNE Colored by: %s' % swax.columns[i])
        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "c%s" % swax.columns[i] + "_tsne.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "c%s" % swax.columns[i] + "_tsne.pdf"
        fig.savefig(pdfcl)
        # plt.savefig(pdfcl)

    tsnescaled = StandardScaler().fit_transform(tsne_features)
    if cmdlscalefeatures:
        swaxx['tSNE0s'] = tsnescaled[:,0]
        swaxx['tSNE1s'] = tsnescaled[:,1]
        xyzc['tSNE0s'] = tsnescaled[:,0]
        xyzc['tSNE1s'] = tsnescaled[:,1]
    else:
        swaxx['tSNE0'] = tsne_features[:,0]
        swaxx['tSNE1'] = tsne_features[:,1]
        xyzc['tSNE0'] = tsne_features[:,0]
        xyzc['tSNE1'] = tsne_features[:,1]

    savefiles(seisf=cmdlallattribcsv,
        sdf=swaxx, sxydf=xyzc,
        outdir=cmdloutdir,
        ssuffix='_tsne')

def process_tSNE2(cmdlwellattribcsv,
        cmdlseisattribcsv,
        cmdlwcolsrange=None,
        cmdlwpredictorcols=None,
        cmdlwtargetcol=None,
        cmdlsxyzcols=None,
        cmdlscolsrange=None,
        cmdlspredictorcols=None,
        cmdlsample=None,
        cmdlwxyzcols=None,
        cmdllearningrate=None,
        cmdlscalefeatures=True,
        cmdloutdir=None,
        cmdlhideplot=None):
    """Student t stochastic neighborhood embedding Using seismic and well csv's."""
    swa = pd.read_csv(cmdlwellattribcsv)
    if cmdlwcolsrange:
        print('Attrib From col# %d to col %d' % (cmdlwcolsrange[0],cmdlwcolsrange[1]))
        swax = swa[swa.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]]
    else:
        swax = swa[swa.columns[cmdlwpredictorcols]]

    wxyzc = swax[swax.columns[cmdlwxyzcols]].copy()
    y = swa[swa.columns[cmdlwtargetcol]].copy()
    targetcname = swa.columns[cmdlwtargetcol]
    print('Target col: %s %d' % (targetcname,y.shape[0]))
    print(swa.columns)
    swa.drop(swa.columns[cmdlwtargetcol],axis=1,inplace=True)

    ssa = pd.read_csv(cmdlseisattribcsv)
    ssa = ssa.sample(frac=cmdlsample)
    ssxyz = ssa[ssa.columns[cmdlsxyzcols]].copy()

    if cmdlscolsrange:
        print('Seismic Predictors From col# %d to col %d' % (cmdlscolsrange[0],cmdlscolsrange[1]))
        sspred = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1] + 1]]
        # colnames = ssa[ssa.columns[cmdlscolsrange[0]: cmdlscolsrange[1]+1]].columns
    else:
        print('Seismic Predictor cols',cmdlspredictorcols)
        sspred = ssa[ssa.columns[cmdlspredictorcols]]
        # colnames = swa[swa.columns[cmdlspredictorcols]].columns

    alldata = pd.concat([swax,sspred])
    wrows = swax.shape[0]

    clustering = TSNE(n_components=2,
                learning_rate=cmdllearningrate)
    start_time = datetime.now()
    tsne_features = clustering.fit_transform(alldata)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    dirsplit,fextsplit = os.path.split(cmdlwellattribcsv)
    fname,fextn = os.path.splitext(fextsplit)

    tsnescaled = StandardScaler().fit_transform(tsne_features)
    if cmdlscalefeatures:
        wxs = tsnescaled[:wrows,0]
        wys = tsnescaled[:wrows,1]
        ssxs = tsnescaled[wrows:,0]
        ssys = tsnescaled[wrows:,1]
    else:
        wxs = tsne_features[:wrows,0]
        wys = tsne_features[:wrows,1]
        ssxs = tsne_features[wrows:,0]
        ssys = tsne_features[wrows:,1]

    swa['tSNE0s'] = wxs
    swa['tSNE1s'] = wys
    swa[targetcname] = y
    wxyzc['tSNE0s'] = wxs
    wxyzc['tSNE1s'] = wys
    wxyzc[targetcname] = y

    ssa['tSNE0s'] = ssxs
    ssa['tSNE1s'] = ssys
    ssxyz['tSNE0s'] = ssxs
    ssxyz['tSNE1s'] = ssys

    # colvals = [dt.hour for dt in datashuf[MIN:MAX].index]
    # for i in range(len(cmdlcolorby)):
    for i in range(alldata.shape[1]):
        colvals = alldata.iloc[:,i].values
        minima = min(colvals)
        maxima = max(colvals)
        matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        # mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues_r)
        # mapper = cm.ScalarMappable(norm=norm, cmap=cm.PiYG)
        # mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
        # mycolors = [mapper.to_rgba(v) for v in colvals]
        # clmp=cm.get_cmap('rainbow_r')
        clmp = cm.get_cmap('hsv')

        fig, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(tsne_features[:,0],tsne_features[:,1],
                 # c=mycolors,
                 # cmap=plt.cm.hsv,
                 c=colvals,
                 cmap=clmp,
                 s=10,
                 alpha=0.5)
        # cbar = plt.colorbar(scatter,mapper)
        plt.colorbar(scatter)
        plt.title('tSNE Colored by: %s' % swax.columns[i])
        if not cmdlhideplot:
            plt.show()
        if cmdloutdir:
            pdfcl = os.path.join(cmdloutdir,fname) + "c%s" % swax.columns[i] + "_tsne.pdf"
        else:
            pdfcl = os.path.join(dirsplit,fname) + "c%s" % swax.columns[i] + "_tsne.pdf"
        fig.savefig(pdfcl)
        # plt.savefig(pdfcl)

    savefiles(seisf=cmdlseisattribcsv,
        sdf=ssa, sxydf=ssxyz,
        wellf=cmdlwellattribcsv,
        wdf=swa, wxydf=wxyzc,
        outdir=cmdloutdir,
        ssuffix='_tsne',
        name2merge=cmdlwellattribcsv)

def process_umap(cmdlallattribcsv,
                cmdlcolsrange=None,
                cmdlcols2cluster=None,
                cmdlsample=None,
                cmdlxyzcols=None,
                cmdlnneighbors=None,
                cmdlmindistance=0.3,
                cmdlncomponents=3,
                cmdlscalefeatures=False,
                cmdloutdir=None,
                cmdlhideplot=None):
    """Uniform Manifold Approximation Projection Clustering."""
    swa = pd.read_csv(cmdlallattribcsv)
    swaxx = swa.sample(frac=cmdlsample).copy()
    print('# of components {:2d}'.format(cmdlncomponents))
    if cmdlcolsrange:
        print('Attrib From col# %d to col %d' % (cmdlcolsrange[0],cmdlcolsrange[1]))
        swax = swaxx[swaxx.columns[cmdlcolsrange[0]: cmdlcolsrange[1] + 1]]
    else:
        swax = swaxx[swaxx.columns[cmdlcols2cluster]]

    xyzc = swaxx[swaxx.columns[cmdlxyzcols]].copy()

    clustering = umap.UMAP(n_neighbors=cmdlnneighbors, min_dist=cmdlmindistance, n_components=cmdlncomponents)

    start_time = datetime.now()
    umap_features = clustering.fit_transform(swax)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    dirsplit,fextsplit = os.path.split(cmdlallattribcsv)
    fname,fextn = os.path.splitext(fextsplit)
    if cmdloutdir:
        pdfcl = os.path.join(cmdloutdir,fname) + "_umapnc%d.pdf" % (cmdlncomponents)
    else:
        pdfcl = os.path.join(dirsplit,fname) + "_umapnc%d.pdf" % (cmdlncomponents)
    fig, ax = plt.subplots(figsize=(8,6))

    nclst = [i for i in range(cmdlncomponents)]
    pltvar = itertools.combinations(nclst,2)
    pltvarlst = list(pltvar)
    for i in range(len(pltvarlst)):
        ftr0 = pltvarlst[i][0]
        ftr1 = pltvarlst[i][1]
        print('umap feature #: {}, umap feature #: {}'.format(ftr0,ftr1))
        # ax.scatter(umap_features[:,pltvarlst[i][0]],umap_features[:,pltvarlst[i][1]],s=2,alpha=.2)
        ax.scatter(umap_features[:,ftr0],umap_features[:,ftr1],s=2,alpha=.2)

        # ax.scatter(umap_features[:,0],umap_features[:,1],s=2,alpha=.2)
        # ax.scatter(umap_features[:,1],umap_features[:,2],s=2,alpha=.2)
        # ax.scatter(umap_features[:,2],umap_features[:,0],s=2,alpha=.2)

    if not cmdlhideplot:
        plt.show()
    fig.savefig(pdfcl)

    if cmdlscalefeatures:
        umapscaled = StandardScaler().fit_transform(umap_features)
        for i in range(cmdlncomponents):
            cname = 'umap' + str(i) + 's'
            swaxx[cname] = umapscaled[:,i]
            xyzc[cname] = umapscaled[:,i]

            # swaxx['umap0s'] = umapscaled[:,0]
            # swaxx['umap1s'] = umapscaled[:,1]
            # swaxx['umap2s'] = umapscaled[:,2]
            # xyzc['umap0s'] =   umapscaled[:,0]
            # xyzc['umap1s'] =   umapscaled[:,1]
            # xyzc['umap2s'] =   umapscaled[:,2]
    else:
        for i in range(cmdlncomponents):
            cname = 'umap' + str(i)
            swaxx[cname] = umap_features[:,i]
            xyzc[cname] = umap_features[:,i]

            # swaxx['umap0'] = umap_features[:,0]
            # swaxx['umap1'] = umap_features[:,1]
            # swaxx['umap2'] = umap_features[:,2]
            # xyzc['umap0'] =  umap_features[:,0]
            # xyzc['umap1'] =  umap_features[:,1]
            # xyzc['umap2'] =  umap_features[:,2]

    savefiles(seisf=cmdlallattribcsv,
        sdf=swaxx, sxydf=xyzc,
        outdir=cmdloutdir,
        ssuffix='_umapnc%d' % (cmdlncomponents))

def process_semisupervised(wfname,sfname,
        cmdlwcolsrange=None,
        cmdlwtargetcol=None,
        cmdlwellsxyzcols=None,
        cmdlsample=0.005,
        cmdlkernel='knn',
        cmdlnneighbors=7,
        cmdlcol2drop=None,
        cmdloutdir=None):
    """Semi supervised: creating extra data from existing data. Regression."""
    i4w = pd.read_csv(wfname)
    if cmdlcol2drop:
        i4w.drop(i4w.columns[cmdlcol2drop],axis=1,inplace=True)
    dirsplitw,fextsplit = os.path.split(wfname)
    fnamew,fextn = os.path.splitext(fextsplit)
    if cmdloutdir:
        ppsave = os.path.join(cmdloutdir,fnamew) + "_pw.csv"
    else:
        ppsave = os.path.join(dirsplitw,fnamew) + "_pw.csv"
    # print('target col',cmdlwtargetcol)
    # print(i4w[i4w.columns[cmdlwtargetcol]],i4w.columns[cmdlwtargetcol])
    # coln = i4w.columns[cmdlwtargetcol]
    # print('coln:',coln)
    if cmdlcol2drop:
        cmdlwtargetcol -= 1
    i4w['qa'],qbins = pd.qcut(i4w[i4w.columns[cmdlwtargetcol]],3,labels=['Low','Medium','High'],retbins=True)

    i4w['qcodes'] = i4w['qa'].cat.codes
    print('codes: ',i4w['qcodes'].unique())

    i4s = pd.read_csv(sfname)

    # i4w.drop(['Av_PHIT', 'qa'],axis=1,inplace=True)
    i4w.drop(i4w.columns[[cmdlwtargetcol,cmdlwtargetcol + 1]],axis=1,inplace=True)
    i4sx = i4s.sample(frac=cmdlsample,random_state=42)
    i4sxi = i4sx.reset_index()
    i4sxi.drop('index',axis=1,inplace=True)
    i4sxi['Well1'] = ['PW%d' % i for i in i4sxi.index]
    i4sxi.insert(0,'Well',i4sxi['Well1'])
    i4sxi.drop('Well1',axis=1,inplace=True)
    i4sxi['qcodes'] = [(-1) for i in i4sxi.index]
    wcols = list(i4w.columns)
    i4sxi.columns = wcols
    i4 = pd.concat([i4w,i4sxi],axis=0)
    X = i4[i4.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1]].values
    y = i4[i4.columns[cmdlwtargetcol]].values
    print(Counter(list(y)).items())
    lblsprd = LabelSpreading(kernel=cmdlkernel,n_neighbors=cmdlnneighbors)
    lblsprd.fit(X,y)
    ynew = lblsprd.predict(X)
    print(Counter(list(ynew)).items())
    i4['qcodespred'] = ynew
    i4.drop(i4.columns[cmdlwtargetcol],axis=1,inplace=True)
    i4.to_csv(ppsave,index=False)
    print('Successfully generated %s file' % ppsave)
    i4xy = i4.copy()
    i4xy.drop(i4xy.columns[cmdlwcolsrange[0]: cmdlwcolsrange[1] + 1],axis=1,inplace=True)
    if cmdloutdir:
        ppxysave = os.path.join(cmdloutdir,fnamew) + "_pw.txt"
    else:
        ppxysave = os.path.join(dirsplitw,fnamew) + "_pw.txt"
    i4xy.to_csv(ppxysave,sep=' ',index=False)
    print('Successfully generated %s file' % ppxysave)

def cmnt(line):
    """Check if a line is comment."""
    if '#' in line:
        return True
    else:
        return False

class ClassificationMetrics:
    """Compute Classification Metrics."""

    def __init__(self,actual,predicted,tolist=True,tocsv=False):
        """Initializer for class."""
        self.actual = actual
        self.predicted = predicted
        self.tolist = tolist
        self.tocsv = tocsv

    def comp_confusion(self):
        """Compute confusion matrix."""
        if self.tolist:
            print('Confusion Report: ')
            print(pd.crosstab(self.actual,self.predicted,rownames=['Actual'], colnames =['Predicted']))

    def comp_accuracy(self):
        """Compute accuracy."""
        if self.tolist:
            print('Accuracy Score: ',accuracy_score(self.actual,self.predicted))

    def comp_clfreport(self):
        """Generate report."""
        if self.tolist:
            print('Classification Report: ')
            print(classification_report(self.actual,self.predicted))

def getcommandline(*oneline):
    """Process command line interface."""
    allcommands = ['workflow','sattrib','dropcols','prepfile','prepfile','listcsvcols','sscalecols',
                'wscalecols','wattrib','wamerge','seiswellattrib','PCAanalysis','PCAfilter','scattermatrix',
                'EDA','linreg','featureranking','linfitpredict','KNNtest','KNNfitpredict','CatBoostRegressor',
                'CatBoostClassifier','testCmodels','logisticreg','GaussianNaiveBayes','clustertest','clustering',
                'tSNE','tSNE2','TuneCatBoostClassifier','TuneCatBoostRegressor','DBSCAN','wscaletarget','semisupervised',
                'GaussianMixtureModel','ANNRegressor','NuSVR','NuSVC','SGDR','QuadraticDiscriminantAnalysis','umap']

    mainparser = argparse.ArgumentParser(description='Seismic and Well Attributes Modeling.')
    mainparser.set_defaults(which=None)
    subparser = mainparser.add_subparsers(help='File name listing all attribute grids')

    wrkflowparser = subparser.add_parser('workflow',help='Workflow file instead of manual steps')
    wrkflowparser.set_defaults(which='workflow')
    wrkflowparser.add_argument('commandfile',help='File listing workflow')
    wrkflowparser.add_argument('--startline',type=int,default =0,help='Line in file to start flow from. default=0')
    # wrkflowparser.add_argument('--stopat',type=int,default =None,help='Line in file to end flow after. default=none')

    saparser = subparser.add_parser('sattrib',help='Merge Seismic Attributes Grids')
    saparser.set_defaults(which ='sattrib')
    saparser.add_argument('gridfileslist',help='grid list file name')
    saparser.add_argument('--gridheader',type=int,default=0,
                        help='grid header lines to skip. default=0')
    saparser.add_argument('--gridcols',nargs='+',type=int,default=(2,3,4),
                        help='grid columns x y z or il xl x y z. default = 2 3 4')
    saparser.add_argument('--ilxl',action='store_true',default=False,
                        help='Use IL XL to fillna with mean of column Enter col loctions in gridcols default= False')
    saparser.add_argument('--outdir',help='output directory,default= same dir as input')

    dropparser = subparser.add_parser('dropcols',help='csv drop columns')
    dropparser.set_defaults(which='dropcols')
    dropparser.add_argument('csvfile',help='csv file to drop columns')
    dropparser.add_argument('--cols2drop',type=int,nargs='*',default=None,help='default=none')
    dropparser.add_argument('--outdir',help='output directory,default= same dir as input')

    dprepparser = subparser.add_parser('prepfile',help='Remove extra rows and columns from csv from Petrel')
    dprepparser.set_defaults(which='prepfile')
    dprepparser.add_argument('csvfile',help='csv file name')
    dprepparser.add_argument('--outdir',help='output directory,default= same dir as input')

    listcolparser = subparser.add_parser('listcsvcols',help='List header row of any csv')
    listcolparser.set_defaults(which='listcsvcols')
    listcolparser.add_argument('csvfile',help='csv file name')

    sscaleparser = subparser.add_parser('sscalecols',help='seismic csv scale columns other than xyz')
    sscaleparser.set_defaults(which='sscalecols')
    sscaleparser.add_argument('csvfile',help='csv file to scale columns')
    sscaleparser.add_argument('--xyzcols',type=int,default=[0,1,2],nargs=3,
                        help='X Y Z columns to remove before scaling and add back after. default = 0 1 2')
    sscaleparser.add_argument('--includexyz',action='store_true',default=False,
        help='Include x y z columns in scaling. default=False')
    sscaleparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                        help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    sscaleparser.add_argument('--outdir',help='output directory,default= same dir as input')

    wscaleparser = subparser.add_parser('wscalecols',help='well csv scale columns other than wxyz')
    wscaleparser.set_defaults(which='wscalecols')
    wscaleparser.add_argument('csvfile',help='csv file to scale columns')
    wscaleparser.add_argument('--wxyzcols',type=int,default=[0,1,2,3],nargs=4,
                        help=' Well X Y Z columns to remove before scaling and add back after. default = 0 1 2 3')
    wscaleparser.add_argument('--outdir',help='output directory,default= same dir as input')

    wstrgtparser = subparser.add_parser('wscaletarget',help='well csv scale target column only')
    wstrgtparser.set_defaults(which='wscaletarget')
    wstrgtparser.add_argument('csvfile',help='csv file to scale columns')
    wstrgtparser.add_argument('--targetcol',type=int,default=-1,
                      help=' Well attribute target column to scale. default = last col')
    wstrgtparser.add_argument('--kind',choices=['standard','quniform','qnormal'],default='standard',
                      help='Scaling kind: Standard, quantile uniform, or quantile normal. default=standard')
    wstrgtparser.add_argument('--outdir',help='output directory,default= same dir as input')

    waparser = subparser.add_parser('wattrib',help='Merge Well Petrophysics csv with well xyz csv ')
    waparser.set_defaults(which='wattrib')
    waparser.add_argument('wattribfile',help='Well attributes file, wname attrib csv format')
    waparser.add_argument('wxyfile',help='All Wells list of wname x y z csv format ')
    waparser.add_argument('--wattribcols',type=int,default=[0,1],nargs=2,
                        help='col nums of wellname attrib . dfv= 0 1 ')
    waparser.add_argument('--wxyzcols',type=int,nargs=4,default=[0,1,2,3],
                        help='well x y columns.default = 0 1 2 3')
    waparser.add_argument('--fillna',choices=['delete','mean','median'],default='delete',
                        help='Fill nan with either mean or median. default= delete complete row')
    waparser.add_argument('--outdir',help='output directory,default= same dir as input')

    wamparser = subparser.add_parser('wamerge',help='Merge Well Attributes in csv format')
    wamparser.set_defaults(which='wamerge')
    wamparser.add_argument('csvfileslist',help='csv list file name')
    wamparser.add_argument('--csvskiprows',type=int,default=3,
                        help='csv header lines to skip. default=3')
    wamparser.add_argument('--csvcols',nargs=5,type=int,default=(1,3,4,5,6),
                        help='csv wellname x y z attribute columns. default = 1 3 4 5 6')
    wamparser.add_argument('--outdir',help='output directory,default= same dir as input')

    swaparser = subparser.add_parser('seiswellattrib',help='Back Interpolate wells at all attributes')
    swaparser.set_defaults(which='seiswellattrib')
    swaparser.add_argument('seiscsv',help='Seismic attributes csv file name')
    swaparser.add_argument('wellcsv',help='Well x y attribute csv file name')
    swaparser.add_argument('--wellcsvcols',type=int,nargs=5,default=[0,1,2,3,4],
                        help='Columns well x y z wellatrribute. default= 0 1 2 3 4')
    swaparser.add_argument('--interpolate',default='idw',choices=['idw','linear','cubic','nearest','rbf','avgmap','triang','ct'],
                        help='Interpolation type. dfv= idw. Other types uses griddata only for no polygon input')
    swaparser.add_argument('--radius',default=5000.00, type=float,help='search radius for zmap interpolation. use w/ -n avgmap.dfv=5000m')
    swaparser.add_argument('--outdir',help='output directory,default= same dir as input')

    pcaaparser = subparser.add_parser('PCAanalysis',help='PCA analysis')
    pcaaparser.set_defaults(which='PCAanalysis')
    pcaaparser.add_argument('allattribcsv',help='csv file with all previously scaled predictors with target column')
    pcaaparser.add_argument('--analysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    pcaaparser.add_argument('--acolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    pcaaparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    pcafparser = subparser.add_parser('PCAfilter',help='PCA filter')
    pcafparser.set_defaults(which='PCAfilter')
    pcafparser.add_argument('allattribcsv',help='csv file with all previously scaled predictors with target column')
    pcafparser.add_argument('--analysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    pcafparser.add_argument('--acolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    pcafparser.add_argument('--ncomponents',type=int,default=None,help='# of components to keep,default =none')
    pcafparser.add_argument('--targetcol',type=int,default=None,
        help='Target column # to add back. You do not have to add a target default = none')
    pcafparser.add_argument('--cols2addback',type=int,nargs='+',default=[0,1,2],
        help='Columns, e.g. well x y z  to remove from fitting model, but addback to saved file. default= 0 1 2 3 ')
    pcafparser.add_argument('--outdir',help='output directory,default= same dir as input')

    sctrmparser = subparser.add_parser('scattermatrix',help='Scatter matrix of all predictors and target')
    sctrmparser.set_defaults(which='scattermatrix')
    sctrmparser.add_argument('allattribcsv',help='csv file with all predictors and target column')
    sctrmparser.add_argument('--wellxyzcols',type=int,nargs='+',default=[0,1,2,3],
        help='Columns well x y z  to remove from fitting model. default= 0 1 2 3 ')
    sctrmparser.add_argument('--sample',type=float,default=.5,help='fraction of data of sample.default=0.5')

    # *************EDA
    edaparser = subparser.add_parser('EDA',help='Exploratory Data Analysis')
    edaparser.set_defaults(which='EDA')
    edaparser.add_argument('allattribcsv',help='csv file with all predictors and target column')
    edaparser.add_argument('--xyzcols',type=int,nargs='+',help='Any # of columns to remove before analysis,e.g. x y z')
    edaparser.add_argument('--polydeg',type=int, default=1,help='degree of polynomial to fit data in xplots choice. default = 1, i.e. st line')
    edaparser.add_argument('--sample',type=float,default=.5,help='fraction of data of sample for ScatterMatrix Plot.default=0.5')
    edaparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')
    edaparser.add_argument('--plotoption',
        choices=['xplots','heatmap','box','distribution','scattermatrix'],help='choices: xplots,heatmap,box,distribution,scattermatrix',default='heatmap')
    edaparser.add_argument('--outdir',help='output directory,default= same dir as input')

    # *************linreg linear regression
    lrparser = subparser.add_parser('linreg',help='Linear Regression Model fit and predict')
    lrparser.set_defaults(which='linreg')
    lrparser.add_argument('allattribcsv',help='csv file with all predictors and target column')
    lrparser.add_argument('--wellxyzcols',type=int,nargs=4,default=[0,1,2,3],
        help='Columns well x y z  to remove from fitting model. default= 0 1 2 3 ')
    lrparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    lrparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    lrparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    lrparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    lrparser.add_argument('--outdir',help='output directory,default= same dir as input')

    frparser = subparser.add_parser('featureranking',help='Ranking of attributes')
    frparser.set_defaults(which='featureranking')
    frparser.add_argument('allattribcsv',help='csv file with all predictors and target column')
    frparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    frparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    frparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    frparser.add_argument('--testfeatures',choices=['rfe','rlasso','svr','svrcv','rfregressor','decisiontree'],default='rfregressor',
        help='Test for features significance: Randomized Lasso, recursive feature elimination #,default= rfregressor')
    # lassalpha is used with randomized lasso only
    frparser.add_argument('--lassoalpha',type=float,default=0.025,help='alpha = 0 is OLS. default=0.005')
    # features2keep is used with svr only
    frparser.add_argument('--features2keep',type=int,default=5,help='#of features to keep in rfe.default=5')
    # following 2 are used with any cross validation e.g. random forest regressor, svrcv
    frparser.add_argument('--cv',type=int,default=3,help='Cross Validation nfold. default=3')
    frparser.add_argument('--traintestsplit',type=float,default=.3,help='Train Test split. default = 0.3')

    lfpparser = subparser.add_parser('linfitpredict',help='Linear Regression fit on one data set and predicting on another ')
    lfpparser.set_defaults(which='linfitpredict')
    lfpparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    lfpparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    lfpparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    lfpparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    lfpparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    lfpparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    lfpparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='Seismic Predictor min max col #')
    lfpparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    lfpparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],
        help='Seismic csv x y z columns  . default= 0 1 2 ')
    lfpparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    lfpparser.add_argument('--outdir',help='output directory,default= same dir as input')

    knntparser = subparser.add_parser('KNNtest',help='Test number of nearest neighbors for KNN')
    knntparser.set_defaults(which='KNNtest')
    knntparser.add_argument('allattribcsv',help='csv file with all previously scaled predictors with target column')
    knntparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column #.default = last col')
    knntparser.add_argument('--predictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    knntparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='KNN test min max col #')
    # knntparser.add_argument('--traintestsplit',type=float,default=.2,
    #     help='train test split ratio, default= 0.2, i.e. keep 20%% for test')
    knntparser.add_argument('--cv',type=int,default=3,help='Cross Validation nfold. default=3')
    knntparser.add_argument('--sample',type=float,default=1.0,help='fraction of data of sample.default=1, i.e. all data')
    knntparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')
    knntparser.add_argument('--outdir',help='output directory,default= same dir as input')

    knnfparser = subparser.add_parser('KNNfitpredict',help='KNN fit on one data set and predicting on another ')
    knnfparser.set_defaults(which='KNNfitpredict')
    knnfparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    knnfparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    knnfparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    knnfparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    knnfparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    knnfparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    knnfparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='Seismic Predictor min max col #')
    knnfparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    knnfparser.add_argument('--kneighbors',type=int,default=10,help='# of nearest neighbors. default = 10')
    knnfparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns.default= 0 1 2 ')
    knnfparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    knnfparser.add_argument('--outdir',help='output directory,default= same dir as input')
    knnfparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    tcbrparser = subparser.add_parser('TuneCatBoostRegressor',help='Hyper Parameter Tuning of CatBoost Regression')
    tcbrparser.set_defaults(which='TuneCatBoostRegressor')
    tcbrparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    tcbrparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    tcbrparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    tcbrparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    tcbrparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default =-1')
    tcbrparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    tcbrparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    tcbrparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    tcbrparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols,default= 0 1 2 3')
    tcbrparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    tcbrparser.add_argument('--outdir',help='output directory,default= same dir as input')
    tcbrparser.add_argument('--iterations',type=int,nargs='+',default=[10,500,1000,5000],
        help='Learning Iterations, default =[10,500,1000,5000]')
    tcbrparser.add_argument('--learningrate',type=float,nargs='+', default=[0.01,0.03,0.1],
        help='learning_rate. default=[0.01,0.03,0.1]')
    tcbrparser.add_argument('--depth',type=int,nargs='+',default=[2,4,6,8],help='depth of trees. default=[2,4,6,8]')
    tcbrparser.add_argument('--cv',type=int,default=3,help='Cross Validation default=3')
    tcbrparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    # *************SVR support vector regresssion: uses nusvr
    nsvrparser = subparser.add_parser('NuSVR',help='Nu Support Vector Machine Regressor')
    nsvrparser.set_defaults(which='NuSVR')
    nsvrparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    nsvrparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    nsvrparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    nsvrparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    nsvrparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    nsvrparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    nsvrparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='Seismic Predictor min max col #')
    nsvrparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    nsvrparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,
                            help='Min Max scale limits. default=use input data limits ')

    nsvrparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns.default= 0 1 2 ')
    nsvrparser.add_argument('--outdir',help='output directory,default= same dir as input')
    nsvrparser.add_argument('--nu',type=float,default=0.5,help='upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. value between 0 1, default =0.5')
    nsvrparser.add_argument('--errpenalty',type=float,default=1.0,help='error penalty. default=1.0')
    nsvrparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    nsvrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    nsvrparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    nsvrparser.add_argument('--radius',default=5000.00, type=float,
        help='search radius for map interpolation. dfv=5000m using idw')
    nsvrparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    # *************SGD Regressor
    sgdrparser = subparser.add_parser('SGDR',help='Stochastic Gradient Descent Regressor: OLS/Lasso/Ridge/ElasticNet')
    sgdrparser.set_defaults(which='SGDR')
    sgdrparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    sgdrparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    sgdrparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    sgdrparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    sgdrparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    sgdrparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    sgdrparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='Seismic Predictor min max col #')
    sgdrparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    sgdrparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,
        help='Min Max scale limits. default=use input data limits ')
    sgdrparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns.default= 0 1 2 ')
    sgdrparser.add_argument('--outdir',help='output directory,default= same dir as input')
    sgdrparser.add_argument('--loss',choices=['squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive'],default='squared_loss',
        help='default= squared_loss')
    sgdrparser.add_argument('--penalty',choices=['l1','l2','elasticnet','none'],default='l2',help='default=l2')
    sgdrparser.add_argument('--l1ratio',type=float,default=0.15,help='elastic net mixing: 0 (l2)to 1 (l1), default =0.15')
    sgdrparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    sgdrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    sgdrparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    sgdrparser.add_argument('--radius',default=5000.00, type=float,
        help='search radius for map interpolation. dfv=5000m using idw')
    sgdrparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    # *************CatBoostRegressor
    cbrparser = subparser.add_parser('CatBoostRegressor',help='CatBoost Regressor')
    cbrparser.set_defaults(which='CatBoostRegressor')
    cbrparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    cbrparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    cbrparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    cbrparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    cbrparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    cbrparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    cbrparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='Seismic Predictor min max col #')
    cbrparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    cbrparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,
        help='Min Max scale limits. default=use input data limits ')
    cbrparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns.default= 0 1 2 ')
    cbrparser.add_argument('--outdir',help='output directory,default= same dir as input')
    cbrparser.add_argument('--iterations',type=int,default=500,help='Learning Iterations, default =500')
    cbrparser.add_argument('--learningrate',type=float,default=0.03,help='learning_rate. default=0.03')
    cbrparser.add_argument('--depth',type=int,default=2,help='depth of trees. default=2')
    cbrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    cbrparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    cbrparser.add_argument('--featureimportance',action='store_true',default=False,
        help='List feature importance.default= False')
    cbrparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    cbrparser.add_argument('--nofilesout',action='store_true',default=False,
        help='Do not create csv/txt files. default = True. Use to test hyperparameters')
    cbrparser.add_argument('--overfittingdetection',action='store_true',default=False,
        help='Over Fitting Detection.default= False')
    cbrparser.add_argument('--odpval',type=float,default=0.005,
        help='ranges from 10e-10 to 10e-2. Used with overfittingdetection')
    cbrparser.add_argument('--radius',default=5000.00, type=float,
        help='search radius for map interpolation. dfv=5000m using idw')
    cbrparser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')

    # *************ANNRegressor
    annrparser = subparser.add_parser('ANNRegressor',help='Artificial Neural Network')
    annrparser.set_defaults(which='ANNRegressor')
    annrparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    annrparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    annrparser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    annrparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    annrparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    annrparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    annrparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='Seismic Predictor min max col #')
    annrparser.add_argument('--minmaxscale',action='store_true',default=False,
        help='Apply min max scaler to scale predicted to input range. default= False i.e. no scaling')
    annrparser.add_argument('--scaleminmaxvalues',type=float,nargs=2,
        help='Min Max scale limits. default=use input data limits ')
    annrparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns.default= 0 1 2 ')
    annrparser.add_argument('--outdir',help='output directory,default= same dir as input')
    annrparser.add_argument('--nodes',type=int,nargs='+',help='# of nodes in each layer. no defaults')
    annrparser.add_argument('--activation',choices=['relu','sigmoid'],nargs='+',
        help='activation per layer.choices: relu or sigmoid. no default, repeat for number of layers')
    annrparser.add_argument('--epochs',type=int,default=100,help='depth of trees. default=100')
    annrparser.add_argument('--batch',type=int,default=5,help='depth of trees. default=5')
    annrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    annrparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    annrparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    annrparser.add_argument('--radius',default=5000.00, type=float,
        help='search radius for map interpolation. dfv=5000m using idw')
    annrparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    tcmdlparser = subparser.add_parser('testCmodels',help='Test Classification models')
    tcmdlparser.set_defaults(which='testCmodels')
    tcmdlparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    tcmdlparser.add_argument('--wanalysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    tcmdlparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    tcmdlparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default =-1')
    tcmdlparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    tcmdlparser.add_argument('--cv',type=int,default=3,help='Cross Validation nfold. default=3')
    tcmdlparser.add_argument('--outdir',help='output directory,default= same dir as input')
    tcmdlparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    lgrparser = subparser.add_parser('logisticreg',help='Apply Logistic Regression Classification')
    lgrparser.set_defaults(which='logisticreg')
    lgrparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    lgrparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    lgrparser.add_argument('--wanalysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    lgrparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    lgrparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default = last col')
    lgrparser.add_argument('--coded',action='store_true',default=False,
        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    lgrparser.add_argument('--classweight',action='store_true',default=False,
        help='Balance classes by proportional weighting. default =False -> no balancing')
    lgrparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    lgrparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    lgrparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    lgrparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    lgrparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols,default= 0 1 2 3')
    lgrparser.add_argument('--outdir',help='output directory,default= same dir as input')
    lgrparser.add_argument('--cv',type=int,help='Cross Validation default=None')
    lgrparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    lgrparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    lgrparser.add_argument('--balancetype',choices=['ros','smote','adasyn'],default=None,
        help='Random Oversampling,Synthetic Minority Oversampling,Adaptive Synthetic Sampling,default=None')
    lgrparser.add_argument('--nneighbors',type=int,default=1,help='use only for smote and adasyn')
    lgrparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    nbparser = subparser.add_parser('GaussianNaiveBayes',help='Apply Gaussian Naive Bayes Classification')
    nbparser.set_defaults(which='GaussianNaiveBayes')
    nbparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    nbparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    nbparser.add_argument('--wanalysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    nbparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    nbparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default =last col')
    nbparser.add_argument('--coded',action='store_true',default=False,
        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    nbparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    nbparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    nbparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    nbparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    nbparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols')
    nbparser.add_argument('--outdir',help='output directory,default= same dir as input')
    nbparser.add_argument('--cv',type=int,help='Cross Validation default=None.')
    nbparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    nbparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    nbparser.add_argument('--balancetype',choices=['ros','smote','adasyn'],default=None,
        help='Random Oversampling,Synthetic Minority Oversampling,Adaptive Synthetic Sampling,default=None')
    nbparser.add_argument('--nneighbors',type=int,default=1,help='use only for smote and adasyn')
    nbparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    qdaparser = subparser.add_parser('QuadraticDiscriminantAnalysis',help='Apply Quadratic Discriminant Analysis Classification')
    qdaparser.set_defaults(which='QuadraticDiscriminantAnalysis')
    qdaparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    qdaparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    qdaparser.add_argument('--wanalysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    qdaparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    qdaparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default =last col')
    qdaparser.add_argument('--coded',action='store_true',default=False,
        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    qdaparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    qdaparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    qdaparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    qdaparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    qdaparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols')
    qdaparser.add_argument('--outdir',help='output directory,default= same dir as input')
    qdaparser.add_argument('--cv',type=int,help='Cross Validation default=None.')
    qdaparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    qdaparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    qdaparser.add_argument('--balancetype',choices=['ros','smote','adasyn'],default=None,
        help='Random Oversampling,Synthetic Minority Oversampling,Adaptive Synthetic Sampling,default=None')
    qdaparser.add_argument('--nneighbors',type=int,default=1,help='use only for smote and adasyn')
    qdaparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    nsvcparser = subparser.add_parser('NuSVC',help='Apply Nu Support Vector Machine Classification')
    nsvcparser.set_defaults(which='NuSVC')
    nsvcparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    nsvcparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    nsvcparser.add_argument('--wanalysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    nsvcparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    nsvcparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default =last col')
    nsvcparser.add_argument('--coded',action='store_true',default=False,
        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    nsvcparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    nsvcparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    nsvcparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    nsvcparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    nsvcparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols')
    nsvcparser.add_argument('--nu',type=float,default=0.5,
            help='upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. value between 0 1, default =0.5')
    nsvcparser.add_argument('--outdir',help='output directory,default= same dir as input')
    nsvcparser.add_argument('--cv',type=int,help='Cross Validation default=None.')
    nsvcparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    nsvcparser.add_argument('--balancetype',choices=['ros','smote','adasyn'],default=None,
        help='Random Oversampling,Synthetic Minority Oversampling,Adaptive Synthetic Sampling,default=None')
    nsvcparser.add_argument('--nneighbors',type=int,default=1,help='use only for smote and adasyn')
    nsvcparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    nsvcparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    cbcparser = subparser.add_parser('CatBoostClassifier',help='Apply CatBoost Classification - Multi Class')
    cbcparser.set_defaults(which='CatBoostClassifier')
    cbcparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    cbcparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    cbcparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    cbcparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    cbcparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column #.default = last col')
    cbcparser.add_argument('--coded',action='store_true',default=False,
        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    cbcparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    cbcparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    cbcparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    cbcparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    cbcparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols,default= 0 1 2 3')
    cbcparser.add_argument('--outdir',help='output directory,default= same dir as input')
    cbcparser.add_argument('--iterations',type=int,default=500,help='Learning Iterations, default =500')
    cbcparser.add_argument('--learningrate',type=float,default=0.3,help='learning_rate. default=0.3')
    cbcparser.add_argument('--depth',type=int,default=2,help='depth of trees. default=2')
    cbcparser.add_argument('--generatesamples',type=int,default=0,
        help='Use GMM to generate samples. Use when you have small # of data to model from.default = 0')
    cbcparser.add_argument('--cv',type=int,default=None,help='Cross Validate. default=None.')
    cbcparser.add_argument('--featureimportance',action='store_true',default=False,
        help='List feature importance.default= False')
    cbcparser.add_argument('--valsize',type=float,default=0.3,help='Validation. default=0.3')
    cbcparser.add_argument('--balancetype',choices=['ros','smote','adasyn'],default=None,
        help='Random Oversampling,Synthetic Minority Oversampling,Adaptive Synthetic Sampling,default=None')
    cbcparser.add_argument('--nneighbors',type=int,default=1,help='use only for smote and adasyn')
    cbcparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    tcbcparser = subparser.add_parser('TuneCatBoostClassifier',help='Hyper Parameter Tuning of CatBoost Classification - Multi Class')
    tcbcparser.set_defaults(which='TuneCatBoostClassifier')
    tcbcparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    tcbcparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    tcbcparser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    tcbcparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    tcbcparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default =-1')
    tcbcparser.add_argument('--coded',action='store_true',default=False,
        help='Target col is already coded-> output from semisupervised. default =False target is not coded')
    tcbcparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    tcbcparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    tcbcparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    tcbcparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    tcbcparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols,default= 0 1 2 3')
    tcbcparser.add_argument('--outdir',help='output directory,default= same dir as input')
    tcbcparser.add_argument('--iterations',type=int,nargs='+',default=[10,500,1000,5000],
        help='Learning Iterations, default =[10,500,1000,5000]')
    tcbcparser.add_argument('--learningrate',type=float,nargs='+', default=[0.01,0.03,0.1],
        help='learning_rate. default=[0.01,0.03,0.1]')
    tcbcparser.add_argument('--depth',type=int,nargs='+',default=[2,4,6,8],help='depth of trees. default=[2,4,6,8]')
    tcbcparser.add_argument('--cv',type=int,default=3,help='Cross Validation default=3')
    tcbcparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    clparser = subparser.add_parser('clustertest',help='Testing of KMeans # of clusters using elbow plot')
    clparser.set_defaults(which='clustertest')
    clparser.add_argument('allattribcsv',help='csv file with all previously scaled predictors ')
    clparser.add_argument('--cols2cluster',type=int,nargs='+',default=[3,4,5],
        help='Columns to use for clustering. default= 3 4 5 ')
    clparser.add_argument('--colsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    clparser.add_argument('--sample',type=float,default=1.0,help='fraction of data of sample.default=1, i.e. all data')
    clparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')
    clparser.add_argument('--outdir',help='output directory,default= same dir as input')

    cl1parser = subparser.add_parser('clustering',help='Apply KMeans clustering')
    cl1parser.set_defaults(which='clustering')
    cl1parser.add_argument('allattribcsv',help='csv file will all attributes')
    cl1parser.add_argument('--cols2cluster',type=int,nargs='+',default=[3,4,5],
        help='Columns to use for clustering. default= 3 4 5 ')
    cl1parser.add_argument('--colsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    cl1parser.add_argument('--nclusters',type=int,default=5,help='# of clusters. default = 5')
    cl1parser.add_argument('--plotsilhouette',action='store_true',default=False,help='Plot Silhouete. default=False')
    cl1parser.add_argument('--sample',type=float,default=1.0,
        help='fraction of data of sample.default=1, i.e. all data. Use with plotsilhouette')
    cl1parser.add_argument('--xyzcols',type=int,nargs='+',default=[0,1,2],help='Attribute csv x y z columns.default= 0 1 2 ')
    cl1parser.add_argument('--addclass',choices=['labels','dummies'],default='labels',
        help='add cluster labels or binary dummies.default=labels')
    cl1parser.add_argument('--outdir',help='output directory,default= same dir as input')
    cl1parser.add_argument('--hideplot',action='store_true',default=False,
                        help='Only save to pdf. default =show and save')

    gmmparser = subparser.add_parser('GaussianMixtureModel',
        help='Gaussian Mixture Model. model well csv apply to seismic csv')
    gmmparser.set_defaults(which='GaussianMixtureModel')
    gmmparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    gmmparser.add_argument('--bayesian',action='store_true',default=False,
        help='Bayesian Gauusian Mixture Model. default= use Gaussian Mixture Model')
    gmmparser.add_argument('--seisattribcsv',help='csv file of seismic attributes to predict at')
    gmmparser.add_argument('--wanalysiscols',type=int,nargs='+',help='Predictor column #s, no default')
    gmmparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    gmmparser.add_argument('--wtargetcol',type=int,help='Target column # in well csv file. no default ')
    gmmparser.add_argument('--ncomponents',type=int,default=4,help='# of clusters.default=4')
    gmmparser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    gmmparser.add_argument('--catcol',type=int,default=None,
        help='Column num to convert from categories to dummies.Only one column is allowed. default=None')
    gmmparser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    gmmparser.add_argument('--saxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns  . default= 0 1 2 ')
    gmmparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols,default= 0 1 2 3')
    gmmparser.add_argument('--outdir',help='output directory,default= same dir as input')
    gmmparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    tsneparser = subparser.add_parser('tSNE',
        help='Apply tSNE (t distribution Stochastic Neighbor Embedding) clustering to one csv')
    tsneparser.set_defaults(which='tSNE')
    tsneparser.add_argument('allattribcsv',help='csv file will all attributes')
    tsneparser.add_argument('--cols2cluster',type=int,nargs='+',default=[3,4,5],
        help='Columns to use for clustering. default= 3 4 5 ')
    tsneparser.add_argument('--colsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    tsneparser.add_argument('--xyzcols',type=int,nargs='+',default=[0,1,2],help='Attribute csv x y z columns  . default= 0 1 2 ')
    # tsneparser.add_argument('--targetcol',type=int,default = None,
    # help='Target column # to add back. You do not have to add a target default = none')
    tsneparser.add_argument('--learningrate',type=int,default=200,help='Learning rate. default=200')
    tsneparser.add_argument('--sample',type=float,default=0.2,help='fraction of data of sample.default=0.2')
    tsneparser.add_argument('--scalefeatures',action='store_false',default=True,
        help='Do not scale tSNE feature. default = to scale featues')
    tsneparser.add_argument('--outdir',help='output directory,default= same dir as input')
    tsneparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    tsne2parser = subparser.add_parser('tSNE2',
        help='Apply tSNE (t distribution Stochastic Neighbor Embedding) clustering to both well and seismic csv')
    tsne2parser.set_defaults(which='tSNE2')
    tsne2parser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    tsne2parser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    tsne2parser.add_argument('--wtargetcol',type=int, default=-1,help='Target column #.default = last col')
    tsne2parser.add_argument('--wpredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    tsne2parser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='Wells Predictor min max col #')
    tsne2parser.add_argument('--wxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols,default= 0 1 2 3')
    tsne2parser.add_argument('--spredictorcols',type=int,nargs='+',help='Predictor column #s, no default')
    tsne2parser.add_argument('--scolsrange',type=int,nargs=2,default=None,help='Seismic Predictor min max col #')
    tsne2parser.add_argument('--sxyzcols',type=int,nargs=3,default=[0,1,2],help='Seismic csv x y z columns.default= 0 1 2 ')
    tsne2parser.add_argument('--learningrate',type=int,default=200,help='Learning rate. default=200')
    tsne2parser.add_argument('--sample',type=float,default=0.2,help='fraction of data of sample.default=0.2')
    tsne2parser.add_argument('--scalefeatures',action='store_false',default=True,
        help='Do not scale tSNE feature. default = to scale featues')
    tsne2parser.add_argument('--outdir',help='output directory,default= same dir as input')
    tsne2parser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    umapparser = subparser.add_parser('umap',help='Clustering using UMAP (Uniform Manifold Approximation & Projection) to one csv')
    umapparser.set_defaults(which='umap')
    umapparser.add_argument('allattribcsv',help='csv file will all attributes')
    umapparser.add_argument('--cols2cluster',type=int,nargs='+',default=[3,4,5],help='Columns to use for clustering. default= 3 4 5 ')
    umapparser.add_argument('--colsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    umapparser.add_argument('--xyzcols',type=int,nargs='+',default=[0,1,2],help='Attribute csv x y z columns  . default= 0 1 2 ')
    umapparser.add_argument('--nneighbors',type=int,default=5,help='Nearest neighbors. default=5')
    umapparser.add_argument('--mindistance',type=float,default=0.3,help='Min distantce for clustering. default=0.3')
    umapparser.add_argument('--ncomponents',type=int,default=3,help='Projection axes. default=3')
    umapparser.add_argument('--sample',type=float,default=1,help='fraction of data of sample 0 -> 1.default=1, no sampling')
    umapparser.add_argument('--scalefeatures',action='store_true',default=False,
        help='Do not scale umap features. default = not to scale featues')
    umapparser.add_argument('--outdir',help='output directory,default= same dir as input')
    umapparser.add_argument('--hideplot',action='store_true',default=False,
        help='Only save to pdf. default =show and save')

    dbsnparser = subparser.add_parser('DBSCAN',help='Apply DBSCAN (Density Based Spatial Aanalysis with Noise) clustering')
    dbsnparser.set_defaults(which='DBSCAN')
    dbsnparser.add_argument('allattribcsv',help='csv file will all attributes')
    dbsnparser.add_argument('--cols2cluster',type=int,nargs='+',default=[3,4,5],
        help='Columns to use for clustering. default= 3 4 5 ')
    dbsnparser.add_argument('--colsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    dbsnparser.add_argument('--xyzcols',type=int,nargs='+',default=[0,1,2],help='Attribute csv x y z columns  . default= 0 1 2 ')
    dbsnparser.add_argument('--targetcol',type=int,default=None,
        help='Target column # to add back. You do not have to add a target default = none')
    dbsnparser.add_argument('--eps',type=float,default=0.5,help='eps. default=0.5')
    dbsnparser.add_argument('--minsamples',type=int,default=10,help='minsamples. default=10')
    dbsnparser.add_argument('--addclass',choices=['labels','dummies'],default='labels',
        help='add cluster labels or binary dummies.default=labels')
    dbsnparser.add_argument('--outdir',help='output directory,default= same dir as input')

    sspparser = subparser.add_parser('semisupervised',help='Apply semi supervised Class prediction ')
    sspparser.set_defaults(which='semisupervised')
    sspparser.add_argument('wellattribcsv',help='csv file of all attributes at well locations to fit model')
    sspparser.add_argument('seisattribcsv',help='csv file of seismic attributes to predict at')
    sspparser.add_argument('--wcolsrange',type=int,nargs=2,default=None,help='analysis min max col #')
    sspparser.add_argument('--wtargetcol',type=int,default=-1,help='Target column # in well csv file. default = last column')
    sspparser.add_argument('--wellsxyzcols',type=int,nargs=4,default=[0,1,2,3],help='well x y z cols,default= 0 1 2 3')
    sspparser.add_argument('--col2drop',type=int,default=None,help='drop column in case of scaled target.default=None')
    # sspparser.add_argument('--qcut',type=int,default=3,help='Divisions to input target data set. default = 3')
    sspparser.add_argument('--sample',type=float,default=.005,help='fraction of data of sample.default=0.005')
    sspparser.add_argument('--outdir',help='output directory,default= same dir as input')
    sspparser.add_argument('--nneighbors',type=int,default=7,help='Used with knn to classify data.default=7')
    sspparser.add_argument('--kernel',choices=['knn','rbf'],default='knn',
        help='Kernel for semi supervised classification.default= knn')

    if not oneline:
        result = mainparser.parse_args()
    else:
        result = mainparser.parse_args(oneline)

    # result = mainparser.parse_args()
    if result.which not in allcommands:
        mainparser.print_help()
        exit()
    else:
        return result

def main():
    """Main program."""
    sns.set()
    warnings.filterwarnings("ignore")

    def process_commands():
        """Command line processing."""
        print(cmdl.which)

        if cmdl.which == 'sattrib':
            process_sattrib(cmdl.gridfileslist,
                gridcols=cmdl.gridcols,
                gridheader=cmdl.gridheader,
                ilxl=cmdl.ilxl,
                outdir=cmdl.outdir)

        elif cmdl.which == 'dropcols':
            process_dropcols(cmdl.csvfile,
                cmdlcols2drop=cmdl.cols2drop,
                cmdloutdir=cmdl.outdir)

        elif cmdl.which == 'prepfile':
            process_prepfile(cmdl.csvfile,cmdl.outdir)

        elif cmdl.which == 'listcsvcols':
            process_listcsvcols(cmdl.csvfile)

        elif cmdl.which == 'sscalecols':
            process_sscalecols(cmdl.csvfile,
                cmdlxyzcols=cmdl.xyzcols,
                cmdlincludexyz=cmdl.includexyz,
                cmdlkind=cmdl.kind,
                cmdloutdir=cmdl.outdir)
            # includexyz is an option to use xyz columns as predictors. default is to drop them
            # they are appended to the data file before scaling.
        elif cmdl.which == 'wscalecols':
            process_wscalecols(cmdl.csvfile,
                cmdlwxyzcols=cmdl.wxyzcols,
                cmdloutdir=cmdl.outdir)

        elif cmdl.which == 'wscaletarget':
            process_wscaletarget(cmdl.csvfile,
                cmdlkind=cmdl.kind,
                cmdltargetcol=cmdl.targetcol,
                cmdloutdir=cmdl.outdir)

        elif cmdl.which == 'wattrib':
            process_wellattrib(cmdl.wattribfile,cmdl.wxyfile,
                cmdlwattribcols=cmdl.wattribcols,
                cmdlwxyzcols=cmdl.wxyzcols,
                cmdlfillna=cmdl.fillna,
                cmdloutdir=cmdl.outdir)

        elif cmdl.which == 'wamerge':
            process_wamerge(cmdlcsvfileslist=cmdl.csvfileslist,
                cmdlcsvcols=cmdl.csvcols,
                cmdlcsvskiprows=cmdl.csvskiprows,
                cmdloutdir=cmdl.outdir)

        elif cmdl.which == 'seiswellattrib':
            process_seiswellattrib(cmdl.seiscsv,cmdl.wellcsv,
                cmdlwellcsvcols=cmdl.wellcsvcols,
                cmdlradius=cmdl.radius,
                cmdlinterpolate=cmdl.interpolate,
                cmdloutdir=cmdl.outdir)

        elif cmdl.which == 'PCAanalysis':
            process_PCAanalysis(cmdl.allattribcsv,
                cmdlacolsrange=cmdl.acolsrange,
                cmdlanalysiscols=cmdl.analysiscols,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'PCAfilter':
            process_PCAfilter(cmdl.allattribcsv,
                cmdlacolsrange=cmdl.acolsrange,
                cmdlanalysiscols=cmdl.analysiscols,
                cmdltargetcol=cmdl.targetcol,
                cmdlncomponents=cmdl.ncomponents,
                cmdloutdir=cmdl.outdir,
                cmdlcols2addback=cmdl.cols2addback)

        elif cmdl.which == 'scattermatrix':
            process_scattermatrix(cmdl.allattribcsv,
                cmdlwellxyzcols=cmdl.wellxyzcols,
                cmdlsample=cmdl.sample)

        elif cmdl.which == 'EDA':
            process_eda(cmdl.allattribcsv,
                cmdlxyzcols=cmdl.xyzcols,
                cmdlpolydeg=cmdl.polydeg,
                cmdlsample=cmdl.sample,
                cmdlhideplot=cmdl.hideplot,
                cmdlplotoption=cmdl.plotoption,
                cmdloutdir=cmdl.outdir)

        elif cmdl.which == 'linreg':
            process_linreg(cmdl.allattribcsv,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwellxyzcols=cmdl.wellxyzcols,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdloutdir=cmdl.outdir)

        elif cmdl.which == 'featureranking':
            process_featureranking(cmdl.allattribcsv,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdltestfeatures=cmdl.testfeatures,
                cmdllassoalpha=cmdl.lassoalpha,
                cmdlfeatures2keep=cmdl.features2keep,
                cmdlcv=cmdl.cv,
                cmdltraintestsplit=cmdl.traintestsplit)

        elif cmdl.which == 'linfitpredict':
            process_linfitpredict(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'KNNtest':
            process_KNNtest(cmdl.allattribcsv,
                cmdlsample=cmdl.sample,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlcv=cmdl.cv,
                cmdlhideplot=cmdl.hideplot,
                cmdloutdir=cmdl.outdir)

        elif cmdl.which == 'KNNfitpredict':
            process_KNNfitpredict(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlkneighbors=cmdl.kneighbors,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'TuneCatBoostRegressor':
            process_TuneCatBoostRegressor(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdloutdir=cmdl.outdir,
                cmdliterations=cmdl.iterations,
                cmdllearningrate=cmdl.learningrate,
                cmdldepth=cmdl.depth,
                cmdlcv=cmdl.cv,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'CatBoostRegressor':
            process_CatBoostRegressor(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdloutdir=cmdl.outdir,
                cmdliterations=cmdl.iterations,
                cmdllearningrate=cmdl.learningrate,
                cmdlscaleminmaxvalues=cmdl.scaleminmaxvalues,
                cmdldepth=cmdl.depth,
                cmdlcv=cmdl.cv,
                cmdlfeatureimportance=cmdl.featureimportance,
                cmdlhideplot=cmdl.hideplot,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlvalsize=cmdl.valsize,
                cmdlnofilesout=cmdl.nofilesout,
                cmdlradius=cmdl.radius,
                cmdloverfittingdetection=cmdl.overfittingdetection,
                cmdlodpval=cmdl.odpval)

        elif cmdl.which == 'ANNRegressor':
            process_ANNRegressor(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdloutdir=cmdl.outdir,
                cmdlnodes=cmdl.nodes,
                cmdlactivation=cmdl.activation,
                cmdlepochs=cmdl.epochs,
                cmdlbatch=cmdl.batch,
                cmdlscaleminmaxvalues=cmdl.scaleminmaxvalues,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlcv=cmdl.cv,
                cmdlhideplot=cmdl.hideplot,
                cmdlvalsize=cmdl.valsize,
                cmdlradius=cmdl.radius)

        elif cmdl.which == 'NuSVR':
            process_NuSVR(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdloutdir=cmdl.outdir,
                cmdlerrpenalty=cmdl.errpenalty,
                cmdlnu=cmdl.nu,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlscaleminmaxvalues=cmdl.scaleminmaxvalues,
                cmdlcv=cmdl.cv,
                cmdlhideplot=cmdl.hideplot,
                cmdlvalsize=cmdl.valsize,
                cmdlradius=cmdl.radius)

        elif cmdl.which == 'SGDR':
            process_SGDR(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlminmaxscale=cmdl.minmaxscale,
                cmdloutdir=cmdl.outdir,
                cmdlloss=cmdl.loss,
                cmdlpenalty=cmdl.penalty,
                cmdll1ratio=cmdl.l1ratio,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlscaleminmaxvalues=cmdl.scaleminmaxvalues,
                cmdlcv=cmdl.cv,
                cmdlhideplot=cmdl.hideplot,
                cmdlvalsize=cmdl.valsize,
                cmdlradius=cmdl.radius)

        elif cmdl.which == 'CatBoostClassifier':
            process_CatBoostClassifier(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdloutdir=cmdl.outdir,
                cmdliterations=cmdl.iterations,
                cmdllearningrate=cmdl.learningrate,
                cmdlcoded=cmdl.coded,
                cmdldepth=cmdl.depth,
                cmdlqcut=cmdl.qcut,
                cmdlcv=cmdl.cv,
                cmdlvalsize=cmdl.valsize,
                cmdlfeatureimportance=cmdl.featureimportance,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlbalancetype=cmdl.balancetype,
                cmdlnneighbors=cmdl.nneighbors,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'TuneCatBoostClassifier':
            process_TuneCatBoostClassifier(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlcoded=cmdl.coded,
                cmdloutdir=cmdl.outdir,
                cmdliterations=cmdl.iterations,
                cmdllearningrate=cmdl.learningrate,
                cmdldepth=cmdl.depth,
                cmdlqcut=cmdl.qcut,
                cmdlcv=cmdl.cv,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'testCmodels':
            process_testCmodels(cmdl.wellattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwanalysiscols=cmdl.wanalysiscols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlqcut=cmdl.qcut,
                cmdlcv=cmdl.cv,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'logisticreg':
            process_logisticreg(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwanalysiscols=cmdl.wanalysiscols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlqcut=cmdl.qcut,cmdlcoded=cmdl.coded,
                cmdloutdir=cmdl.outdir,
                cmdlcv=cmdl.cv,
                cmdlvalsize=cmdl.valsize,
                cmdlhideplot=cmdl.hideplot,
                cmdlclassweight=cmdl.classweight,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlbalancetype=cmdl.balancetype,
                cmdlnneighbors=mdl.nneighbors)

        elif cmdl.which == 'GaussianNaiveBayes':
            process_GaussianNaiveBayes(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwanalysiscols=cmdl.wanalysiscols,
                cmdlqcut=cmdl.qcut,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlcoded=cmdl.coded,
                cmdloutdir=cmdl.outdir,
                cmdlcv=cmdl.cv,
                cmdlvalsize=cmdl.valsize,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlbalancetype=cmdl.balancetype,
                cmdlnneighbors=cmdl.nneighbors,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'QuadraticDiscriminantAnalysis':
            process_QuadraticDiscriminantAnalysis(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwanalysiscols=cmdl.wanalysiscols,
                cmdlqcut=cmdl.qcut,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlcoded=cmdl.coded,
                cmdloutdir=cmdl.outdir,
                cmdlcv=cmdl.cv,
                cmdlvalsize=cmdl.valsize,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlbalancetype=cmdl.balancetype,
                cmdlnneighbors=cmdl.nneighbors,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'NuSVC':
            process_NuSVC(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwanalysiscols=cmdl.wanalysiscols,
                cmdlqcut=cmdl.qcut,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlcoded=cmdl.coded,
                cmdloutdir=cmdl.outdir,
                cmdlcv=cmdl.cv,
                cmdlvalsize=cmdl.valsize,
                cmdlnu=cmdl.nu,
                cmdlbalancetype=cmdl.balancetype,
                cmdlnneighbors=cmdl.nneighbors,
                cmdlgeneratesamples=cmdl.generatesamples,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'GaussianMixtureModel':
            process_GaussianMixtureModel(cmdl.wellattribcsv,
                cmdlseisattribcsv=cmdl.seisattribcsv,
                cmdlbayesian=cmdl.bayesian,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwanalysiscols=cmdl.wanalysiscols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlsaxyzcols=cmdl.saxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlcatcol=cmdl.catcol,
                cmdlncomponents=cmdl.ncomponents,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'clustertest':
            process_clustertest(cmdl.allattribcsv,
                cmdlcolsrange=cmdl.colsrange,
                cmdlcols2cluster=cmdl.cols2cluster,
                cmdlsample=cmdl.sample,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'clustering':
            process_clustering(cmdl.allattribcsv,
                cmdlcolsrange=cmdl.colsrange,
                cmdlcols2cluster=cmdl.cols2cluster,
                cmdlnclusters=cmdl.nclusters,
                cmdlplotsilhouette=cmdl.plotsilhouette,
                cmdlsample=cmdl.sample,
                cmdlxyzcols=cmdl.xyzcols,
                cmdladdclass=cmdl.addclass,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'DBSCAN':
            process_dbscan(cmdl.allattribcsv,
                cmdlcolsrange=cmdl.colsrange,
                cmdlcols2cluster=cmdl.cols2cluster,
                cmdlxyzcols=cmdl.xyzcols,
                cmdlminsamples=cmdl.minsamples,
                cmdladdclass=cmdl.addclass,
                cmdleps=cmdl.eps,
                cmdloutdir=cmdl.outdir)

        elif cmdl.which == 'tSNE':
            process_tSNE(cmdl.allattribcsv,
                cmdlcolsrange=cmdl.colsrange,
                cmdlcols2cluster=cmdl.cols2cluster,
                cmdlsample=cmdl.sample,
                cmdlxyzcols=cmdl.xyzcols,
                cmdllearningrate=cmdl.learningrate,
                cmdlscalefeatures=cmdl.scalefeatures,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'tSNE2':
            process_tSNE2(cmdl.wellattribcsv,cmdl.seisattribcsv,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwpredictorcols=cmdl.wpredictorcols,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlwxyzcols=cmdl.wxyzcols,
                cmdlscolsrange=cmdl.scolsrange,
                cmdlspredictorcols=cmdl.spredictorcols,
                cmdlsxyzcols=cmdl.sxyzcols,
                cmdlsample=cmdl.sample,
                cmdllearningrate=cmdl.learningrate,
                cmdlscalefeatures=cmdl.scalefeatures,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'umap':
            process_umap(cmdl.allattribcsv,
                cmdlcolsrange=cmdl.colsrange,
                cmdlcols2cluster=cmdl.cols2cluster,
                cmdlsample=cmdl.sample,
                cmdlxyzcols=cmdl.xyzcols,
                cmdlnneighbors=cmdl.nneighbors,
                cmdlmindistance=cmdl.mindistance,
                cmdlncomponents=cmdl.ncomponents,
                cmdlscalefeatures=cmdl.scalefeatures,
                cmdloutdir=cmdl.outdir,
                cmdlhideplot=cmdl.hideplot)

        elif cmdl.which == 'semisupervised':
            process_semisupervised(cmdl.wellattribcsv,
                cmdl.seisattribcsv,
                cmdlwtargetcol=cmdl.wtargetcol,
                cmdlwcolsrange=cmdl.wcolsrange,
                cmdlwellsxyzcols=cmdl.wellsxyzcols,
                cmdlsample=cmdl.sample,
                cmdlnneighbors=cmdl.nneighbors,
                cmdlkernel=cmdl.kernel,
                cmdlcol2drop=cmdl.col2drop,
                cmdloutdir=cmdl.outdir)

    # print(__doc__)
    cmdl = getcommandline()
    if cmdl.which == 'workflow':
        lnum = 0
        startline = cmdl.startline
        with open(cmdl.commandfile,'r') as cmdlfile:
            for line in cmdlfile:
                lnum += 1
                print()
                print('%00d:>' % lnum,line)
                if lnum >= startline:
                    parsedline = shlex.split(line)[2:]
                    # if len(parsedline) >=1:
                    if len(parsedline) >= 1 and not cmnt(line):
                        cmdl = getcommandline(*parsedline)
                        process_commands()
                else:
                    print('Skip line:%00d' % lnum,line)
    else:
        process_commands()








if __name__=='__main__':
	main()
