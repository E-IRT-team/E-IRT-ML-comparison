"""
----------------------------------------------------------------
Here we run the KNN Classifier, we evaluate it on several datasets
in nested CV, tuning the parameters internally with GridSearchCV
---------------------------------------------------------------
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold#, GroupKFold
from sklearn.model_selection import StratifiedKFold as StratCV

from sklearn.model_selection import GridSearchCV
# import classifier here
from sklearn.neighbors import KNeighborsClassifier
## our metrics are loaded here:
from sklearn.metrics import roc_auc_score,average_precision_score,mean_squared_error


##-----------------------------------------------------------------------------
## root paths
datapath = "C:/Users/konst/Box/ML_EIRT/Experiments/data/"
datapath = "C:/Users/u0135479/Documents/GitHub/E-IRT-ML-comparison/data/"
respath = 'C:/Users/konst/Box/ML_EIRT/Experiments/results/'
respath = "C:/Users/u0135479/Documents/GitHub/E-IRT-ML-comparison/results/"

## function to read all the data in a folder
dnames = os.listdir(datapath)
dnames.remove("latentfeatures4") #to drop when finalised
dnames.remove("latentfeatures30")

##-----------------------------------------------------------------------------
# p_grid for kNN
# empirical setting
p_grid = {'n_neighbors' : [5,10,25,50,75,100]}
##-----------------------------------------------------------------------------
outer_folds = 10
inner_folds = 3
##-----------------------------------------------------------------------------

def load_data(dataset, datapath):
    """ 
    This function accesses the data in a file and stores the 3 matrices.
    X1: row feature set, X2: column feature set, Y: output set
    """ 
    if dataset[:2] == 'RW':
        X1 = pd.read_csv(datapath+str(dataset)+'/X1.csv')
        X2 = pd.read_csv(datapath+str(dataset)+'/X2.csv')
        Y = pd.read_csv(datapath+str(dataset)+'/Y.csv')
    else:
        X1 = pd.read_csv(datapath+str(dataset)+'/X1.txt', sep=" ")
        X2 = pd.read_csv(datapath+str(dataset)+'/X2.txt', sep=" ")
        Y = pd.read_csv(datapath+str(dataset)+'/Y.txt', sep=" ")
    
    X1,X2,Y = X1.values,X2.values,Y.values
    
    return X1,X2,Y

# generating the cartesian product of two feature sets  
def global_repr_model(X1,X2):
    N1,nof1 = X1.shape
    N2,nof2 = X2.shape
    count = 0
    Xs = np.zeros([N1*N2, nof1 + nof2])
    for i in range(N1):
        for j in range(N2):
            Xs[count] = np.hstack((X1[i],X2[j]))
            count += 1
    return Xs


outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=1 )
inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=1 )
strat_cv = StratCV(n_splits=outer_folds, shuffle=True, random_state=1 )

def nested_cv_Classfier(X1,X2,Y):  #inner and outer folds are defined globally
    """
    There are 3 scenario per dataset: -row-wise splitting  (evaluate new studs)
                                      -cols-wise splitting (evaluate new items)
                                      -standard splitting (evaluate new interactions)
    
    they are all considered and nested-cross-validated in the main function
    """
    # grouping are set here for each scenario:
    scenarios = {'rows':outer_cv.split(X1), 'columns':outer_cv.split(X2), 'items':strat_cv.split(np.zeros(N1*N2),Y.ravel())}
    my_results = []     
    for group in scenarios.keys():
        print("    ")
        print("Scenario: ", group)
        
#        inner_cv_scores = np.zeros([outer_folds])
        cv_scores = np.zeros([outer_folds, 3])
        
        KNNClassifier,clname = KNeighborsClassifier(algorithm='kd_tree', n_jobs=-1),"KNN"

        i = 0  # counting progress from completed outer folds
        for train, test in scenarios[group]:  # KFold.split is a generator object, outputs 2 ndarrays
            print('---- outer fold', i+1, 'out of', outer_folds, '-----')
            
            #we can also make a dict out of the following
            if group == "rows":
                Xtotrain = global_repr_model(X1[train],X2)
                Ytotrain = Y[train].ravel()
                Xtotest = global_repr_model(X1[test],X2)
                Ytotest = Y[test].ravel()
            elif group == "columns":
                Xtotrain = global_repr_model(X1,X2[train])
                Ytotrain = Y[:,train].ravel()
                Xtotest = global_repr_model(X1,X2[test])
                Ytotest = Y[:,test].ravel()
            elif group == "items":
                Xtotrain = global_repr_model(X1,X2)[train]
                Ytotrain = Y.ravel()[train]
                Xtotest = global_repr_model(X1,X2)[test]
                Ytotest = Y.ravel()[test]
                
            clf = GridSearchCV(estimator=KNNClassifier, param_grid=p_grid, 
                               cv=inner_cv, scoring='roc_auc',refit=True)  # set iid=False to avoid firing warnings, if it happens
            
            clf.fit(Xtotrain, Ytotrain)
            print("best params on train set:", clf.best_params_)
            
            ypred = clf.predict_proba(Xtotest)[:, 1]
            
            cv_scores[i,0] = roc_auc_score(Ytotest,ypred)
            cv_scores[i,1] = average_precision_score(Ytotest,ypred)
            cv_scores[i,2] = mean_squared_error(Ytotest,ypred)

            i = i + 1

        np.save(respath + dataset + clname + group, cv_scores) # save cvscores
        np.savetxt(respath + dataset + clname + group, cv_scores.round(3)) #save round cv scores in txt 
            
        print("AUROC test score:", cv_scores.mean(0)[0])
        print("AUPR test score:", cv_scores.mean(0)[1])
        print("MSE test score:", cv_scores.mean(0)[2])
        
    print("Analysis complete on the dataset with dimensions", Y.shape)
    my_results.append(cv_scores)
    
    return my_results


for dataset in dnames:
    print("-----------"+dataset+"-----------")
    X1,X2,Y = load_data(dataset, datapath)
    N1,nof1 = X1.shape
    N2,nof2 = X2.shape
    
    my_results = nested_cv_Classfier(X1,X2,Y)
    
