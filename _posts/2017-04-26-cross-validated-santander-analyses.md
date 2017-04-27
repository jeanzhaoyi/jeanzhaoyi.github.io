---
layout: post
title: "Cross validated Santander Analyses"
date: 2017-04-26
---


* Use preprocessed features to train a predictive model
* First construct a function to implement cross validation
* Then set a benchmark for model evaluation with randomized data and logistic regression
* Finally, fit and evaluate logistic regression, bootstrapped balanced datasets, as well as a ensemble method still based on logistic regression


```python
# define predictors and predicted variable
less_ft = muted.dropna(axis=1)
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load in the data set
train_raw = pd.read_csv('train.csv')
muted = pd.read_csv('/Users/yizhao/Documents/Second_capstone/muted_data.csv')
```


```python
# load in libraries for ML
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation #StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import model_selection # StratifiedKFold

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from pandas_ml import ConfusionMatrix
```

### Set up functions for cross-validation


```python
seed = 1
# use stratified kfold to ensure that there are 1 and 0 in each fold
kfold = cross_validation.StratifiedKFold(train_raw.TARGET, 
                                         n_folds=5,  # five fold cross validation
                                         random_state=seed) 
```


```python
def cv_score(k, X, y, classifier): # X is a data frame, y is a list
    kfold = model_selection.StratifiedKFold(n_splits= k)
    scores = 0.
    for train, test in kfold.split(X, y):
        probas_t = classifier.fit(X.iloc[train,:], y[train]).predict_proba(X.iloc[test,:])
        sc = metrics.roc_auc_score(y[test], probas_t[:,1])
        scores += sc
    return scores/k

```

### Construct a benchmark null model


```python
rand_index = np.random.randint(less_ft.shape[0], size = less_ft.shape[0])
randomized = less_ft.iloc[rand_index,:] # randomized data frame
```


```python
y = train_raw[['TARGET']].values.ravel()

Xlr, Xtestlr, y_lr, y_testlr = train_test_split(randomized, 
                                                y,
                                                random_state=5)

scale = MinMaxScaler()
X_train = scale.fit_transform(Xlr) # scale all predictor variables
X_test = scale.fit_transform(Xtestlr)

LG = LogisticRegression(C=0.1)
LG.fit(X_train, y_lr)

pred_train = LG.predict(X_train)
pred_test = LG.predict(X_test)

metrics.roc_auc_score(y_testlr, LG.predict_proba(X_test)[:,1])
```




    0.48363682517536721




```python
# use cross validation to find an average value of the ROC AUC
results = cv_score(5,  randomized, y, LogisticRegression(C=0.1))
print('Null model no transformation Logistic Regression\n mean ROC AUC score: \n',
      results)
```

    Null model no transformation Logistic Regression
     mean ROC AUC score: 
     0.49807360146


### Logistic Rression on actual data set


```python
print('Logistic Regression\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, LogisticRegression(C=0.1))
)
```

    Logistic Regression
     mean ROC AUC score: 
     0.599080519934


* The result above is better than the null model
* Now add a MinMaxScaler to improve the model


```python
pipe_LG_scaled = Pipeline([
    ('minmax', MinMaxScaler()),
    ('LG', LogisticRegression(C=0.1))  
])
print('MinMax scalled Logistic Regression\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, pipe_LG_scaled)
      )
```

    MinMax scalled Logistic Regression
     mean ROC AUC score: 
     0.786704261508


* Now there is a significant improvement on the model
* Try Chi2 feature selection to see if there is further improvment


```python
pipe_LG_chi2 = Pipeline([
    ('minmax', MinMaxScaler()),
    ('kBest', SelectKBest(chi2, k = 100)),
    ('LG', LogisticRegression(C=0.1))  
])

print('MinMax scalled Logistic Regression\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, pipe_LG_chi2))
```

    MinMax scalled Logistic Regression
     mean ROC AUC score: 
     0.778433472633


* There was not any improvement. Actually, the AUC score was even lower. 
* Instead of reducing feature space, try an ensemble method that also uses Logistic Regression as the loss function -- sklearn.ensemble.GradientBoostingClassifier


```python
pipe_boost = Pipeline([
    ('minmax', MinMaxScaler()),
    ('LG', GradientBoostingClassifier())  
])

print('MinMax scalled GradientBoostingClassifier\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, pipe_boost))
```

    MinMax scalled Logistic Regression
     mean ROC AUC score: 
     0.773383059245


* Again, the result was worse than a simple logistic regression
* This model's default has max_depth = 3. Try some other numbers to see if the results improve


```python

print('MinMax scalled Gradient Boosting Classifier (max_depth = 2)\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, 
              Pipeline([
                    ('minmax', MinMaxScaler()),
                    ('LG', GradientBoostingClassifier(max_depth = 2))  
                ])
              ))
```

    MinMax scalled Gradient Boosting Classifier (max_depth = 2)
     mean ROC AUC score: 
     0.726656727175


* A smaller max_depth led to a worse result
* Try larger max_depth, or allow deeper learning in the trees to account for interactions among features


```python
print('MinMax scalled Gradient Boosting Classifier (max_depth = 4)\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, 
              Pipeline([
                    ('minmax', MinMaxScaler()),
                    ('LG', GradientBoostingClassifier(max_depth = 4))  
                ])
              ))
print('MinMax scalled Gradient Boosting Classifier (max_depth = 5)\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, 
              Pipeline([
                    ('minmax', MinMaxScaler()),
                    ('LG', GradientBoostingClassifier(max_depth = 5))  
                ])
              ))
```

    MinMax scalled Gradient Boosting Classifier (max_depth = 4)
     mean ROC AUC score: 
     0.757331178641
    MinMax scalled Gradient Boosting Classifier (max_depth = 5)
     mean ROC AUC score: 
     0.762505814474



```python
print('MinMax scalled Gradient Boosting Classifier (max_depth = 10)\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, 
              Pipeline([
                    ('minmax', MinMaxScaler()),
                    ('LG', GradientBoostingClassifier(max_depth = 10))  
                ])
              ))
```

    MinMax scalled Gradient Boosting Classifier (max_depth = 10)
     mean ROC AUC score: 
     0.728726317078


* Increasing the number of maximum depth in boosted regression trees did not lead to better result.
* We conclude to use the simple logistic regression on MinMaxScaled features


```python

```
