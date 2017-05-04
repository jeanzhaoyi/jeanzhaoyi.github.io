---
layout: post
title: "Kaggle_predict_Santander_customer_satisfaction"
date: 2017-5-3
---
# Predict Santander Customer Satisfaction
***

## I. Background 
***
<div class = "span5 alert alert-info">
 
<p>Santander is a retail and commercial bank founded in 1856 with a headquarter in the U.K. They have 125 million customers worldwide. Customer satisfaction is a key measure of their success and is a strong indicator to future customer retention. Identifying unhappy customers, finding ways to improve their satisfaction will prevent them from leaving Santander. Yet, unhappy customers rarely voice their dissatisfaction before leaving.Thus, it is crucial for Santander to predict and identify dissatisfied customers early in their relationship in order to take proactive steps to improve a customer's happiness before it's too late.</p>
<p> In a Kaggle competition, we are provided with 370 of anonymized features to predict if a customer is satisfied or dissatisfied with his or her banking experiences.</p>
<p>A major challenge in solving this prediction problem is that no prior information was provided about those features. We do not know which variables were numerical and which were categorical, not to say the meaning behind the numbers.</p>
</div>

## II. Data and Description of Features
***
<div class = "span5 alert alert-info">
There are in total 76020 observations and 370 predictive features. There is no information on the meaning and type of those features. It is assumed that are the observations are independent with each other and each represent a customer.
</div>



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load in the data set
train_raw = pd.read_csv('train.csv')
print(train_raw.shape)
print(train_raw.columns)
```

    (76020, 371)
    Index(['ID', 'var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1',
           'imp_op_var39_comer_ult3', 'imp_op_var40_comer_ult1',
           'imp_op_var40_comer_ult3', 'imp_op_var40_efect_ult1',
           'imp_op_var40_efect_ult3',
           ...
           'saldo_medio_var33_hace2', 'saldo_medio_var33_hace3',
           'saldo_medio_var33_ult1', 'saldo_medio_var33_ult3',
           'saldo_medio_var44_hace2', 'saldo_medio_var44_hace3',
           'saldo_medio_var44_ult1', 'saldo_medio_var44_ult3', 'var38', 'TARGET'],
          dtype='object', length=371)


<div class = "span alert alert-info">
<p>The "TARGET" column is the variable to predict. </p>
    <ul>
        <li>It equals 1 for unsatisfied customers and 0 for satisfied customers. </li>
        <li>There are 3008 out of 76020 dissatisfied customers, or only 4%.</li>
    </ul>
</div>


```python
train_raw.groupby('TARGET').size()

```




    TARGET
    0    73012
    1     3008
    dtype: int64



<div class = "span5 alert alert-info">
<p>Some of them have hidden NA values, such as "-999999" as a minimum value of var3, and “99” as a maximum value of a variable with all other vales being 0, 1, 2, 3, 4,5. </p>
<p>Since there are over 30K of observations with missing values in those two variables, and there is no information to help filling in missing value, either all those observations or those two predictor variables need to be removed.</p>
</div>



<div class = "span5 alert alert-info">
Categorical variables usually have a much smaller number of unique values than numerical variables.
Without prior information on which variables were categorical, there needs to be an arbitratry threhold set to distinguish categorical from numerical variables.
</div>

### Test sensitivity to definition of categorical variabes
***
<div class = "span5 alert alert-info">
Here we define a function to transform the original dataset into one with dummy variables of arbitrarily defined categorical variables. The threhold for a minimum number of unique values is a parameter input 'N'.  
</div>


```python
def transf_dat(muted, N):
    muted = muted.drop(['var3','var36', 'TARGET'], axis = 1)
    # remove variables with missing values and the target column
    muted = muted.set_index(["ID"]) 
    for field in list(muted.columns):
    
        n_uniq = np.unique(muted[field]).size # get the number of unique values
        if n_uniq == 1:
            muted = muted.drop(field,axis = 1) # drop fields with only 1 unique value,m,,,,,,

        if n_uniq > 2 and n_uniq <=N:
            # consider the column as a categorical variable
            muted = pd.concat([muted, 
                               pd.get_dummies(muted[field], prefix=field).iloc[:,:-1]], 
                              axis=1) # keep one less column of dummy variables 
            muted = muted.drop(field, axis=1)
    return muted
```


```python
# assume 20 to be the threshold
muted = transf_dat(train_raw, 20)

print(muted.shape)

```

    (76020, 848)


<div class = "span5 alert alert-info">
<p>After taking all feature transformation, there were 76020 observations and 848 predictive features used when 20 is the threshold of the a minimum number of unique values to be considered a numerical varaible.</p>

<p>Before running any models, first set up a stratified kfold cross-validation to evaluate classifiers systematically. </p>
</div>

### Set up functions for cross-validation
***
<div class = "span5 alert alert-info">
The reason that a stratifiation is needed is because only 4% of the dataset are dissatisfied customers, and we need to make sure that there is a balanced representation of satisfied and dissatisfied customers in each fold of te splits. We use a 5 fold cross validation, so the testing set is one fifth the size of the dataset, or 19004 observations.
</div>


```python
seed = 1
# use stratified kfold to ensure that there are 1 and 0 in each fold
kfold = cross_validation.StratifiedKFold(train_raw.TARGET, 
                                         n_folds=5,  # five fold cross validation
                                         random_state=seed) 

```

<div class = "span5 alert alert-info">
<p>We have a binary classification problem. A basic evaluation of a classifier is the accuracy, or the ratio of correct predictions over the total number of predictions. The problem with this metric is that it doesn't provide any information on the number of correctly identified dissatisfied customers. </p>
<p> A better metric is to have information on the True Positive Rate (TPR), False Negative Rate (FNR), precision and recall (same as TPR). ROC curve is a plot of TPR against FPR, and the higher the area under this curve (AUC), the better the classifier is. Also, a higher precision and a higher recall is indicative of a better classifier. </p>
<p>Here we use the area under ROC curve as the metric in the cross validation function.</p>
<p>In addition to returning the mean AUC of all folds in the splits, an ROC curve is drawn if the "show_plot" parameter in this function is set to true.</p>

</div>


```python
from sklearn.metrics import roc_curve
from scipy import interp

def cv_score(k, X, y, classifier, show_plot = False): # X is a data frame, y is a list
    kfold = model_selection.StratifiedKFold(n_splits= k)
    scores = 0.
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    # colors to plot
    colors = ['cyan',  'seagreen', 'yellow', 'blue', 'darkorange']
    lw = 2

    i = 0 # index the fold
    for train, test in kfold.split(X, y):
        probas_t = classifier.fit(X.iloc[train,:], y[train]).predict_proba(X.iloc[test,:])
        sc = metrics.roc_auc_score(y[test], probas_t[:,1])
        scores += sc
        
        fpr, tpr, thresholds = roc_curve(y[test], probas_t[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr) # mean true positive rate to plot
        mean_tpr[0] = 0.0
        
        plt.plot(fpr, tpr, lw=lw, color=colors[i],
                 label='ROC fold %d (area = %0.2f)' % (i+1, sc)) # fold i and area under the curve

        i += 1
        
    # plot all fold's ROC curve
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    mean_tpr /= kfold.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = scores/k
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.legend(loc="lower right")
    
    if show_plot == True:
        plt.show()
        
    return mean_auc

```

## III. Apply Cross validated Machine Learning
***
<div class = "span5 alert alert-info">
First load in all the libraries used for both model seletion, classifier itself, and model evaluation. 
</div>


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
from sklearn.naive_bayes import BernoulliNB

from sklearn.pipeline import Pipeline
from pandas_ml import ConfusionMatrix

```

### 1. Construct a benchmark null model
***
<div class = "span5 alert alert-info">
<p>A statstical benchmark is assuming no relationship between our target predicted feature and all the predictive features and randomly guessing whether a customer was satisifed or not. We would expect a 50% chance of guessing correclty.</p>
<p>To generate a randomized dataset, all the predicted values are kept the same, but all the features are resampled without replacement, so there shall be no relationship between the predictive features and the predicted values. </p>
</div>


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
ConfusionMatrix(y_true = y_testlr, y_pred = pred_test)

```




    Predicted      0  1  __all__
    Actual                      
    0          18300  0    18300
    1            705  0      705
    __all__    19005  0    19005



* According to this confusion matrix, no observation is predicted to be a dissatisfied customer.


```python
# use cross validation to find an average value of the ROC AUC
results = cv_score(5,  randomized, y, LogisticRegression(C=0.1))
print('Null model no transformation Logistic Regression\n mean ROC AUC score: \n',
      results)

```

    Null model no transformation Logistic Regression
     mean ROC AUC score: 
     0.49807360146


<div class = "span5 alert alert-info">
<p>The average of AUC from running a 5-fold cross validation is 0.50. This would be expected, as in a random scenario, the probability of guessing the head or tail correctly from flipping a coin would be 50%. </p>
<p> In comparison, our actual model shall have a higher than 0.5 AUC. </p>
</div>


### 2. Logistic Rression on actual data set
***
<div class = "span5 alert alert-info">
Logistic regression is a most common method for binary classification that predicts the probability of an observation being 1. In this case, we are predicting the probability of a customer being dissatisfied. 
</div>


```python
print('Logistic Regression\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, LogisticRegression(C=0.1))
)

```

    Logistic Regression
     mean ROC AUC score: 
     0.599080519934


<div class = "span5 alert alert-info">
    <li>The result above is better than the null model</li>
    <li> Now add a MinMaxScaler to improve the model</li>
</div>


```python
pipe_LG_scaled = Pipeline([
    ('minmax', MinMaxScaler()),
    ('LG', LogisticRegression(C=0.1))  
])
print('MinMax scalled Logistic Regression\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, pipe_LG_scaled, True)
      )

```


![alt img](https://github.com/jeanzhaoyi/jeanzhaoyi.github.io/blob/master/images/output_27_0.png)


    MinMax scalled Logistic Regression
     mean ROC AUC score: 
     0.786704261508



```python
# run one individual logsitic regression to see the model performance in recall
X = MinMaxScaler().fit_transform(less_ft)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= seed)

LG = LogisticRegression(C=0.1)
LG.fit(X_train, y_train)

pred_train = LG.predict(X_train)
pred_test = LG.predict(X_test)

prob_test = LG.predict_proba(X_test)
print("Scaled Logistic regression AUC score: "
    ,metrics.roc_auc_score(y_true = y_test,  y_score = prob_test[:,1]))

ConfusionMatrix(y_true = y_test, y_pred = pred_test)
```

    Scaled Logistic regression AUC score:  0.777781223847

    Predicted     0  1  __all__
    Actual               
    0          18256  2    18258
    1            747  0      747
    __all__    19003  2    19005




```python
cut_off = np.arange(0.01,0.9,0.01)

recall_test = [metrics.recall_score(y_pred= prob_test[:,1]>x, 
                            y_true = y_test) for x in cut_off]
precision_test = [metrics.precision_score(y_pred= prob_test[:,1]>x, 
                            y_true = y_test) for x in cut_off]
f1_test = [metrics.f1_score(y_pred= prob_test[:,1]>x, 
                            y_true = y_test) for x in cut_off]

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

ax1.plot(cut_off, recall_test, label = 'Recall', color = 'purple')
ax1.plot(cut_off, precision_test, color = 'green', label = 'Precision')
ax1.plot(cut_off, f1_test, color = 'blue', label = 'f1-score')
ax1.legend(loc='upper right', shadow=True)
ax1.set_title('Logistic Regression')
ax1.set_xlabel('Cut off probability for being 1 (dissatisfied customer)')
ax1.set_ylabel('Recall and precision score for test set')

ax2.set_xlim(left= 0.01, right = 0.9)
ax2.set_ylim(bottom=0, top= 1)
ax2.plot( recall_test, precision_test)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
plt.savefig('Precion_recall_LR.jpeg')
plt.show()

```


![alt curves](https://github.com/jeanzhaoyi/jeanzhaoyi.github.io/blob/master/images/output_29_1.png)



```python
ConfusionMatrix(y_true = y_test, y_pred = prob_test[:,1]>0.08)

```




    Predicted  False  True  __all__
    Actual                     
    False      16255  2003    18258
    True         368   379      747
    __all__    16623  2382    19005



<div class = "span5 alert alert-info">
* The MinMaxScale significantly improved the model
<li> Try Chi2 feature selection to see if there is further improvment </li>
</div>


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


<div class = "span5 alert alert-info">
There was not any improvement. Actually, the AUC score was even lower. So there is no need to apply a chi-squared feature selection.
</div>

### 2.Test the sensitivity of results to the threshold of unique values for categorical versus numerical variables
***


```python
raw = train_raw.drop(['var3','var36', 'TARGET'], axis = 1)
print('Raw training data\n scalled Logistic Regression\n mean ROC AUC score: \n',
      cv_score(5,  raw, y, pipe_LG_scaled)
      )
```

    /Users/yizhao/anaconda3/lib/python3.5/site-packages/sklearn/linear_model/base.py:352: RuntimeWarning: overflow encountered in exp
      np.exp(prob, prob)


    Raw training data
     scalled Logistic Regression
     mean ROC AUC score: 
     0.783026687606



```python
uniq_10 = transf_dat(train_raw, 10)

```


```python
uniq_10.shape

```




    (76020, 631)




```python
print('Categorical variables defined as having less than 10 unique values\n scalled Logistic Regression\n mean ROC AUC score: \n',
      cv_score(5,  uniq_10, y, pipe_LG_scaled)
      )
```

    Categorical variables defined as having less than 10 unique values
     scalled Logistic Regression
     mean ROC AUC score: 
     0.784806125663



```python
uniq_30 = transf_dat(train_raw, 30)
uniq_30.shape

```




    (76020, 1156)




```python
print('Categorical variables defined as having less than 30 unique values\n scalled Logistic Regression\n mean ROC AUC score: \n',
      cv_score(5,  uniq_30, y, pipe_LG_scaled)
      )
```

    Categorical variables defined as having less than 30 unique values
     scalled Logistic Regression
     mean ROC AUC score: 
     0.785973809131


<div class = "span5 alert alert-info">
<li>It seems that there difference among choices of categorical variables did not make a significant difference</li>
<li>Even if using the raw dataset without distinguishing any categorical versus not or removing features with only one unique value, the resulting auc score is lower by 0.003 compared to transforming an arbitrarily defined set of categorical variables as dummy variables. </li>
</div>

### 3. Naive Bayes
***
<div class = "span5 alert alert-info">
<li>Apply the same cross-validation function on Bernouli Naive Bayes. We use Bernoulli as opposed to Gaussian because the dataset is highly sparse, and most of the predictor variables do not exhibit a Gaussian distribution. On the other hand, there are a large number of categorical variables organized in binary forms.</li>
<li>First apply Naive Bayes classifier on the dataset without scalling, and then add the MinMaxScaler, and finally also attempt a chi-squared feature selection step.</li>
</div>


```python
print('Naive Bayes\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, BernoulliNB()))
```

    Naive Bayes
     mean ROC AUC score: 
     0.712689622542



```python
pipe_NB = Pipeline([
    ('minmax', MinMaxScaler()),
    ('nb', BernoulliNB())  
])

print('MinMax scalled Naive Bayes\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, pipe_NB))

```

    MinMax scalled Naive Bayes
     mean ROC AUC score: 
     0.703902264811



```python
pipe_NB_chi2 = Pipeline([
    ('minmax', MinMaxScaler()),
    ('kBest', SelectKBest(chi2, k = 100)),
    ('nb', BernoulliNB())  
])

print('MinMax scaled chi2 top 100 features\n Bernouli Naive Bayes\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, pipe_NB_chi2, True))

```


![alt text](https://github.com/jeanzhaoyi/jeanzhaoyi.github.io/blob/master/images/output_45_0.png)


    MinMax scalled chi2 top 100 features
     Bernouli Naive Bayes
     mean ROC AUC score: 
     0.713750329496


<div class = "span5 alert alert-info">
<li>It seems that the best performing model among Naive Bayes models was combining MinMaxScaler, and chi-squared feature selection.</li>
<li>Still the Logistic Regression classifier has a better performance.</li>
</div>

### 4. Decesion tree based classifiers
***
<div class = "span5 alert alert-info">
<li> Both Logistic Regression and Naive Bayes have only one decision boundary.</li>
<li>Decision tree based ensemble models are different from Logistic Regression and Naive Bayes in that each tree partitions the feature space into half-spaces with each split and resulting in multiple linear decision boundaries. </li>


<p>In the following, Random Forest classifier, Gradient Boost Classifier, and XGBoost classifier are presented sequentially. </p>
</div>

#### a. Random Forest


```python
# simple random forest classifier
print('Random Forest on original dataset\n mean ROC AUC score: \n',
      cv_score(5,  raw, y, RandomForestClassifier()))

print('\nRandom Forest \n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, RandomForestClassifier()))


print('\nMinMax scalled RandomForestClassifier\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, Pipeline([
    ('minmax', MinMaxScaler()),
    ('RF', RandomForestClassifier())  
])))
```

    Random Forest on original dataset
     mean ROC AUC score: 
     0.693697681224
    
    Random Forest 
     mean ROC AUC score: 
     0.686270439471
    
    MinMax scalled RandomForestClassifier
     mean ROC AUC score: 
     0.6700023768


<div class = "span5 alert alert-info">
The default Random Forest classifier performed significantly worse than logistic regression and Naive Bayes classifiers.
</div>

#### b. Gradient Boosting Classifier


```python
# gradient boosting without any scaling
print('GradientBoostingClassifier\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, GradientBoostingClassifier()  ))
```

    GradientBoostingClassifier
     mean ROC AUC score: 
     0.707529186673



```python
# Gradient Boosting classifier
pipe_boost = Pipeline([
    ('minmax', MinMaxScaler()),
    ('G', GradientBoostingClassifier())  
])

print('MinMax scalled GradientBoostingClassifier\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, pipe_boost))

```

    MinMax scalled Logistic Regression
     mean ROC AUC score: 
     0.773383059245


<div class = "span5 alert alert-info">
<li>The results are better than the Random Forest classifier, but still worse than a simple logistic regression</li>
<li>This model's default has max_depth = 3. Try some other numbers to see if the results improve</li>
</div>


```python

print('MinMax scalled Gradient Boosting Classifier (max_depth = 2)\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, 
              Pipeline([
                    ('minmax', MinMaxScaler()),
                    ('G', GradientBoostingClassifier(max_depth = 2))  
                ])
              ))
```

    MinMax scalled Gradient Boosting Classifier (max_depth = 2)
     mean ROC AUC score: 
     0.726656727175



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
                    ('G', GradientBoostingClassifier(max_depth = 5))  
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
                    ('G', GradientBoostingClassifier(max_depth = 10))  
                ])
              ))
```

    MinMax scalled Gradient Boosting Classifier (max_depth = 10)
     mean ROC AUC score: 
     0.728726317078



<div class = "span5 alert alert-info">
<li>smaller max_depth led to a worse result</li>
<li>Increasing the number of maximum depth in boosted regression trees did not lead to better result.</li>
</div>

#### c. XGBoost Classifier


```python
print('XGBoost\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, XGBClassifier(max_depth = 3), True))
```


![alt text](https://github.com/jeanzhaoyi/jeanzhaoyi.github.io/blob/master/images/output_56_0.png)


    XGBoost
     mean ROC AUC score: 
     0.831932077938



```python
pipe_xgb = Pipeline([
    ('minmax', MinMaxScaler()),
    ('xgb', XGBClassifier(max_depth = 3))  
])

print('MinMax scalled XGBoost\n mean ROC AUC score: \n',
      cv_score(5,  less_ft, y, pipe_xgb))
```

    MinMax scalled XGBoost
     mean ROC AUC score: 
     0.831932077938



<div class = "span5 alert alert-info">
<li> This model appears to be the best performing </li>
<li>Look at one run's result, especially the confusion matrix and recall, precision, f1-score</li>
</div>


```python
X = MinMaxScaler().fit_transform(less_ft)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= seed)

xgb = XGBClassifier(max_depth= 3)
xgb.fit(X_train, y_train)

pred_train = xgb.predict(X_train)
pred_test = xgb.predict(X_test)

prob_test = xgb.predict_proba(X_test)
print("AUC score: "
    ,metrics.roc_auc_score(y_true = y_test,  y_score = prob_test[:,1]))

ConfusionMatrix(y_true = y_test, y_pred = pred_test)

```

    AUC score:  0.835076494681





    Predicted  |    0  |1 | __all__
    Actual  | 
    --- | --- | --- | ---                   
    0        |  18257 | 1 |   18258
    1        |    746 | 1   |   747
    __all__  |  19003 | 2  |  19005




```python
cut_off = np.arange(0.01,0.9,0.01)

recall_test = [metrics.recall_score(y_pred= prob_test[:,1]>x, 
                            y_true = y_test) for x in cut_off]
precision_test = [metrics.precision_score(y_pred= prob_test[:,1]>x, 
                            y_true = y_test) for x in cut_off]
f1_test = [metrics.f1_score(y_pred= prob_test[:,1]>x, 
                            y_true = y_test) for x in cut_off]

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

ax1.plot(cut_off, recall_test, label = 'Recall', color = 'purple')
ax1.plot(cut_off, precision_test, color = 'green', label = 'Precision')
ax1.plot(cut_off, f1_test, color = 'blue', label = 'f1-score')
ax1.legend(loc='upper right', shadow=True)
ax1.set_title('XGBoost')
ax1.set_xlabel('Cut off probability for being 1 (dissatisfied customer)')
ax1.set_ylabel('Recall and precision score for test set')

ax2.set_xlim(left= 0.01, right = 0.9)
ax2.set_ylim(bottom=0, top= 1)
ax2.plot( recall_test, precision_test)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
plt.savefig('Precision_recall_XGBoost.jpeg')
plt.show()

```

    /Users/yizhao/anaconda3/lib/python3.5/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/yizhao/anaconda3/lib/python3.5/site-packages/sklearn/metrics/classification.py:1113: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)



![alt text](https://github.com/jeanzhaoyi/jeanzhaoyi.github.io/blob/master/images/output_60_1.png)



```python

print('XGBoost area under precision-recall curve:\n',
      metrics.average_precision_score(y_true= y_test, y_score= prob_test[:,1]))

```

    XGBoost area under precision-recall curve:
     0.186005796767



```python
ConfusionMatrix(y_true = y_test, y_pred = prob_test[:,1]>0.13)

```




    Predicted  False  True  __all__
    Actual                         
    False      16769  1489    18258
    True         390   357      747
    __all__    17159  1846    19005



<div>
Recall always increases with a decreasing probability threshold for predicting as positive.
<ul>
    <li>The number of False Positives is less alarming, because trying to improve service to those that were not going to leave wouldn't hurt. Falsely ignoring the dissatisfied customers, however, would be much more hurtful. </li>
    <li>The recall score at a highest F1-score of this XGBoost model is 0.478, which is slightly lower than the recall of 0.507 in Logistic Regression with an optimal threshold of 0.08. </li>
</ul>

<p>Yet, this XGBoost model does have a higher precision, and a higher f-1 score. For the purpose of reducing False Negatives, however, logistic regression was still slighly better.</p>

<p>The XGBoost classifier used a threshold of 0.13, so observations with a probability of higher than 13% being dissatisfied customers were classified as dissatisfied customers. This probability threshold for logistic regression was 8%. These thresholds were small because there was only 4% of the raw dataset being dissatisfied customers. </p>

<p> Bootstrapping resampling could help to balance the dataset such that there is equal representation of both satisfied and dissatisfied customers in the training set.

</div>

### 5. Bootstrap the training set to ensure balanced positive and negative observations
***


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= seed)

# divid the training set into 0 and 1
y_train_pos = y_train[y_train == 1] # only dissatisfied customers
X_train_pos = X_train[y_train == 1]

y_train_neg = y_train[y_train == 0] # subset the training set to only those with satisfied customers
X_train_neg = X_train[y_train == 0]

reps = round(X_train.shape[0]/2)

pos_i = np.random.choice(X_train_pos.shape[0], reps)
neg_i = np.random.choice(X_train_neg.shape[0], reps)

X_res = np.concatenate([X_train_pos[pos_i] , 
                        X_train_neg[neg_i]])
            # pd.concat([X_train_pos.iloc[pos_i, :], 
         #          X_train_neg.iloc[neg_i, :]], axis = 0)
y_res = np.concatenate([y_train_pos[pos_i] , 
                        y_train_neg[neg_i]])
```


```python
LG = LogisticRegression(C=0.1)
LG.fit(X_res, y_res)

pred_train = LG.predict(X_res)
pred_test = LG.predict(X_test)

prob_test = LG.predict_proba(X_test)
print("Bootstrapped Logistic Regression AUC score: "
    ,metrics.roc_auc_score(y_true = y_test,  y_score = prob_test[:,1]))

ConfusionMatrix(y_true = y_test, y_pred = pred_test)

```

    Bootstrapped Logistic Regression AUC score:  0.784215329203





    Predicted     0     1  __all__
    Actual                     
    0          12271  5987    18258
    1            203  544      747
    __all__    12474  6531   19005




```python
prob_test = LG.predict_proba(X_test)

recall_test = [metrics.recall_score(y_pred= prob_test[:,1]>x, 
                            y_true = y_test) for x in cut_off]
precision_test = [metrics.precision_score(y_pred= prob_test[:,1]>x, 
                            y_true = y_test) for x in cut_off]
f1_test = [metrics.f1_score(y_pred= prob_test[:,1]>x, 
                            y_true = y_test) for x in cut_off]
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

ax1.plot(cut_off, recall_test, label = 'Recall', color = 'purple')
ax1.plot(cut_off, precision_test, color = 'green', label = 'Precision')
ax1.plot(cut_off, f1_test, color = 'blue', label = 'f1-score')
ax1.legend(loc='upper right', shadow=True)
ax1.set_title('Logistic regression with bootstrapped balanced dataset')
ax1.set_xlabel('Cut off probability for being 1 (dissatisfied customer)')
ax1.set_ylabel('Recall and precision score for test set')

ax2.set_xlim(left= 0.01, right = 0.9)
ax2.set_ylim(bottom=0, top= 1)
ax2.plot( recall_test, precision_test)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')

plt.savefig('Precision_recall_LR_bootstraped.jpeg')
plt.show()

```


![alt text](https://github.com/jeanzhaoyi/jeanzhaoyi.github.io/blob/master/images/output_67_0.png)



```python
ConfusionMatrix(y_true = y_test, y_pred = prob_test[:,1]>0.66)

```


   |Predicted  | False |  True | Total  |
   | Actual    |------------------------|  
   | --------- |:-----:|:-----:|-------:|                  
   | False     | 16025 | 2233  | 18258  |
   | True      |   358 |  389  |    747 |
   | Total     | 16383 |2622   |  19005 |
    
<div class = "span5 alert alert-info">
<li> After bootstraping the optimal treshold that maximizes f1-score has increased from 8% to 66%, 
but the highest F-1 score is no different from without any resampling</li>
<li>Precision (0.15) and recall (0.52) were similar to without the bootstrap as well.</li>
Thus, there is no added benefits from resampling the training set.
</div>

## IV. Conclusion
***
<div>
Given an anonymized dataset with no information of observations and predictor variables, transformation of those predictor variables is crucial to building high performance predictive models. In this analysis, variables with less than 20 unique values out of 76K observations were considered to be categorical and transformed into binary formats. 

<p> Overall, a simple logistic regression predicts dissatisfied customers of Santander’s at a reasonable accuracy level, outperforming Naïve Bayes and Random Forest models. The extreme gradient ensemble classifier has a better performance overall model performance in terms of AUC and f1-score, but the recall rate is no better than logistic regression. </p>
<p> For Santander, recall is more important than precision or other measures, because the number of False Positives is less alarming than the number of False Negatives. Falsely identifying satisfied customers as dissatisfied and trying to improve service to those that were not going to leave would not hurt. However, falsely ignoring the dissatisfied customers, or identifying them as satisfied customers, would be much more hurtful. </p>
<p> The higher the probability threshold, the better recall rate becomes, but precision also drops in the meantime. The question is how much it costs Santander to improve satisfaction of customers or to prevent a customer from potentially leaving. If the marginal cost of each customer is low, there would be a higher return to Santander to insure a higher recall rate at the cost of more falsely identified dissatisfied customers to please. However, if the marginal cost is high, the optimal probability threshold for identifying dissatisfied customers would be lower, or a lower recall rate but higher precision. In both scenarios, there need to be an assessment of both the potential monetary gain from preventing each dissatisfied customer from leaving and the expected cost of improving satisfaction of each extra falsely identified dissatisfied customer.</p>
</div>


