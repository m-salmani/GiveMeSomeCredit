import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc


def data_collection():
    '''
    This function reads labeled training data and test data from the .csv files provided.
    :return: List of features, training data, test data, and the corresponding labels for each of those data sets.
    '''
    train_data = pd.read_csv('/Users/mahsasalmani/Desktop/Kaggle_Project/GiveMeSomeCredit/edited_cs-training.csv')
    test_data = pd.read_csv('/Users/mahsasalmani/Desktop/Kaggle_Project/GiveMeSomeCredit/cs-test.csv')
    test_y_data = pd.read_csv('/Users/mahsasalmani/Desktop/Kaggle_Project/GiveMeSomeCredit/sampleEntry.csv')
    train_data.columns = train_data.columns.values
    test_data.columns = test_data.columns.values
    feature_list = list(train_data.columns.values)
    train_labels = train_data['SeriousDlqin2yrs']
    test_labels = test_y_data['Probability']
    # Removing the first two columns that are not 'features'
    feature_list.remove('Unnamed: 0')
    feature_list.remove('SeriousDlqin2yrs')
    return train_data,test_data,train_labels,test_labels,feature_list


def nan_detector(data, feature_list):
    '''
    This function detects the features that include NaN contents in both training data set and test data set,
    and outputs those features with the number of NaN values in each feature.
    :param data: is the training or test data over which NaN values are searched for.
    :param feature_list: the list of features extracted from the data sheet provided above.
    :return: nan_features, which are the features with NaN values, and nan_cnt that counts the number of NaN values.
    '''
    nan_cnt = {}
    nan_features = []
    for c in feature_list:
        nan_val = 0
        for val in data[c].isnull():
            if val == True:
                nan_val = nan_val + 1
                if nan_val == 1:
                    nan_features.append(c)
        nan_cnt[c] = [nan_val]

    return nan_features, nan_cnt


def nan_rmv_indication(data, nan_features):
    '''
    This function generates a new data frame with all NaN samples removed, 'data_nan'. This new data frame will be used
    later to obtain distribution of the samples.
    The function also creates extra columns/features which are named after the features with NaN values, e.g.,
    'feature name'_nans. Content 'i' of the newly added 'feature name'_nans is 1 if content 'i' of 'feature name' is NaN,
    and it is 0 otherwise. This column indicates the existence of NaN value in the corresponding index which may
    be informative in training the final model.
    :param data: training or test data provided.
    :param nan_features: the list of features that contain NaN values.
    :return: data with extra columns/features that indicate the existence of NaN values,
    and a data frame in which NaN values are removed.
    '''
    data_nan = pd.DataFrame()
    for feature in nan_features:
        data[str(feature) + '_nans'] = pd.isnull(data[[feature]]) * 1
    data_nan = data.dropna(how='any')
    return data, data_nan


def nan_replacement(data, nan_features):
    '''
    This function replaces NaN values of each feature with the median of the values in that feature.
    Considering the distribution of the provided data, which seems to have few numbers of outliers,
    median would be a good choice to be used for NaN values.
    :param data: training or test data provided.
    :param nan_features: the list of features that contain NaN values.
    :return: data with NaN values replaced by median of the contents of corresponding feature.
    '''
    for feature in nan_features:
        data[feature] = data[feature].fillna(data[feature].median())
    return data


def feature_dist_plot(data, data_nan_drop, feature_list, nan_features):
    '''
    This function generates different plots each of which illustrating the distribution of either the contents
    or log scale of the contents of each of the features. For the features with skewed distribution of the contents
    the log scale of the contents can be used.
    (For convenience, the distributions in both cases are provided for each feature.)
    :param data: training or test data provided.
    :param data_nan_drop: training or test data in which NaN values are all dropped (without replacing with any new value)
    :param feature_list: the list of features.
    :param nan_features: the list of features that contain NaN values.
    :return: different plots each of which showing the distribution of the data or log scale of data.

    '''
    data_log = pd.DataFrame()
    data_nan_drop_log = pd.DataFrame()
    for features in feature_list:
        if features in nan_features:
            data_nan_drop_log[features] = np.log(1 + data_nan_drop[features].values)
            sns.distplot(data_nan_drop[features])
            plt.show()
            sns.distplot(data_nan_drop_log[features])
            plt.show()
        else:
            data_log[features] = np.log(1 + data[features].values)
            sns.distplot(data[features])
            plt.show()
            sns.distplot(data_log[features])
            plt.show()


def outlier_detector(data, feature_list):
    for feature in feature_list:
        print(feature)
        data_Q1 = data[feature].quantile(0.25)
        data_Q3 = data[feature].quantile(0.75)
        IQR = data_Q3-data_Q1
        lower_bnd = data_Q1 - 1.5 * IQR
        upper_bnd = data_Q3 + 1.5 * IQR
        outlier_idx = (data[feature]<lower_bnd) | (data[feature]>upper_bnd)
        data.loc[outlier_idx,feature] = data[feature].median()
    data
    return data


def feature_augmentation(train_data,test_data,feature_list):
    '''
    In order to build a more accurate and more robust model, this function generates some additional features
    based on the provided features. The "WeightedPastDue" includes a weighted sum of the number of times with past due.
    The rationale behind this is the fact that a longer past due could have more impact on the probability of delinquency.
    To capture this observation, by considering the lower-bound of each interval as a representative
    for each case, e.g., 30, 60, and 90 days, the weights are chosen to be 1,2, and 3, respectively.
    Another feature is "MonthlyNetIncome" which represents the difference between the income and the amount payed for all
     debts.
    :param train_data:
    :param test_data:
    :param feature_list:
    :return:
    '''
    # 4- Introducing new features according to the existing ones
    train_data['WeightedPastDue'] = (train_data['NumberOfTimes90DaysLate'] + 3 * train_data[
        'NumberOfTime60-89DaysPastDueNotWorse'] + 5 * train_data['NumberOfTime30-59DaysPastDueNotWorse']) / 9
    test_data['WeightedPastDue'] = (test_data['NumberOfTimes90DaysLate'] + 3 * test_data[
        'NumberOfTime60-89DaysPastDueNotWorse'] + 5 * test_data['NumberOfTime30-59DaysPastDueNotWorse']) / 9
    train_data['MonthlyNetIncome'] = train_data['MonthlyIncome'] * (1 - train_data['DebtRatio'])
    test_data['MonthlyNetIncome'] = test_data['MonthlyIncome'] * (1 - train_data['DebtRatio'])

    return train_data, test_data


def GB_Classifier_func(train_X, train_Y, test_X):
    '''
    This function build a classification model based on the Gradient Boosting classifier.
    :param train_X: training data set which is provided.
    :param train_Y: the probability of delinquency corresponding to each index.
    :param test_X: test data set which is provided.
    :return: It returns the predictions on the delinquency probabilities for the test data set.
    '''
    GB_Classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                               min_samples_split=2, min_samples_leaf=1,
                                               min_weight_fraction_leaf=0.0, max_depth=3, init=None,
                                               random_state=None, max_features=None, verbose=1)
    GB_Params = {'loss': ['deviance'], 'n_estimators': (100, 200, 300, 400, 500), 'max_depth': (6, 8,10)}
    GB_Search = RandomizedSearchCV(estimator=GB_Classifier, param_distributions=GB_Params, n_iter=2, scoring='roc_auc',
                                   fit_params=None, cv=None, verbose=2).fit(train_X, train_Y)
    GB_Predicts = GB_Search.predict_proba(test_X)
    # Saving the trained model
    GB_Filename = "GB_Classifier.pkl"
    with open(GB_Filename, 'wb') as file:
        pickle.dump(GB_Search, file)
    return GB_Predicts

def RF_Classifier_func(train_X, train_Y, test_X):
    '''
    This function build a classification model based on the Random Forest classifier.
    :param train_X: train_X: training data set which is provided.
    :param train_Y: the probability of delinquency corresponding to each index.
    :param test_X: test data set which is provided.
    :return: It returns the predictions on the delinquency probabilities for the test data set.
    '''
    RF_Classifier = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                           min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
                                           bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=1)
    RF_Params = {'criterion': ['gini'], 'n_estimators': [100, 200, 300, 400,500], 'max_depth': [6, 8, 10]}
    RF_Search = RandomizedSearchCV(estimator=RF_Classifier, param_distributions=RF_Params, n_iter=2, scoring='roc_auc',
                                   fit_params=None, cv=None, verbose=2).fit(train_X, train_Y)
    RF_Predicts = RF_Search.predict_proba(test_X)
    # Saving the trained model
    RF_Filename = "RF_Classifier.pkl"
    with open(RF_Filename, 'wb') as file:
        pickle.dump(RF_Search, file)
    return RF_Predicts

# Function to evaluate performance of the proposed classifiers

# def Perf_Eval(classifier, test_features, test_labels):
#     predictions = classifier.predict(test_features)
#     errors = abs(predictions - test_labels)
#     accuracy = 100 * np.mean(errors / test_labels)
#     return accuracy
#
#

def plot_auc_roc(classifier, train_X, train_Y, nfolds):
    '''

    :param classifier:
    :param train_X:
    :param train_Y:
    :param nfolds:
    :return:
    '''
    d_fold = KFold(nfolds, shuffle=True)
    for KFold_train, KFold_test in d_fold.split(train_X, train_Y):
        predics_kfold = classifier.fit(train_X.iloc[KFold_train], train_Y.iloc[KFold_train]).predict_proba(
            train_X.iloc[KFold_test])
        fpr, tpr, thresholds = roc_curve(train_Y.iloc[KFold_test], predics_kfold[:, 1])
        roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='(area = %0.2f)' % roc_auc)


[train_data,test_data,train_labels,test_labels,feature_list] = data_collection()
[train_nan_features,train_an_cnt] = nan_detector(train_data, feature_list)
[test_nan_features,test_nan_cnt] = nan_detector(test_data, feature_list)
[train_data,train_data_nan_drop] = nan_rmv_indication(train_data, train_nan_features)
[test_data, test_data_nan_drop] = nan_rmv_indication(test_data, test_nan_features)
#feature_dist_plot(train_data,train_data_nan_drop,feature_list, train_nan_features)



# For features with narrow distributions log-scale distribution provides better insights
log_feature_list = ['RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse','MonthlyIncome']
for l_feature in log_feature_list:
    train_data[l_feature] = np.log(1 + train_data[l_feature].values)
    test_data[l_feature] = np.log(1 + test_data[l_feature].values)

train_data = nan_replacement(train_data, train_nan_features)
test_data = nan_replacement(test_data, test_nan_features)
train_data = outlier_detector(train_data, feature_list)
train_data, test_data = feature_augmentation(train_data,test_data,feature_list)




train_X = train_data.drop(['SeriousDlqin2yrs', 'Unnamed: 0'], axis=1, inplace=False)
train_Y = train_labels
test_X = test_data.drop(['SeriousDlqin2yrs', 'Unnamed: 0'], axis=1, inplace=False)
test_Y = test_labels


GB_Predicts = GB_Classifier_func(train_X, train_Y, test_X)
RF_Predicts = RF_Classifier_func(train_X, train_Y, test_X)

plot_auc_roc(GB_Predicts, train_X, train_Y, nfolds=5)
plot_auc_roc(RF_Predicts, train_X, train_Y, nfolds=5)

