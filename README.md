# GiveMeSomeCredit
This is a Python code for the Give_Me_Some_Credit project in Kaggle, the goal of which is to obtain the probability of delinquency for borrowers.

The labeled-data provided in this project is bunch of information about 150,000 clients that is in the format of different features such as the income, debt, number of independent people and etc., along with the corresponding labels indicating if delinquency occurs for each client. 

This project can be considered as a classification problem via supervised-learning methods. In particular, the goal is to train a classifier to obtain a sufficiently accurate label/probability about delinquency occurrence. 

The first step to tackle this problem is to preprocess the provided data to make sure that the analysis is based on reliable and informative data. Four conventional and important steps in preprocessing data are:
i)	Removing NaN entries in data set.
ii)	Data visualization and data scaling.
iii)	Finding and replacing outliers in data set.
iv)	Data augmentation via extracting new features.

Let’s briefly explain the methods used for each of those four steps:

i) Removing NaN entries in data set:
First, the features that include NaN values and the number of NaN entries in each of those features are detected; i.e., function: “nan_detector”. Then, the sample data with NaN values are all dropped so that the distribution of each feature can be plotted. (All the data samples with no NaN entry is saved as another data frame with the name “data_nan”.) At the same time for each of the features which includes a NaN entry, a new column/feature is generated to indicate the entries with NaN value. The rationale behind this step is the fact that NaN data could be potentially informative and it could facilitate the training process of the classification model. 

ii) Data visualization and data scaling:
Now that the features with NaN values are detected and the NaN values are dropped from the corresponding features, a visualized observation would be helpful to get an idea about the distributions of the provided data. Accordingly, in this code distributions of each feature are plotted in both linear and logarithmic scale. Those plots are as follow in which the plots for each feature are in the order of linear and log-scale:

<img width="635" alt="1-linear" src="https://user-images.githubusercontent.com/51095441/63051324-04c1b700-beab-11e9-9bce-7f8e26b60593.png">
<img width="632" alt="1-log" src="https://user-images.githubusercontent.com/51095441/63051352-173bf080-beab-11e9-9717-3b988f8084eb.png">
<img width="638" alt="2-linear" src="https://user-images.githubusercontent.com/51095441/63051418-3470bf00-beab-11e9-90d7-1e1bb4baf014.png">
<img width="638" alt="2-log" src="https://user-images.githubusercontent.com/51095441/63051430-389cdc80-beab-11e9-98b8-67e95a5bbc5c.png">
<img width="634" alt="3-linear" src="https://user-images.githubusercontent.com/51095441/63051442-3c306380-beab-11e9-83ef-e7124faf1901.png">
<img width="636" alt="3-log" src="https://user-images.githubusercontent.com/51095441/63051447-3e92bd80-beab-11e9-9f50-1c495dd7f707.png">
<img width="635" alt="4-linear" src="https://user-images.githubusercontent.com/51095441/63051453-418dae00-beab-11e9-98bf-d66588e6e81d.png">
<img width="637" alt="4-log" src="https://user-images.githubusercontent.com/51095441/63051460-45213500-beab-11e9-9f9e-f6024c833597.png">
<img width="635" alt="5-linear" src="https://user-images.githubusercontent.com/51095441/63051470-4bafac80-beab-11e9-95c0-2540f179cb89.png">
<img width="638" alt="5-log" src="https://user-images.githubusercontent.com/51095441/63051477-4e120680-beab-11e9-9f6c-7e820929f952.png">
<img width="634" alt="6-linear" src="https://user-images.githubusercontent.com/51095441/63051484-50746080-beab-11e9-9e96-de335a58574d.png">
<img width="640" alt="6-log" src="https://user-images.githubusercontent.com/51095441/63051487-52d6ba80-beab-11e9-9ecb-f174dfa93dd2.png">
<img width="636" alt="7-linear" src="https://user-images.githubusercontent.com/51095441/63051490-55391480-beab-11e9-9a07-717070b3c02f.png">
<img width="639" alt="7-log" src="https://user-images.githubusercontent.com/51095441/63051495-5702d800-beab-11e9-96d9-777205d739f2.png">
<img width="636" alt="8-linear" src="https://user-images.githubusercontent.com/51095441/63051500-59653200-beab-11e9-9354-7cc6f2d5b5bf.png">
<img width="635" alt="8-log" src="https://user-images.githubusercontent.com/51095441/63051506-5b2ef580-beab-11e9-9289-eee74b5f0766.png">
<img width="633" alt="9-linear" src="https://user-images.githubusercontent.com/51095441/63051508-5d914f80-beab-11e9-98b9-8e3b973d9295.png">
<img width="632" alt="9-log" src="https://user-images.githubusercontent.com/51095441/63051517-6124d680-beab-11e9-91e6-4d3221e80bee.png">
<img width="637" alt="10-linear" src="https://user-images.githubusercontent.com/51095441/63051528-63873080-beab-11e9-84ae-ee7e8ba5ab6f.png">
<img width="635" alt="10-log" src="https://user-images.githubusercontent.com/51095441/63051533-65e98a80-beab-11e9-92c6-fb319527ca81.png">

It can be seen from the above plots that the distribution in logarithmic-scale is more informative (less-skewed) for some features, including:
'RevolvingUtilizationOfUnsecuredLines', 
'NumberOfTime30-59DaysPastDueNotWorse', 
'DebtRatio',
'NumberOfTimes90DaysLate', 
'NumberRealEstateLoansOrLines', 
'NumberOfTime60-89DaysPastDueNotWorse',
'MonthlyIncome'.

After finding the distributions and making decisions about using either linear scale or log scale of the (NaN-removed) data, the NaN values can be replaced by any reasonable values. In this code NaN values in each feature are replaced by the median of the contents of that feature.

iii) Finding and dropping the outliers in data set:
The last step in processing the data is to detect outliers and replace them with a rational value by which no information would be missed or manipulated. Considering the distribution figures provided above, it can be seen that each feature has a few number of outliers, and hence replacing the outliers of each feature with median value of the corresponding feature would not lead to any significant information loss. 

iv) Data augmentation via extracting new features:
Before starting to train the desired model one more step is taken to improve the accuracy and robustness of the model and that is extracting new features from the existing features. In this step by considering the features that are provided in the data and the possible relationship between each of those features, new features is generated that can lead to an improvement in the performance of the model. In this code, two more features are extracted and added to the feature list; namely, ‘WeightedPastDue’ and ‘MonthlyNetIncome’. The former one is a weighted sum of the number of days that past due occurs. The rational behind the weights is the fact that a longer past due could have more impact on (or let’s say it could be more informative about) the probability of delinquency. The latter one, ‘MonthlyNetIncome’ is nothing but the amount of money left after paying all the debts in each month. It is expected that the enteries with negative ‘MonthlyNetIncome’ are more prone to delinquency. 

Training:
Now that the data is processed to deal with NaN values and outliers and some additional features are extracted, training can be started. In order to tackle this classification problem two different classifiers are considered in this code, namely Random Forest classifier and Gradient Boosting classifier. The performance of each of those classifiers are provided at the end of this report. 

The first classifier is Random Forest classifier which is based on bagging of the predictions of different decision trees. The second classifier is Gradient Boosting classifier which is based on boosting the predictions of sub-predictors to improve the final performance of the classification. In both cases ‘RandomizedSearchCV’ is employed to find the best estimator according to the parameters that are given for each classifier. After finding the best estimator under each of the classifiers that estimator is applied to the test data to measure how accurate the trained model is. Note that when the model is trained according to the obtained best estimator, the model is saved using pickle package so that the model can be used any time after. In order to evaluate the accuracy of the trained model ‘cross validation’ procedure, more specifically K-fold cross validation, is employed. That would illustrate how accurate the trained model is over different sets of data. The performance of the model is then presented using ROC curve, which is plotted with True Positive Ration (TPR) vs False Positive Ratio (FPR). 
<img width="633" alt="RF-10Folds" src="https://user-images.githubusercontent.com/51095441/63052207-c2997500-beac-11e9-8520-ee31e471414d.png">
<img width="631" alt="GB-10Folds" src="https://user-images.githubusercontent.com/51095441/63052212-c5946580-beac-11e9-9871-d9c73c87ab32.png">

It can be observed from the above figures that both classifiers can achieve a satisfying performance, while the performance of Gradient Boosting classifier is on average slightly better than that of Random Forest classifier. In order to further improve the performance, some other ideas can be implemented which are briefly discussed in the following part.

Some More Ideas to improve the performance:
There are some other methods that can be employed to improve either the accuracy or robustness of the model.

- More steps in data preprocessing: Different methods for replacing the outliers and NaN values can be employed and investigating whether or not there is a relationship between having a NaN entry in one feature and an outlier entry in another feature of a specific client. 
- Extracting more sophisticated features: This could illustrate hidden information in the relationship between different provided features may lead to significant improvement in the accuracy of the trained model. 
- More sophisticated methods in parameter selection: As an example ‘GridSearchCV’ can be employed instead of ‘RandomizedSearchCV’.
- Other methods for evaluating the cross validation: For instance, stratified K-fold cross validation can be used rather than the K-fold

References:
Working on this project I have investigated some different resources to gather some ideas and insights about different classification methods and the applications each classification method would fit. The very first resource was Kaggle website and the forums on GiveMeSomeScore project. The discussions and analysis provided there gave me a great idea to start this project. The Python website itself was also used for looking for different commands and libraries that I need to implement this project. Moreover, the website https://towardsdatascience.com was a big help in providing some insights on choosing and tuning the hyper parameters.  




