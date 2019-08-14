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
Now that the features with NaN values are detected and the NaN values are dropped from the corresponding features, a visualized observation would be helpful to get an idea about the distributions of the provided data. Accordingly, in this code distributions of each feature are plotted in both linear and logarithmic scale. Those plots are as follow:
