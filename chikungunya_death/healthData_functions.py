#!/usr/bin/env python
# coding: utf-8

# In[147]:


import pandas as pd
import numpy as np
import time
import math
import warnings

import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pyhard.classification import ClassifiersPool

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer





# In[148]:


# set number of folds for cross validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

# set preprocessing
preprocessing = StandardScaler()



# In[149]:


'''
Returns balanced data, with the same number of samples in both classes. 
If the minority class is less than 5%, applies oversampling and undersampling. 
On the other hand, applis just undersample. 
'''

def data_sample(X,y):
    
    # define sampling strategies 
    undersample = RandomUnderSampler(sampling_strategy='majority', random_state = 1)
    #oversample = SMOTE(sampling_strategy=0.05, random_state = 1)
    oversample = RandomOverSampler(sampling_strategy=0.05, random_state = 1)
    
    # check if the minority class is present in at least 5% of the dataset   

    # identify the minority class
    # identify the minority class
    count_1 = (y == 1).sum()
    count_0 = (y == 0).sum()
    count_min = min(count_0, count_1)
    count_max = max(count_0, count_1)

    # calculate the percentage of the minority class compared to the total number of instances
    ratio = (count_min / count_max) 
    
    # if the minority class is more than 60% of the majority class does not apply any technique
    if ratio > 0.6: 
        X_resampled, y_resampled = X,y

    # check if the percentage of 1 is at least 5% of the total number of instances
    # if is less than 5% applies over and under sample    
    else: 
        if ratio <=  0.05:
            X_resampled, y_resampled = oversample.fit_resample(X, y)
            X_resampled, y_resampled = undersample.fit_resample(X_resampled, y_resampled)

        # if not applies just undersample    
        else:
            X_resampled, y_resampled = undersample.fit_resample(X, y) 
        
    return (X_resampled,y_resampled)


# In[209]:


'''
Receives data and algorithms to be evaluated. 
Returns the average performance inside cross-validation, using 3 metrics.
Applies over-under sample to get balanced datasets. 
Standardize features. 
'''
algorithms = {
    
    'svc_linear': (SVC(probability=True, kernel='linear',random_state=0)),
    
    'svc_rbf': (SVC(probability=True, kernel='rbf', random_state=0)),
    
    'random_forest' : (RandomForestClassifier(random_state=0)),    
    
    'gradient_boosting' : (GradientBoostingClassifier(random_state=0)),
    
    'logistic_regression' : (LogisticRegression()),
    
    'bagging' : (BaggingClassifier(random_state=0)),
    
    'mlp': (MLPClassifier(random_state=0)),
            
}

def evaluate_cv(data):    
    
    # record the start time
    start_time = time.time()
    
    #identify the target column
    target_feature = data.columns[-1]
    
    # separate features (X) and target (y)
    X = data.drop(columns=[target_feature])
    y = data[target_feature]
    
    # stores recall and auc of each algorithm
    sen = {}
    for algorithm in algorithms.keys():
        sen[algorithm] = []

    spe = {}
    for algorithm in algorithms.keys():
        spe[algorithm] = []

    auc = {}
    for algorithm in algorithms.keys():
        auc[algorithm] = []

    # for each round of the cross validation
    for train, test in kf.split(X, y):

        # alocate train and test
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]

        # applies oversampling / undersampling
        X_train, y_train = data_sample(X_train, y_train)

        X_train = preprocessing.fit_transform(X_train)
        X_test = preprocessing.transform(X_test)

        # for each algorithm 
        for algorithm, (clf) in algorithms.items():
            
            # vector to store y_pred 
            y_pred = [] 

            # train model
            clf.fit((X_train), y_train)

            # make predictions for the test data
            y_pred = [*y_pred, *(clf.predict(X_test))]

            # calculate sensibility and specificity 
            recallscore = recall_score(y_test, y_pred, labels = [0,1], average = None)
            sen[algorithm].append(recallscore[1])
            spe[algorithm].append(recallscore[0])

            # calculate the area under the roc curve
            aucscore = roc_auc_score(y_test, (clf.predict_proba((X_test)))[:, 1])     
            auc[algorithm].append(aucscore)

    # create a df with the mean and sd of each algorithm accross the 5 folds 
    df = pd.DataFrame(columns = list(algorithms.keys()))
    
    df.loc['auc (mean)'] = [np.mean(auc['svc_linear']), np.mean(auc['svc_rbf']), np.mean(auc['random_forest']), 
                            np.mean(auc['gradient_boosting']), np.mean(auc['logistic_regression']), 
                            np.mean(auc['bagging']), np.mean(auc['mlp'])]
    
    df.loc['auc (stdev)'] = [np.std(auc['svc_linear']), np.std(auc['svc_rbf']), np.std(auc['random_forest']), 
                            np.std(auc['gradient_boosting']), np.std(auc['logistic_regression']), 
                            np.std(auc['bagging']), np.std(auc['mlp'])]

    df.loc['sen (mean)'] = [np.mean(sen['svc_linear']), np.mean(sen['svc_rbf']), np.mean(sen['random_forest']), 
                            np.mean(sen['gradient_boosting']), np.mean(sen['logistic_regression']), 
                            np.mean(sen['bagging']), np.mean(sen['mlp'])]

    df.loc['sen (stdev)'] = [np.std(sen['svc_linear']), np.std(sen['svc_rbf']), np.std(sen['random_forest']), 
                            np.std(sen['gradient_boosting']), np.std(sen['logistic_regression']), 
                            np.std(sen['bagging']), np.std(sen['mlp'])]

    df.loc['spe (mean)'] = [np.mean(spe['svc_linear']), np.mean(spe['svc_rbf']), np.mean(spe['random_forest']), 
                            np.mean(spe['gradient_boosting']), np.mean(spe['logistic_regression']), 
                            np.mean(spe['bagging']), np.mean(spe['mlp'])]

    df.loc['spe (stdev)'] = [np.std(spe['svc_linear']), np.std(spe['svc_rbf']), np.std(spe['random_forest']), 
                            np.std(spe['gradient_boosting']), np.std(spe['logistic_regression']), 
                            np.std(spe['bagging']), np.std(spe['mlp'])]

   
    df = df.style.set_caption('Average performance and stdev among 5-fold cross-validation')

    # record the end time
    end_time = time.time()
    
    # calculate the time taken
    total_time = end_time - start_time
    
    display(df)
    
    print(f"Total time taken to run cross-validation: {total_time:.2f} seconds")
    
    return(df)


# In[210]:


'''
Receives data and algorithms to be evaluated. 
Returns the average performance, using 3 metrics.
Applies over-under sample to get balanced datasets. 
Standardize features. 
'''

def evaluate_external(data, data_test): 
    
    # record the start time
    start_time = time.time()
    
    #identify the target column
    target_feature = data.columns[-1]
    
    # separate features (X) and target (y)
    X = data.drop(columns=[target_feature])
    y = data[target_feature]

    # separate features (X) and target (y)
    X_test = data.drop(columns=[target_feature])
    y_test = data[target_feature]
    
    # stores recall and auc of each algorithm
    sen = {}
    for algorithm in algorithms.keys():
        sen[algorithm] = []

    spe = {}
    for algorithm in algorithms.keys():
        spe[algorithm] = []

    auc = {}
    for algorithm in algorithms.keys():
        auc[algorithm] = []        
        
    # applies oversampling / undersampling
    X_train, y_train = data_sample(X, y)
  
    # applies preprocessing
    X_train = preprocessing.fit_transform(X_train)
    X_test = preprocessing.transform(X_test)
  
    # for each algorithm 
    for algorithm, (clf) in algorithms.items():

        # vector to store y_pred 
        y_pred = [] 

        # train model
        clf.fit((X_train), y_train)

        # make predictions for the test data
        y_pred = [*y_pred, *(clf.predict(X_test))]

        # calculate sensibility and specificity 
        recallscore = recall_score(y_test, y_pred, labels = [0,1], average = None)
        sen[algorithm].append(recallscore[1])
        spe[algorithm].append(recallscore[0])

        # calculate the area under the roc curve
        aucscore = roc_auc_score(y_test, (clf.predict_proba((X_test)))[:, 1])     
        auc[algorithm].append(aucscore)
    
    # Create a df with the mean and sd of each algorithm across the 5 folds 
    df = pd.DataFrame(columns=list(algorithms.keys()))

    df.loc['auc (mean)'] = [auc['svc_linear'][0], auc['svc_rbf'][0], auc['random_forest'][0], 
                            auc['gradient_boosting'][0], auc['logistic_regression'][0], 
                            auc['bagging'][0], auc['mlp'][0]]

    df.loc['sen (mean)'] = [sen['svc_linear'][0], sen['svc_rbf'][0], sen['random_forest'][0], 
                            sen['gradient_boosting'][0], sen['logistic_regression'][0], 
                            sen['bagging'][0], sen['mlp'][0]]

    df.loc['spe (mean)'] = [spe['svc_linear'][0], spe['svc_rbf'][0], spe['random_forest'][0], 
                            spe['gradient_boosting'][0], spe['logistic_regression'][0], 
                            spe['bagging'][0], spe['mlp'][0]]
   
    df = df.style.set_caption('Performance for external validation')
   
    # record the end time
    end_time = time.time()
    
    # calculate the time taken
    total_time = end_time - start_time
    
    display(df)
    
    print(f"Total time taken to run external-validation: {total_time:.2f} seconds")

    return(df)


# In[211]:


def importance(data):
    
    #identify the target column
    target_feature = data.columns[-1]

    # separate features (X) and target (y)
    X = data.drop(columns=[target_feature])
    y = data[target_feature]

    X,y = data_sample(X,y)

    # Create the three models
    models = [
        RandomForestClassifier(random_state=0),
        GradientBoostingClassifier(random_state=0),
        LogisticRegression()
    ]

    # List to store the importance values for each model
    importances = []

    # Fit each model and record the importance scores
    for model in models:
        model.fit(X, y)
        if isinstance(model, LogisticRegression):
            importance = model.coef_[0]  # Coefficients for logistic regression
        else:
            importance = model.feature_importances_  # Feature importances for other models
        importances.append(importance)

    # Create a 1x3 grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot feature importance for each model
    for i, model in enumerate(models):
        axes[i].bar(X.columns, importances[i])
        axes[i].set_xticklabels(X.columns, rotation=90)
        axes[i].set_xlabel('Feature')
        axes[i].set_ylabel('Importance Score')
        axes[i].set_title(f'Feature Importance - Model {i + 1}')

    plt.suptitle('Feature Importance for Random Forest, Gradient Boosting and Logistic Regression', fontsize=16)
    
    # Adjust the layout
    plt.tight_layout()

    return(plt)


# In[ ]:


def evaluate(data, data_test):
    
    evaluate_cv(data)
    evaluate_external(data, data_test)
    importance(data)


# In[ ]:


cmap = sns.cubehelix_palette(n_colors=8, as_cmap=True)
color_0 = cmap.colors[50]
color_1 = cmap.colors[255]

# Define your custom color palette here
custom_palette = [color_0, color_1]

# Set the custom color palette
sns.set_palette(custom_palette)

sns.set_style('white') 
  
sns.set_context("notebook", font_scale=0.8)

def binary_columns(data):
    """
    Get the names of columns in the dataset that are composed of binary numbers.

    Parameters:
        dataset (pd.DataFrame): Input DataFrame.

    Returns:
        list: List of column names containing binary numbers.
    """
    data = data.iloc[:, :-1]
    
    binary_columns = []
    for column in data.columns:
        unique_values = data[column].unique()
        if set(unique_values) == {0, 1}:
            binary_columns.append(column)
    return binary_columns

'''
Receives dataset name.
Read data.
Return class value_counts. 
'''

def size(data):

    # read and identify the target feature
    target_feature = data.columns[-1]
    
    # number of rows and columns
    num_rows, num_columns = data.shape
    print(f"Dataset size: {num_rows} rows, {num_columns} columns")
    
    # distribution of the target feature
    target_distribution = data[target_feature].value_counts(normalize=True)
    print(f"\nDistribution of the target feature:")
    print(target_distribution)

'''
Receives a dataset.
Returns the heatmap.
'''
def heatmap(data): 
    
    #heatmap 
    corr_matrix = data.corr() 
    
    plt.figure(figsize=(7, 5))
    
    heatmap = sns.heatmap(corr_matrix, cmap=sns.cubehelix_palette(as_cmap=True))

    # Customize the plot
    plt.title("Correlation Between Features")

    return heatmap

'''
Receives data, name of the each class. 
Returns the distribution of symptoms and comorbities.
'''

def symptoms(data, class_0, class_1):
    
    # identify binary columns
    binary_col = binary_columns(data)
    
    # select symptoms and comorbidities and filter by the target_column
    ratios = data.groupby(data.columns[-1])[binary_col].mean()

    # create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # plot the bar plot with the custom color palette
    ratios.T.plot(kind='bar', ax=ax, width=0.5)
    ax.grid(False)

    # set the labels and title
    ax.set_xticklabels(binary_col, rotation=90)
    ax.set_xlabel('Symptoms and Comorbidities')
    ax.set_ylabel('Proportion of Symptom and Comorbidities')
    ax.set_title('Proportion of Symptoms and Comorbidities in each Class')

    # Place the legend inside the plot on the left side
    ax.legend(title='Class', labels=[class_0, class_1], loc='upper right')
    
    plt.tight_layout()

    return(plt)

'''

'''

def age(data, class_0, class_1):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4),  gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.4})
    
    #identify the target column
    target_feature = data.columns[-1]

    num_bins = 20

    # plot the histogram for the 'class_1' class
    sns.histplot(data=data[data[target_feature] == 1], x='age', kde=False, ax=axes[0], bins=num_bins, color=color_1)
    axes[0].set_title('Age Distribution for '+ class_1)
    axes[0].set_xlabel('Age - years')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(False)

    # plot the histogram for the 'class_0' class
    sns.histplot(data=data[data[target_feature] == 0], x='age', kde=False, ax=axes[1], bins=num_bins, color=color_0)
    axes[1].set_title('Age Distribution for '+ class_0)
    axes[1].set_xlabel('Age - years')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(False)
    
    plt.tight_layout()
    
    return(plt)

def bars(data, class_0, class_1): 
    
    target_feature = data.columns[-1]

    # Filter the data for each class
    class_1_data = data[data[target_feature] == 1]
    class_0_data = data[data[target_feature] == 0]

    # Count the occurrences of each 'race' category for each class
    class_1_raca_counts = class_1_data['race'].value_counts(normalize=True).reset_index()
    class_0_raca_counts = class_0_data['race'].value_counts(normalize=True).reset_index()

    # Rename the columns for clarity
    class_1_raca_counts.columns = ['race', 'proportion']
    class_0_raca_counts.columns = ['race', 'proportion']

    # Map the numeric 'race' codes to their corresponding labels
    raca_labels = {
        1: 'White',
        2: 'Black',
        3: 'Yellow',
        4: 'Parda',
        5: 'Indigene'
    }

    class_1_raca_counts['race'] = class_1_raca_counts['race'].map(raca_labels)
    class_0_raca_counts['race'] = class_0_raca_counts['race'].map(raca_labels)

    # Merge the two datasets
    merged_data_race = pd.concat([class_1_raca_counts, class_0_raca_counts], keys=[class_1, class_0])

    # Filter the data for each class
    class_1_data = data[data[data.columns[-1]] == 1]
    class_0_data = data[data[data.columns[-1]] == 0]

    # Count the occurrences of each 'schooling_years' category for each class
    class_1_schooling_counts = class_1_data['schooling_years'].value_counts(normalize=True).reset_index()
    class_0_schooling_counts = class_0_data['schooling_years'].value_counts(normalize=True).reset_index()

    # Rename the columns for clarity
    class_1_schooling_counts.columns = ['schooling_years', 'proportion']
    class_0_schooling_counts.columns = ['schooling_years', 'proportion']

    # Map the numeric 'schooling_years' codes to their corresponding labels
    schooling_labels = {
        0: 'Whitout schooling',
        1: 'Elemen. school I - incomplete',
        2: 'Elemen. school I -    complete',
        3: 'Elemen. school II - incomplete',
        4: 'Elemen. school II -    complete',
        5: 'High School - incomplete',
        6: 'High School -    complete',
        7: 'Higher Education - incomplete',
        8: 'Higher Education -    complete'
    }

    #bar_color = sns.color_palette("dark")[6]

    class_1_schooling_counts['schooling_years'] = class_1_schooling_counts['schooling_years'].map(schooling_labels)
    class_0_schooling_counts['schooling_years'] = class_0_schooling_counts['schooling_years'].map(schooling_labels)

    schooling_order = [
        'Whitout schooling',
        'Elemen. school I - incomplete',
        'Elemen. school I -    complete',
        'Elemen. school II - incomplete',
        'Elemen. school II -    complete',
        'High School - incomplete',
        'High School -    complete',
        'Higher Education - incomplete',
        'Higher Education -    complete'
    ]
    merged_data_schooling = pd.concat([class_1_schooling_counts, class_0_schooling_counts], keys=[class_1, class_0])
    #merged_data_schooling['schooling_years'] = pd.Categorical(merged_data_schooling['schooling_years'], categories=schooling_order, ordered=True)

    # Set up the figure and axes for subplots, adjust the width_ratios to give more space to the first plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [1, 0.8], 'wspace': 0.25})

    sns.barplot(data=merged_data_schooling, x='proportion', y='schooling_years', 
                hue=merged_data_schooling.index.get_level_values(0), ax=axes[0], order=schooling_order,
                hue_order=[class_1, class_0], palette=custom_palette) 

    # Set the title, x-axis label, and y-axis label for Schooling Years with fontsize
    axes[0].set_title('Proportion of Schooling Years in Each Class', fontsize=10)
    axes[0].set_xlabel('Proportion', fontsize=8)
    axes[0].set_ylabel('', fontsize=8)  # Empty y-axis label to align with the first plot

    sns.barplot(data=merged_data_race, x='proportion', y='race', width=0.4, 
                hue=merged_data_race.index.get_level_values(0), ax=axes[1], 
                hue_order=[class_1, class_0], palette=custom_palette)

    # Set the title, x-axis label, and y-axis label for Race with fontsize
    axes[1].set_title('Proportion of Race in Each Class', fontsize=10)
    axes[1].set_xlabel('Proportion', fontsize=8)
    axes[1].set_ylabel('', fontsize=8)  # Empty y-axis label to align with the first plot

    # Remove the legend for Race
    axes[0].legend().set_visible(False)

    # Place the custom legend inside the plot on the left side and increase the fontsize and legend box size
    legend_elements = [
        mpatches.Patch(facecolor=color_1, edgecolor='black', label=class_1),
        mpatches.Patch(facecolor=color_0, edgecolor='black', label=class_0)
    ]
    axes[1].legend(handles=legend_elements, title='Class', loc='lower right', fontsize=8)

    plt.tight_layout()
    
    return plt



def state(data, class_0, class_1):
    
    target_feature = data.columns[-1]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Map the numeric 'state' codes to their corresponding labels
    state_labels = {
        11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
        21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL',
        28: 'SE', 29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR',
        42: 'SC', 43: 'RS', 50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'
    }

    # Make a copy of the DataFrame to avoid modifying the original DataFrame
    data_copy = data.copy()

    # Map the 'id_state' column to state labels
    data_copy['State_Label'] = data_copy['id_state'].map(state_labels)

    # Filter the data for each class
    class_1_data = data_copy[data_copy[target_feature] == 1]
    class_0_data = data_copy[data_copy[target_feature] == 0]

    # Calculate the proportion of each state for each class
    class_1_proportions = class_1_data['State_Label'].value_counts(normalize=True).reset_index()
    class_0_proportions = class_0_data['State_Label'].value_counts(normalize=True).reset_index()

    # Rename the columns for clarity
    class_1_proportions.columns = ['State_Label', 'Proportion']
    class_0_proportions.columns = ['State_Label', 'Proportion']

    # Sort the data by state labels for better visualization
    class_1_proportions = class_1_proportions.sort_values('State_Label')
    class_0_proportions = class_0_proportions.sort_values('State_Label')

    # Merge the two datasets
    merged_data = pd.concat([class_1_proportions, class_0_proportions], keys=[class_1, class_0])


    # Plot the barplot with different colors for each class
    sns.barplot(data=merged_data, x='State_Label', y='Proportion',
                hue=merged_data.index.get_level_values(0), hue_order = [class_0, class_1],
                ax=ax)

    # Set the title, x-axis label, and y-axis label
    ax.set_title('Proportion of State in Each Class')
    ax.set_xlabel('State')
    ax.set_ylabel('Proportion')

    # Rotate the state labels on the x-axis for better readability
    ax.tick_params(axis='x', rotation=90)

    # Show the legend outside the plot
    ax.legend(title='Class', loc='upper right')

    # Adjust the layout
    plt.tight_layout()

    # Display the plot
    plt.show()

'''

def place(data, class_0, class_1):
    
    target_feature = data.columns[-1]

    # Group data for class_1 == 1 by the number of occurrences of each id_place
    class_1_counts = data[data[target_feature] == 1]['id_place'].value_counts().value_counts().sort_index()
    class_1_counts_df = class_1_counts.reset_index(name='Count_class_1')
    class_1_counts_df.columns = ['Occurrences', 'Count_class_1']

    # Group data for class_1 == 0 by the number of occurrences of each id_place
    class_0_counts = data[data[target_feature] == 0]['id_place'].value_counts().value_counts().sort_index()
    class_0_counts_df = class_0_counts.reset_index(name='Count_class_0')
    class_0_counts_df.columns = ['Occurrences', 'Count_class_0']

    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the distribution of frequencies for class_1
    axes[0].plot(class_1_counts_df['Occurrences'], class_1_counts_df['Count_class_1'], 
                 marker='o', color=color_1)
    axes[0].set_xlabel('Occurrences')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of id_place Frequencies for ' + class_1)
    axes[0].grid(True)
    #axes[0].set_ylim(-10, 500)  # Set the y-axis limit for the first plot

    # Plot the distribution of frequencies for class_0
    axes[1].plot(class_0_counts_df['Occurrences'], class_0_counts_df['Count_class_0'], 
                 marker='o', color=color_0)
    axes[1].set_xlabel('Occurrences')
    axes[1].set_title('Distribution of id_place Frequencies for ' + class_0)
    axes[1].grid(True)
    #axes[1].set_ylim(-50, 2100)  # Set the y-axis limit for the second plot

    plt.tight_layout()
    return plt

'''
def week(data, class_0, class_1):
    
    target_feature = data.columns[-1]
    
    # Separate data for class_1 and class_0
    class_1_data = data[data[target_feature] == 1]
    class_0_data = data[data[target_feature] == 0]

    # Group data by epidemiological week and count class_1s for each week
    class_1_counts = class_1_data.groupby('epidemiological_week').size()

    # Group data by epidemiological week and count class_0s for each week
    non_class_1_counts = class_0_data.groupby('epidemiological_week').size()

    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot class_1 counts
    axes[0].plot(class_1_counts.index, class_1_counts.values, marker='o', linestyle='-', color=color_1)
    axes[0].set_xlabel('Epidemiological Week')
    axes[0].set_ylabel('Count')
    axes[0].set_title(class_1 + ' by Epidemiological Week')
    axes[0].grid(True)

    # Plot class_0 counts
    axes[1].plot(non_class_1_counts.index, non_class_1_counts.values, marker='o', linestyle='-', 
                 color=color_0)
    axes[1].set_xlabel('Epidemiological Week')
    axes[1].set_title(class_0 + ' by Epidemiological Week')
    axes[1].grid(True)

    plt.tight_layout()

    return plt


def visualization (df, class_0, class_1): 
    size(df)
    heatmap(df)
    age(df, class_0, class_1)
    symptoms(df, class_0, class_1)
    bars(df, class_0, class_1)
    #place(df, class_0, class_1)
    state(df, class_0, class_1)
    week(df, class_0, class_1)

