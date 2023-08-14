# HealthDataBR
A repository with classification problems from the Unified Health System (SUS) in Brazil.

Data present is this dataset was originally extracted from DATASUS, the TI department of SUS. We adopted databases of the Notifiable Diseases Information System (SINAM). 

The main branch includes a file with functions for data visualization and evaluating Machine Learning (ML) models performance.

Please note that this repository is a work in progress. In the coming days, we will be adding the following folders: dengue, COVID-19, leishmaniasis, leptospirosis, meningitis, and accidents involving venomous animals.

Each folder, named as the original dabase + outcome, contains: 

## Data pre-processing notebooks ([disease]\_[outcome]\_preprocessing_[year].ipynb)

Data of the year 2022 was adopted for the main dataset. A second dataset, from the years 2021 or 2020, was assembled to evaluate models performance. 

## Variables dictionary ([disease]_[outcome]_dictionary.pdf)

Provides the meaning and content of each feature. Includes a brief description of filters used to select features and patients.

## Main and validation datasets ([disease]\_[outcome]\_dataset_[year].csv)

One is the main dataset, containing data from the year 2022. The second is a dataset containg data from the year 2020 or 2021, employed as external validation data. 

## Exploratory analysis and model performance notebook ([disease]\_[outcome]\_exploration_[year].ipynb)

Exploring features distribution in the main dataset. ML models performance inside a cross-validation and inside a external validation. 
