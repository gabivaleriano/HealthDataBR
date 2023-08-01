# HealthDataBR
A repository with classification problems from the Unified Health System (SUS) in Brazil.

Data present is this dataset was originally extracted from DATASUS, the TI department of SUS. We adopted databases of the Notifiable Diseases Information System (SINAM). 

The main branch includes a file with functions for data visualization and evaluating Machine Learning (ML) models performance.

The repository is structured as follows:

Each folder, named as the original dabase, contains: 

## Data pre-processing notebooks (pre_[disease]_[year].ipynb)

Data of the year 2022 was adopted for the main dataset. A second dataset, from the years 2021 or 2020, was assembled to evaluate models performance. 

## Variables dictionary ([disease]_dictionary.pdf)

Provides the meaning and content of each feature.

## Pre_processing filters ([disease]_filters.pdf)

Includes a brief description of filters used to select features and patients.

## Dataset folders

Each dataset folder contains: 

### Main and validation datasets ([disease]_[outcome]_[year}.csv)

One is the main dataset, containing data from the year 2022. The second is a dataset containg data from the year 2020 or 2021, employed as external validation data. 

### Exploratory analysis and model performance notebook

Exploring features distribution and ML models performance. 
