# Lab 1

Scalabale Machine Learning (ID2223)

By Tobias Lord and Jacob Dahlkvist

## Purpose

The purpose of the lab was to learn how you can implement a machine learning pipeline utilizing cloud compute. This is the general flow of the implementation:

1. Dataset is cleaned and transformed into features (locally or via Modal)
2. Features are stored in Hopsworks
3. Model is created and trained using above mentioned fatures.
4. Hugging face is used to provide an interactive UI for the user to test the model.
5. Modal is used to utilize computation power in the cloud and to perform repeating jobs, such as (a) adding data to our feature store, (b) regenerating the model with regards to new data, and (c) generate metrics for the Hugging face UI.

### Links

#### Iris

Monitor: https://huggingface.co/spaces/tlord/iris_monitor

Predictor: https://huggingface.co/spaces/tlord/iris

#### Titanic

Monitor: https://huggingface.co/spaces/tlord/titanic_monitoring

Predictor: https://huggingface.co/spaces/tlord/titanic

## Titanic implementation

### Data

*The dataset can be found here: https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv*

We decided to keep following columns:

**Categorical**
- PClass
- Sex
- Embarked

We used SimpleImputer to replace unknown values with most_frequent. OrdinalEncoder was used to transform into numerical values.

**Numerical**
- Age
- Fare
- SibSp
- ParCh

We used SimpleImputer to replace unknown values with the mean. StandardScaler was used to scale values using the standard (x - u)/s.

**Other**
- Name, for indexing reasons only

Remaining columns were dismissed.

### Model

The Titanic model is just a simple Support Vector Machine with an RBF kernel. No parameter tuning was applied. We evaluated multiple models using cross-validation. The performance of the model was an average accuracy of 0.825 and std of 0.035.