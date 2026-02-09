# Tennis Breakthrough Project Overview

This is an end-to-end supervised machine learning pipeline to predict the next tennis players who will breakthrough into the ATP top 100. It uses a SQLite database to store match, tournament, player, and ranking data sourced from [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp). Predictive variables, or features, are then using database queries which are then read by a python script. This adds more features, selects the optimal feature set for a logistic regression and random forest model, tunes their hyperparameters, and then measures the models performance on a final test set. The result are Macro F1 metrics for logistic regression and random forest.

## Context

Every year, hundreds of player compete on the ATP Challenger and Futures tour hoping to breakthrough onto the elusive ATP tour, a collection of tournaments reserved for only the best of the tennis world. This project uses supervised machine learning to predict the players who are the most likely to break into the ATP's top 100 within the next year.

**The Challenge:** With less than 5% of players breaking through annually, this is an extreme class imbalance problem requiring sophisticated feature engineering to filter noise from signal.

**The Approach:** This project engineered composite predictive variables composed of the following to use in features and hyperparameter optimized random forest and logistical regression models.

- **Competitive Volume** Number of matches played, quality of opponent
- **Psychological Toughness** Average loss time
- **Ranking Trajectory** Initial ranking, ranking velocity, ranking percent change

## Key Results

This section outlines achievement in feature engineering and model performance.

### Feature Engineering Achievement

The initial feature set's best feature had 0.167 correlation with breakthrough. After iteratively engineering over 50 features including composite metrics combining match psychology, ranking trajectory, and competitive volume, the resulting best feature has a 0.734 correlation with breakthrough.

### Model Performance

- Logistic Regression achieved 0.5113 Macro F1 on test set with 18 features and balanced class weights.
- Random Forest achieved 0.4937 Macro F1 on test set with 6 features.
- Strong Performance on No Progress, F1: 0.89, 0.92.
- The other classes struggled, limited sample sizes (n<25) hindered the models ability to train effectively.
- Breakthrough class proved most difficult, F1:(0.14, 0.0), which was significantly worse than the three other classes. This is a result of an inherently difficult problem with extremely limited training data. Improvement here would require significantly stronger features.
- The challenger class had the second worst f1-score for both models despite having the second highest sample size, F1:(0.52, 0.34). This suggests features struggle to distinguish elite players and players near boundaries. These are areas for future work.

**Logistic Regression Model Performance**:

| Tier | Macro F1 | Precision | Recall | Sample Size |
| --- | --- | --- | --- | --- |
| No Progress (301+) | 0.88 | 0.97 | 0.81 | 116 |
| Challenger (201-300) | 0.50 | 0.41 | 0.65 | 23 |
| Rising (101-200) | 0.53 | 0.60 | 0.47 | 19 |
| Breakthrough (1-100) | 0.13 | 0.08 | 0.33 | 3 |

**Random Forest Model Performance**:

| Tier | Macro F1 | Precision | Recall | Sample Size |
| --- | --- | --- | --- | --- |
| No Progress (301+) | 0.92 | 0.92 | 0.92 | 116 |
| Challenger (201-300) | 0.34 | 0.39 | 0.40 | 23 |
| Rising (101-200) | 0.71 | 0.62 | 0.84 | 19 |
| Breakthrough (1-100) | 0.00 | 0.00 | 0.00 | 3 |

## ETL Pipeline

The source data includes information about each tournament, a weekly ranking update, and result and relevant statistics of each match played from every year dating back to 1968 (the year the Open Era began). The primary transformation in the pipeline splits each match into two copies, one from each player's point of view. This significantly simplifies querying the data for matches won and win rates. 44126 match copies were collected from 2021 (22063 matches from the source repo). Ranking data was loaded using the source table structure.

It uses an ETL pipeline to load the data into a SQLite database.

## Exploratory Data Analysis (EDA)

EDA found that my dataset had no missing values and the predictive window had a 3.1% breakthrough rate. My initial set of predictive variables, listed bellow, had a max of 37% and average 16% difference in mean between breakthrough and non breakthrough players.

Initial predictive variables:

- 4 month win rate
- win rate and count against players ranked 25, 50, and 100 spots higher

These variables represent ceiling and floor skill levels. An unrestricted win rate represents a player's floor. Constrained win rates and counts represent a player's ceiling, for example in a match against a player ranked 100 spots higher.

Box Plot analysis showed that each predictive variable had significant or complete overlap between the breakthrough and non breakthrough players. While the range of the win counts had slight difference difference, the win rates spanned from 0.0 to 1.0 for non breakthrough players, making distinguishing breakthrough players very difficult.

Correlation analysis showed 5/7 of these features had less than 0.1 correlation with breakthrough and five pairs of the ceiling features had greater than 0.85 multicolinearity. This clearly suggests that improved predictive variables were needed. The initial binary breakthrough classification restricted the features' correlation and several rounds of feature engineering were needed.

The project was reframed to a 4-tier multiclass system, with the following ranking based tiers:

- No Progress (300+)
- Challenger (201 - 300)
- Rising (101 - 200)
- Breakthrough (1 - 100)

While this shrunk the set of breakthrough players in this predictive window from 3.1% of the dataset to 1.7% it better captured the gradient of player development. This allows a model to better understand players at different levels so that the predictive variables have higher correlation with breakthrough. This improved the initial features to have up to a 0.28 correlation (win count against players ranked 25 spots higher).

Several rounds of feature engineering were performed using average match lengths of wins and losses, demographic data, ranking data, match volume, and other data. There were 4 features with over 0.50 correlation and 12 features with over 0.3 correlation to breakthrough. The feature set is documented in [features.md](./features.md).

## Feature Selection

To select features for a Random Forest (RF) and Logistic Regression (LR) model the recursive feature elimination method was used with cross validation testing. This iteratively removes the weakest features starting with every feature to find the optimal feature set. Cross validation splits the data set into several smaller sets to train and test, which prevents flawed accuracy due to overfitting. Recursive feature elimination found that the LR model was most accurate, based on Macro F1, using 18 features and the RF model was most accurate with 6 features. Macro F1 is used because it balances precision and recall as well as treats all classes equally. With an extreme class imbalance and focus being on the smallest class it is important to put equal weight on each class.

The LR model using a larger feature set is surprising because LR models typically struggle with real world, noisy data. LR assumes strict linear relationships between features and cannot handle non linear relationships. This makes it difficult for LR to handle large sets of correlated features and finding highly predictive variables. Despite this the a large feature set performed best which suggests that even with numerous highly correlated feature pairs that each feature provides enough independent predictive signal to justify its addition to the feature set.

## Hyperparameter Tuning

Hyperparameter tuning was used for both the LR and RF models to optimize the pre-training model settings. This step isn't strictly necessary, it works best with significantly larger data sets, but was done to learn the process for future use.

### LR Tuning

Hyperparameter optimization tested regularization strength and type to find the optimal C value, which balances the trade off between training accuracy and risk of overfitting. This resulted in a 0.5225 cross validation Macro F1 for LR, slightly better than the final test set Macro F1. This suggests that the model slight was overfit on the training data which decreases accuracy on other data sets.

### RF Tunning

Hyperparameter optimization tested the following:

- n estimators, the number of decision trees built.
- max depth, the max depth of each tree.
- min sample splits, the minimum number of samples on each side for a node to split.
- min samples leaf, the minimum number of samples a node must have before splitting.
- max features, the maximum number of features used by each tree.

This tuning resulted in a 0.5214 cross validation Macro F1. This had a larger gap over the final test set Macro F1 which suggests the model also overfit on the training data.

## Modeling

The limited number of breakthrough players makes it very difficult for the model to predict this group. An 80:20 train/test split is used, which result in a test set of around 161 players. The test set is representative of the overall set, hence there are only two or three breakthrough players. Model precision and accuracy can be tested well for the no progress tier because it makes up the vast the majority of the test set. The extreme lack of test cases causes there to be little to no statistical significance in the models prediction of breakthrough players. This is an inherit issue with this problem.

To address the difficulty of predicting breakthrough players either an improved model, dataset, or feature set is required and likely a combination of three would be most effective. Expanding the dataset over a longer time period will be the next focus for this project.

## Results

The tuned LR model had a 0.5225 CV Macro F1 and a 0.5113 test set Macro F1. This used 18 feature set selected by RFE, and the following tuned hyperparameters:

- C: 0,1, default: 1
- class weight: balanced, default: None
- pentaly: l2, default: l2
- solver: saga, default: lbfgs
- max iter: 2000, default: 100

The no progress, 0.89, challenger, 0.52, and rising, 0.56, classes all had good Macro F1 scores. However the breakthrough class had just a 0.14 Macro F1 score with incredibly bad precision at just 0.09.

The tuned RF model had a 0.5214 CV Macro F1 and a 0.4937 test set Macro F1. This was the result of the 6 feature set from the RFE-CV feature selection, and the following tuned hyperparameters:

- class weight: balanced, default: None
- max depth: 20, default: None
- max features: sqrt, default: sqrt
- min samples leaf: 4, default: 1
- min sample split: 10, default: 2
- n estimators: 300, default: 100

The no progress, 0.92 and rising, 0.71, classes had impressive Macro F1 scores. The challenger class had a 0.34 Macro F1 score with similarly bad precision and recall. The breakthrough class had a 0.0 Macro F1. The breakthrough sample size being so low gives little meaning to this result, however it is obvious higher quality data and training is needed.

## Future Work

**Data Expansion**:

- A major issue with this project is the lack of data. One solution to this is to train on multiple 16 month windows at once. Changes in the game would have to be accounted for as to not skew or confused the model.
- This model only uses ATP, men's tour, data. There is an equivalent WTA, women's tour, which could easily be added to this model or used to make a separate model.

**Prediction Pipelines**:

- Transform jupyter notebooks into callable prediction functions with configurable time windows
- This function would return a breakthrough prediction on a single player, list of player, players of a certain nationality, or the whole predicted breakthrough class

**Model Improvements**:

- Engineer transition or challenger tier specific features to improve model prediction accuracy at the challenger tier
- Continue to engineer independent features. For example using match stats over time, 1st serve percentage, unforced errors
