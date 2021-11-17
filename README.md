# Verve: Data Science Case Study


---

Author: Karen Danielyan

---


## Questions/Problems


> ** 1. ** Look at the information we have given you and identify 3-5 potential problems you can see with the provided dataset that might make building a classification model difficult.  
> ** 2. ** Describe briefly how you would find the features that are likely to be the most important for your model.  
> ** 3. ** Identify which model you would try first, and at least one advantage and disadvantage of this choice.  


## The Answer

*The answer addresses the problems in general rather than individually.*  


Some pecularities of the data can be identified as problems if suplied to one type of a model. However, some other type of a model can be tolerant to such pecularities. 
Thus, I will be mentioning the problems I can identify mostly related to the tree based model, as that is what I would suggest trying first too. 


- The target variable is slightly imbalanced, but most of the possible models that I would try on it, I am sure won't fail to fit the data because of that slight imbalance. 
- The features contain some NaN values, those should be sufficiently treated before modeling. 
I would certainly not recommend dropping those observations, instead, treat them as a separate level of a categorical variable, as the variables are mostly categorical. 
That could be done during the performance of some type of encoding (let's sat one hot encoding).
- Some levels of categorical variables have quite small sample sizes (example: ad_category Luxury cars, app_category Desserts and baking or the click variable). 
Groupping can be performed between some similar levels based on a domain knowledge to pump up the number of occurences within each level (example: group luxury and mid range cars together). 
However, this decision should depend on either domain knowledge or further analysis about their relation to the target variable.
Why is it important? The tree based models will never consider that feature if, for example in Random Forest model minimum samples leaf hyperparameter would be set on higher number than the number of observations falling into that level of category, and sometimes it makes sense to keep that hyperparameter on a high value to avoid overfitting.
- Non unique user_id. This is a conceptual problem affecting many areas. 
Firstly, conceptually we are trying to predict the gender of a person beind the app, not the gender of an event. So we have a dilemma here, whether to treat each event as a separate observation or to aggregate the events by user_id and create aggregated features. 
Secondly, if we decide to build a model with each event being a separate observation, then we have problem with the definitions of model evaluation metrics. If we predict the same users' generated event's gender correctly for 10 times, do we count 10 True Positives or True Negatives? If no, then the metrics should be redefined.
Moreover, if we want to do hyperparameter tuning with cross validation, we have to be careful with the splits, events that belong to one user should not be in different splits not to elevate the ROC AUC or any other metric on validation set
, as the training data would contain the test data's users' other events, which would make the metrics calculations not fair, as even though the events can be related to different app usage behaviours, nevertheless, the person generating those events is the same, thus there will be extra similarity which model would pick up during the training. 
Therefore, if we deploy that model in production we can expect to see decline in accuracy metrics in production data, where the user pool will be different. 

---


Finding features can can involve various approaches. I would divide them into two groups of approaches:
1. Finding a single feature's individual relation to the target variable.
2. Finding a single feature's relation to the target variable in collaboration with other features.

The first approach includes many metrics including correlation, mutual information value and so on.
The second approach also can use many predefined metrics, as well as algorithms that help select features based on those metrics either by step by step adding features or removing them.
Though my favourite approach is to use feature importance metrics from a model that is trained on those features (example: Gini-importance, entropy).

---


Here I will describe my approach on modeling this problem and the model I would pick.
I would consider aggregating all events by users and presenting them as a single observation, such as user_id becomes a column with unique values.
The feature engineering would include generating features during the aggregations. I would aggregate interaction_with_app variable by (user_id, app_category) pairs, and calculating min, max, avg, std.
(for each user_id features would look like Heath_interaction_with_app_min, Heath_interaction_with_app_max, Heath_interaction_with_app_avg, Heath_interaction_with_app_std, an so on, value would be 0, if there is no specific app usage for a user).
I understand that this will create 10 \* 4 features, but we will get rid most of them in the feature selection phase.
Similarly, I would generate some aggregations for ad categories and clicks. Though, looking at the data tells me that it will mostly be 0, as there are only 17 clicks.
This whole approach could be rejected by me, if I could look at the data and see how many observations are left after the user_id aggregation of course, if the data sample is too small, then I would just get back to modeling event level observations.
Given the data size, I would drop the device_name most probably, but alternatively, I could look for NLP, sentiment analysis libraries that have already trained models, and if there is a masculine/feminine sentiment trained Deep Learning models ready, I would use them to get a score for the masculine/feminine sentiment for the device names.


I would go with a **Random Forest Classifier** as it is quite a stable and also powerful model in general as opposed to boosting models (not always but in my experience). 
It reduces the variance of the predictions as all bagging models tend to do and it does not require heavy hyperparameter tunning, like XGBoost would require. Maybe one could squeeze 1 or 2 pp ROC AUC gain on very well tunned boosting model, but for a quick and stable solution I would choose Random Forest.
No need to consider any Deep Learning models here, as the data size is tiny, and all legacy type GLMs rarely outperform a Random Forest.  

I would use Random Forest twice in the pipeline in two stages.
1. Train a default not very deep (smaller maximum depth) RF model and select features based on the output of importances. Discard all zero importance features, maybe also discard some below some threshold.
2. Having dropped all unnecassary and barely important features (of course also meta features), I would do a Cross Validation to tune hyperparameters with optimizing AUC metric, then train a final model with the best parameters.
