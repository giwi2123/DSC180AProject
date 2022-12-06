#!/usr/bin/env python
# coding: utf-8

# # Evaluating Fairness-increasing Techniques in ML Modeling 

# ### Abstract

# This report explores fairness in ML modeling by displaying how certain methods can repair the discrimination that protected attributes can cause when predicting outcomes. Within datasets with protected attributes such as gender, age, and race, it is often the case that models learn the discrimination hidden within the training data and make future predictions based on these features. This is problematic because there are often biases in the data that are the direct cause of the discriminatory patterns learned by models. Attempts are made to repair these biases in this paper using a preprocessing, inprocessing, and postprocessing fairness repair technique. With the combination of the three on a single dataset, it is shown that model discrimination can be reduced significantly.

# ### Introduction
# 
# Fairness is the ability of the algorithm to correctly respond to the biases present in the dataset, producing results that will not be discriminatory in the future. This is vital due to the increasing role of machine learning algorithms in many fields of society nowadays: from determining a person's ability to get a credit from the bank, to evaluating a person's tendency to reoffend in criminal court cases. Due to the biases present in historical data due to centuries of discrimination and gender and racial inequaility, modern day algorithms that make use of these data may potentially reinforce on these inequalities. In an attempt to decrease the negative effect of biases in the data, several models have been proposed that strive to increase the fairness potential of the algorithm. The prior work that has been done in this field has introduced several competing approaches to pre-processing of the data, in-processing during the algorithm execution, and post-processing of the resultant predictions; these approaches make use of mathematical and statistical tools to numerically quantify fairness as a function of the dataset.
# 
# The presence of multiple fairness algorithms and notions creates a situation where no single approach is ubiquitously applicable to every possible dataset. Therefore, the purpose of our research is to make use of some existing algorithms on different types of datasets in order to evaluate the ability of the algorithms to work together and adapt to different datasets. In other words, we are striving to explore the possibilities for fairness-increasing pipelines on different datasets by employing different combinations of pre-, in-, and post-processing algorithms introduced by prior researcheres.
# 
# In terms of data, we need to use datasets that include "protected" attributes and have a clear outcome attribute. Protected attributes are variables in the data that are prone to causing discriminatory bias. Some examples of these attributes would be gender, ethnicity, and age. It is important to note that we cannot simply drop these attributes for a few reasons. The first reason is because, while they might cause a model to learn inherent bias, they are still hold crucial value in being informative to an outcome. The second reason for this is because the same bias can be reflected through combinations of other attributes' values. This is what is called a multivalued dependency (or an MVD). This happens when a combination of other attribute values explains the protected attribute and therefore is learned by the model to exhibit bias. The fairness methods we apply in this paper aim to remove biases by addressing these MVDs and reducing the discriminatory strain that the protected attribute(s) themselves have on the resulting model.
# 
# All of the fairness methods we used were very efficient in achieving fairness. However, these algorithms were dataset specific and worked better on certain dataset than others. Initially, we tested out different pre-processing, in-processing, and post-processing methods on four different datasets to descipher how these intervention methods worked in different contexts. However, we soon realized that it was more important to focus on measuring the effectiveness of our chosen methods of intervention on a single dataset to align with the goal of this paper.
# 
# For preprocessing, we decided to use Kam-Cal, which is described in “Through the Data Management Lens: Experimental Analysis and Evaluation of Fair Classification”. This method is also known as reweighing, and enforces demographic parity. In other words, Kam-Cal ensures that the outcome attribute (predicted by the model) is independent of the protected attribute(s). This is done by resampling the training data with weights. Kam-Cal comes with the assumption that there isn’t a dependency between the protected attribute(s) and the outcome label. According to the paper, Since Kam-Cal performs equivalently or better than post-processing techniqiues in efficiency and better than most in-processing approaches, it’s a no brainer to use this simple fairness notion.
# 
# For post processing, we followed the Pleiss method describes in the same paper as the previous methods. This method modifies the tuples after the algorithm has been run through. The method enforces true positive rates across the sensitive groups or false positive rates across the sensitive groups so that a randome sample would have the same rates across all random subset of the groups. In other words, it is ensuring that all samples that we select from have equal rates and we choose equal number of samples from each group.
# 
# ### Literature review
# 
# In order to proceed with our experiments, we used texts from researchers to grasp an idea of how these fairness methods work. We used several papers, most notably “Through the Data Management Lens: Experimental Analysis and Evaluation of Fair Classification”. In the paper, the authors discuss several methods on how to achieve fairness on several datasets using different forms of pre-processing, in-processing, and post-processing intervention.  Post-processing is typically more efficient and scalable than pre- and in-processing methods.  However, pre-processing and in-processing methods, while usually having higher runtimes, generally offer more flexibility in correcting trade-offs. Pre-processing methods manipulate the data before we proceed with our machine learning algorithm; they arise from the idea that many of the biases in the machine learning process actually come from the initial steps of collecting the data. While in-processing methods are model specific, they are the most preferred methods because they allow modification of the classifiers for a certain dataset. Finally, post-processing methods achieve fairness by manipulating the predictions of the machine learning algorithms. In “Capuchin: Causal Database Repair for Algorithmic Fairness”, Salimi et. al discuss the fairness classification method called Capuchin, which achieves fairness through dataset repair by manipulating discriminatory variables.  He begins by giving a background discussing a real-life case study where an automated system reviewing job applicants was biased towards men over women.  He then gives an analysis of the methods he plans to use to correct this discrepancy. This fairness classification method provides a basis for the tests we will be performing, as it provides detailed steps and algorithms behind achieving fairness on the dataset. .... As a method to solve the missingness and incompleteness in the data, "Crab: Learning Certifiabl Fair Predictive Models in the Presence of Selection Bias" discusses approaches that could possibly fix biases from the dataset. The paper describes a method called Consistenct Range Approximation, which aims to find a consistent range for the queries so that the true answer will lie in between the queries performed on the biased data set. In other words, we are creating many possible outcomes from the given biased dataset so that one of the outcomes will be from the true outcome. This method does not only handle the bias in the dataset but also performs better than the processing methods mention in the previous papers. 
# 
# 
# 
# ### Description of data
# 
# In order to evaluate fairness in our algorithm, we require algorithms with some potential for biases being present in the dataset. Therefore, we include several datasets of different natures:
# 
# #### Dataset 1
# 
# The [Civilian Complaints Against the NYC Police Officers](https://www.kaggle.com/datasets/asimislam/civilian-complaints-against-nyc-police-officers?select=allegations_20200726939.csv), available on Kaggle.com. The dataset has been previously cleaned in a separate EDA notebook, [available on Github](https://github.com/alecpanattoni/NYPD-Analysis-of-Racial-Bias-and-Discrimination/blob/main/Cleaning_and_Analysis.ipynb).
# 
# The dataset is a collection of allegations against nypd officers over the last ~35 years. There are just over 33,000 samples in the dataset and 31 columns, after adding a few during EDA. All allegations in the dataset have been investigated by a separate identity known as the Civilian Complaint Review Board (CCRB), who are supposed to be unbiased reviewers of each case. Each allegation has a determination outcome, which says whether or not the officer was guilty of the claim against them. Some of the notable features in this dataset include: complainant gender, complainant ethnicity, officer gender, officer ethnicity, and substantiated. The substantiated column is an example of a column added to the dataset in the cleaning process of the data. The cleaning process is described below:
# 
# The following was done to clean the data. I started by looking through each column to see if missingness was explained in different ways. For example, in the shield_no column, missingness was usually defined as a shield_no of 0 instead of nans, since this is an integer column. Similarly, the precinct column contained an absurd amount of 0s and 1000s. This wouldn't make sense since there are not precincts with these numbers. If I had not filled these with nans, we might have seen much different results with our later analysis of NYC's precincts since there were so many precinct inputs of 0 and 1000. It was found that some of the complainant_ages were below 0, which also doesn't make sense. There were also "Unknown"s in the complainant_ethnicity column. For all of these cases, the mentioned observations were converted to nans. In order to retain the type of each of the integer columns, the columns were converted to type Int8 or Int16. Some of the complainant ages were found to be between ages 1 and 10. This wouldn't make much sense, since the age input should be that of the person filing the complaint. It is hard to imagine a child this young filing a formal complaint and more likely that the parent of a child filed it for them with the child's age (or that it was simply mistyped). For this reason, I was conservative and converted ages 8 and below to nans. Next, I added a few columns to the dataframe for potential future EDA/analysis. One of these columns added was the substantiated column. This column is a series of booleans, which tells whether each allegation was found to be true or not. Next, a column was added to aggregate the month/year columns into one for all-in-one access. 
# 
# This dataset is suitable for the purpose of this paper because it contains protected attributes for the complainant; notably their gender and ethnicity. With this it's very possible that a model could learn existing bias in the data and incorrectly predict future outcomes as a result. In the case of this paper, the model will attempt to predict whether the accused officer will be substantiated for their accusation after being investigated by the board. This dataset is also appropriate to use because of the context of the data. We all know that there always seems to be controversy when it comes to accusations against police officers and how punishments against them are handled. This could very well be the case here. Therefore, this proves to be an excellent dataset choice to experiment with fairness notions and methods.
# 
# There are, however, many missing values in the data. Luckily, there are still several thousand tuples of tuples with no missing values. Since NYC is a very diverse city, it is unlikely that there is underrepresentation occuring. However, due to the long history of discrimination against minorities by police officers, it's possible that the data will contain a significantly higher proportion of minority complainants. 
# 
# #### Dataset 2
# 
# The second dataset is a fictional dataset emulating exam performance scores for a sample of 1000 students. The dataset can be found on [Royce Kimmons website](http://roycekimmons.com/tools/generated_data/exams). The dataset generated contains data on `race/ethnicity` (across fictional groups `A` through `E`), `gender`, level of education and level of preparation received. The dataset, while fictional, fits the purpose of testing the fairness of the algorithm by using `gender` and `race/ethnicity` as sensitive groups. The data was not edited beyond renaming columns and combining the Math, Reading and Writing scores into a single `composite` score to act as a prediction value. In `race/ethnicity`, Groups A and B are chosen as privileged and the others are chosen as underpriveleged.
# 
# #### Dataset 3
# 
# The third dataset is a dataset pulled from [Kaggle](https://www.kaggle.com/datasets/vikasp/loadpred), which is a dataset that analyzes the credit scores of different subjects and their likelihood of being approved or denied a loan, based on a number of factors such as their gender, if they are educated, self-employed, and other factors.  While this dataset is assumed to be fictional given no source is provided as well, it has a number of sensitive variables such as gender and education, which can impact fairness.  I used the dataset train_AV3 and re-named it to credit risk to avoid confusion.
# 
# #### Dataset 4
# 
# Last but not least, we chose a dataset on wages of people found on [Kaggle](https://www.kaggle.com/datasets/mastmustu/income). The dataset provides information about race, gender, ethnicity, education and salary on random individuals. The dataset is assumed to be fictional, as no source is provided, however, the dataset has variables such as race and gender which can be determined as sensitive groups on measuring the wages of the individuals. Columns 'education','race','gender','hours-per-week','fnlwgt' were used as the variables for testing fairness. 'fnlwgt' was renamed to 'salary' which is the prediction variables for our dataset. The test was mostly performed on manipulating the 'gender' and 'race' column which seemed the most discriminatory on determining the wage of inidividuals and leads to most biases in the dataset where performing the machine learning algorithms. 

# Note: While there are descriptions of each student's dataset described above, each student has only included their own code for their own dataset in the notebook for the time being. Therefore, there is only the data described in "Dataset 1" included in the coding section.

# 
# 

# ### Code

# In[1]:


import pandas as pd
import numpy as np
import aif360
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from sklearn.model_selection import train_test_split
from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.metrics import average_odds_difference
from aif360.sklearn.metrics import equal_opportunity_difference


# In[2]:


allegations = pd.read_csv("allegations.csv").drop(columns = "Unnamed: 0")
print(allegations.shape)
allegations.head()


# In[3]:


allegations["complainant_gender"].value_counts(normalize=True)


# We will want to test fairness on whether the gender of those accusing nypd officers of misconduct has a connection to whether the accused officer was substantiated for such accusation. The hypothesis is that the model will treat women and/or minorities that accused nypd officers of wrongdoings unfairly. Not only are women underrepresented in the data (as seen above), but it is also possible that the model will pick up on the possibility that the board's disposition for each case is influenced by the complainant's gender. It goes without saying why the complainant's ethnicity might also have an impact on the model being unfair for the latter reason.

# In[4]:


allegations.columns


# We're left with the following columns:

# In[5]:


allegations.isna().sum() / allegations.shape[0]


# There is a shocking 12.5% of the missing protected attribute "complainant_gender". Normally, we would impute this feature, but this violates fairness and adds potentially false values to the data, which can impact the resulting model. Since we still have a significant amount of nonmissing data within the protected attribute, we can drop the ~4000 missing tuples.

# ## ML Task without Fairness Algorithms

# In[6]:


# storring accyracies for plotting later
accuracies = []
eqopp = []


# In[7]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


# In[8]:


allegations = allegations.dropna()


# In[9]:


train, test = train_test_split(allegations, test_size=0.2)


# In[10]:


cat = ["mos_ethnicity", "mos_gender", "fado_type", "allegation", "contact_reason"]


# In[11]:


def model(train, test, cats):
    ohe = OneHotEncoder(handle_unknown='ignore')
    
    traincat_df = train[cats]
    # OHE train categorical
    train_ohe = ohe.fit_transform(traincat_df)
    # concat non-cat train features
    train_len = train.shape[0]

    train_num_feats = np.concatenate(
        [np.reshape(train.mos_age_incident.values, (train_len, 1)), 
        np.reshape(train.complainant_age_incident.values, (train_len, 1)),
        np.reshape(train.precinct.values, (train_len, 1))
        ], axis = 1
    )
    # concatenate train OHE features with non-cat features
    train_feats = pd.DataFrame(np.concatenate([train_ohe.todense(), train_num_feats], axis = 1))
    train_feats['complainant_ethnicity'] = (train['complainant_ethnicity'] == "White").tolist()
    train_feats['complainant_gender'] = (train['complainant_gender'] == "Male").tolist()
    y_train = train.Substantiated.values.astype('int')
    
    mod = LogisticRegression(C = 1.0, class_weight='balanced')
    mod.fit(train_feats, y_train)
    
    testcat_df = test[cats]
    # OHE train categorical
    test_ohe = ohe.transform(testcat_df)
    # concat non-cat train features
    test_len = test.shape[0]

    test_num_feats = np.concatenate(
        [np.reshape(test.mos_age_incident.values, (test_len, 1)), 
        np.reshape(test.complainant_age_incident.values, (test_len, 1)),
        np.reshape(test.precinct.values, (test_len, 1))
        ], axis = 1
    )
        
    # concatenate test OHE features with non-cat features
    test_feats = pd.DataFrame(np.concatenate([test_ohe.todense(), test_num_feats], axis = 1))
    test_feats['complainant_ethnicity'] = (test['complainant_ethnicity'] == "White").tolist()
    test_feats['complainant_gender'] = (test['complainant_gender'] == "Male").tolist()
    y_test = test.Substantiated.values.astype('int')
    
    pred = mod.predict(test_feats)
    
    fp = sum([1 if (y_test[i] == 0) and (pred[i] == 1) else 0 for i in range(len(y_test))]) / len(y_test)
    fn = sum([1 if (y_test[i] == 1) and (pred[i] == 0) else 0 for i in range(len(y_test))]) / len(y_test)
    parity = statistical_parity_difference(pd.Series(y_test), pd.Series(pred))
    odds = average_odds_difference(pd.Series(y_test), pd.Series(pred))
    opportunity = equal_opportunity_difference(pd.Series(y_test), pd.Series(pred))
    
    
    print("false positive rate: " + str(fp))
    print("false negative rate: " + str(fn))
    print("statistical parity: " + str(parity))
    print("Equality of odds: " + str(odds))
    print("Equality of opportunity: " + str(opportunity))
    eqopp.append(opportunity)
    
    
    return pred, mod.score(test_feats, y_test)


# In[12]:


initial_accuracy = model(train, test, cat)[1]
accuracies.append(initial_accuracy)
initial_accuracy


# In[13]:


allegations.head(5)


# Now, a AI360 dataset object needs to be constructed from our original dataset. To do this, we need to define a few things, including which values in columns are considered "privileged". Since being White could give priviledge to either a complainant (board disposition could be more likely to substantiate the officer compared to if the accuser is a minority) or an officer (board disposition could be less likely to substantiate the officer if they are White), we can put white down as a priviledged value for both columns. "Male" in the protected attribute is also considered priviledged for obvious reasons.

# In[14]:


protected = ["complainant_gender", "complainant_ethnicity"]
privileged = [["Male"], ["White"]]


# We also need to define all of the categorical features, so that they can be OneHot-Encoded.

# In[15]:


allegations.dtypes


# In[16]:


def make_ai360_object(df, pred, protected, privileged, categorical, favorable_func):
    """
    creates an ai360 dataframe object. The parameters are the following:
    df: pandas dataframe object
    pred: str of label outcome column 
    protected: list of columns of protected attributes
    privileged: list of lists, where each internal list contains the value(s) considered "privileged" for each
                of the listed protected attributes
    categorical: list of categorical columns as strings 
    favorable_func: simple lambda function entailing which values of pred indicate discrimination
    """
    df_obj = aif360.datasets.StandardDataset(
        df,
        label_name = pred,
        favorable_classes = favorable_func,
        protected_attribute_names = protected,
        privileged_classes = privileged,
        categorical_features = categorical
    )
    return df_obj


# In[17]:


test.head()


# In[18]:


aif_train = make_ai360_object(train, "Substantiated", protected, privileged, cat, lambda x: x == False)
aif_test = make_ai360_object(test, "Substantiated", protected, privileged, cat, lambda x: x == False)


# Now we must define what classes combinations are considered priviledged and which are considered unpriviledged.

# In[19]:


privileged_groups = [{'complainant_gender': 1, 'complainant_ethnicity': 1}]
unprivileged_groups = [{'complainant_gender': 1, 'complainant_ethnicity': 0},
                        {'complainant_gender': 0, 'complainant_ethnicity': 1},
                        {'complainant_gender': 0, 'complainant_ethnicity': 0}]


# For our preprocessing technique, we will use reweighing.

# In[20]:


rw = Reweighing(unprivileged_groups=unprivileged_groups,
                   privileged_groups=privileged_groups)


# In[21]:


rw.fit(aif_train)


# In[22]:


transf_train = rw.transform(aif_train)
transf_test = rw.transform(aif_test)


# In[23]:


transftrain_df, _ = transf_train.convert_to_dataframe(de_dummy_code=True)
transftest_df, _ = transf_test.convert_to_dataframe(de_dummy_code=True)


# In[24]:


transftrain_df["weights"] = transf_train.instance_weights
transftest_df["weights"] = transf_test.instance_weights


# In[25]:


def pre_proc_model(train, test, cats):
    ohe = OneHotEncoder(handle_unknown='ignore')
    
    traincat_df = train[cats]
    # OHE train categorical
    train_ohe = ohe.fit_transform(traincat_df)
    # concat non-cat train features
    train_len = train.shape[0]
    
    train_num_feats = np.concatenate(
        [np.reshape(train.mos_age_incident.values, (train_len, 1)), 
        np.reshape(train.complainant_age_incident.values, (train_len, 1)),
        np.reshape(train.precinct.values, (train_len, 1)),
        np.reshape(train.weights.values, (train_len, 1))
        ], axis = 1
    )

    # concatenate train OHE features with non-cat features
    train_feats = pd.DataFrame(np.concatenate([train_ohe.todense(), train_num_feats], axis = 1))
    train_feats['complainant_ethnicity'] = (train['complainant_ethnicity'] == "White").tolist()
    train_feats['complainant_gender'] = (train['complainant_gender'] == "Male").tolist()
    y_train = train.Substantiated.values.astype('int')
    
    mod = LogisticRegression(C = 3.0, class_weight='balanced')
    mod.fit(train_feats, y_train)
    
    testcat_df = test[cats]
    # OHE train categorical
    test_ohe = ohe.transform(testcat_df)
    # concat non-cat train features
    test_len = test.shape[0]
    
    test_num_feats = np.concatenate(
        [np.reshape(test.mos_age_incident.values, (test_len, 1)), 
        np.reshape(test.complainant_age_incident.values, (test_len, 1)),
        np.reshape(test.precinct.values, (test_len, 1)),
        np.reshape(test.weights.values, (test_len, 1))
        ], axis = 1
    )
        
    # concatenate test OHE features with non-cat features
    test_feats = pd.DataFrame(np.concatenate([test_ohe.todense(), test_num_feats], axis = 1))
    test_feats['complainant_ethnicity'] = (test['complainant_ethnicity'] == "White").tolist()
    test_feats['complainant_gender'] = (test['complainant_gender'] == "Male").tolist()
    
    y_test = test.Substantiated.values.astype('int')
    pred = mod.predict(test_feats)
    
    fp = sum([1 if (y_test[i] == 0) and (pred[i] == 1) else 0 for i in range(len(y_test))]) / len(y_test)
    fn = sum([1 if (y_test[i] == 1) and (pred[i] == 0) else 0 for i in range(len(y_test))]) / len(y_test)
    parity = statistical_parity_difference(pd.Series(y_test), pd.Series(pred))
    odds = average_odds_difference(pd.Series(y_test), pd.Series(pred))
    opportunity = equal_opportunity_difference(pd.Series(y_test), pd.Series(pred))
    
    acc = np.mean([1 if pred[i] == y_test[i] else 0 for i in range(len(pred))])
    
    print("false positive rate: " + str(fp))
    print("false negative rate: " + str(fn))
    print("statistical parity: " + str(parity))
    print("Equality of odds: " + str(odds))
    print("Equality of opportunity: " + str(opportunity))
    eqopp.append(opportunity)
    
    return pred, acc


# In[26]:


subpred, pre_accuracy = pre_proc_model(transftrain_df, transftest_df, cat)
accuracies.append(pre_accuracy)
pre_accuracy


# In[27]:


# in-proc is in a separate notebook, got the value here
eqopp.append(0.0)


# In[ ]:





# In[28]:


def post_proc_model(train, test, cats):
    ohe = OneHotEncoder(handle_unknown='ignore')
    
    traincat_df = train[cats]
    # OHE train categorical
    train_ohe = ohe.fit_transform(traincat_df)
    # concat non-cat train features
    train_len = train.shape[0]

    train_num_feats = np.concatenate(
        [np.reshape(train.mos_age_incident.values, (train_len, 1)), 
        np.reshape(train.complainant_age_incident.values, (train_len, 1)),
        np.reshape(train.precinct.values, (train_len, 1))
        ], axis = 1
    )
    # concatenate train OHE features with non-cat features
    train_feats = pd.DataFrame(np.concatenate([train_ohe.todense(), train_num_feats], axis = 1))
    train_feats['complainant_ethnicity'] = (train['complainant_ethnicity'] == "White").tolist()
    train_feats['complainant_gender'] = (train['complainant_gender'] == "Male").tolist()
    y_train = train.Substantiated.values.astype('int')
    
    mod = LogisticRegression(C = 8.0, class_weight='balanced')
    mod.fit(train_feats, y_train)
    
    testcat_df = test[cats]
    # OHE train categorical
    test_ohe = ohe.transform(testcat_df)
    # concat non-cat train features
    test_len = test.shape[0]

    test_num_feats = np.concatenate(
        [np.reshape(test.mos_age_incident.values, (test_len, 1)), 
        np.reshape(test.complainant_age_incident.values, (test_len, 1)),
        np.reshape(test.precinct.values, (test_len, 1))
        ], axis = 1
    )
        
    # concatenate test OHE features with non-cat features
    test_feats = pd.DataFrame(np.concatenate([test_ohe.todense(), test_num_feats], axis = 1))
    test_feats['complainant_ethnicity'] = (test['complainant_ethnicity'] == "White").tolist()
    test_feats['complainant_gender'] = (test['complainant_gender'] == "Male").tolist()
    y_test = test.Substantiated.values.astype('int')
    
    pred = mod.predict(test_feats)
    
    post = CalibratedEqOddsPostprocessing(
        unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    
    # convert to pandas df with concatenated test features and actual label outcomes
    with_actual = test_feats.assign(Substantiated = y_test)
    
    # revert to non-OHE categorical features
    actual_revert = pd.DataFrame(
        np.hstack(
            [with_actual[["complainant_gender", "complainant_ethnicity", "Substantiated"]].to_numpy(),
            test[cat].to_numpy()]),
        columns = ["complainant_gender", "complainant_ethnicity", "Substantiated"] + cat)
    
    actual_revert['complainant_ethnicity'] = test['complainant_ethnicity'].values
    actual_revert['complainant_gender'] = test['complainant_gender'].values
    actual_revert['Substantiated'] = actual_revert['Substantiated'].astype(bool)
    
    # convert to aif360 df
    actual_test = make_ai360_object(
        actual_revert, "Substantiated", protected, privileged, cat, lambda x: x == False)
    
    # convert to pandas df with concatenated test features and predictions
    with_pred = test_feats.assign(Substantiated = pred)
    
    # revert to non-OHE categorical features
    pred_revert = pd.DataFrame(
        np.hstack(
            [with_pred[["complainant_gender", "complainant_ethnicity", "Substantiated"]].to_numpy(),
            test[cat].to_numpy()]),
        columns = ["complainant_gender", "complainant_ethnicity", "Substantiated"] + cat)
    
    pred_revert['complainant_ethnicity'] = test['complainant_ethnicity'].values
    pred_revert['complainant_gender'] = test['complainant_gender'].values
    pred_revert['Substantiated'] = with_pred['Substantiated'].astype(bool)

    # convert to aif360 df
    pred_test = make_ai360_object(
        pred_revert, "Substantiated", protected, privileged, cat, lambda x: x == False)
    
    # pass into postproc
    post = post.fit_predict(actual_test, pred_test)
    
    # convert test aif360 df back to pandas df 
    testallegations, _ = post.convert_to_dataframe(de_dummy_code=True)
    
    pred = testallegations['Substantiated'].values
    
    
    fp = sum([1 if (y_test[i] == 0) and (pred[i] == 1) else 0 for i in range(len(y_test))]) / len(y_test)
    fn = sum([1 if (y_test[i] == 1) and (pred[i] == 0) else 0 for i in range(len(y_test))]) / len(y_test)
    parity = statistical_parity_difference(pd.Series(y_test), pd.Series(pred))
    odds = average_odds_difference(pd.Series(y_test), pd.Series(pred))
    opportunity = equal_opportunity_difference(pd.Series(y_test), pd.Series(pred))
    
    acc = np.mean([1 if pred[i] == y_test[i] else 0 for i in range(len(pred))])
        
    print("false positive rate: " + str(fp))
    print("false negative rate: " + str(fn))
    print("statistical parity: " + str(parity))
    print("Equality of odds: " + str(odds))
    print("Equality of opportunity: " + str(opportunity))
    eqopp.append(opportunity)
    
    
    return pred, acc


# In[ ]:





# In[29]:


subpred, post_accuracy = post_proc_model(train, test, cat)
accuracies.append(post_accuracy)
post_accuracy


# ### Visualizing the accuracy

# In[30]:


import matplotlib.pyplot as plt


# In[31]:


labels = ['no intervention', 'pre-processing (Reweighing)', 'in-processing (Adversarial Debiasing)', 'post-processing (Calibrated Equality Odds)']


# In[33]:


plt.figure(figsize = (20, 10))
plt.title('Equality of Opportunity after each intervention techniques applied')
plt.xlabel('Intervention technique')
plt.ylabel('Equality of Opportunity')
plt.plot(labels, eqopp, marker='.', markersize = 50)


# In[ ]:




