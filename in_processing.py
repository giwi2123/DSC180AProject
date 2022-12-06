#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, matplotlib.pyplot as plt
import numpy as np
import aif360
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from sklearn.model_selection import train_test_split
from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.metrics import average_odds_difference
from aif360.sklearn.metrics import equal_opportunity_difference
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
import tensorflow as tf

# from aif360.algorithms.inprocessing import 


# In[2]:


# !pip install aif360


# In[3]:


#%% Import Dataset
df_nypd = pd.read_csv('nypd-1.csv')
print(df_nypd.shape)
df_nypd.head()


# In[4]:


df_nypd.columns


# In[5]:


df_nypd["complainant_gender"].value_counts(normalize=True)


# We will want to test fairness on whether the gender of those accusing nypd officers of misconduct has a connection to whether the accused officer was substantiated for such accusation. The hypothesis is that the model will treat women and/or minorities that accused nypd officers of wrongdoings unfairly. Not only are women underrepresented in the data (as seen above), but it is also possible that the model will pick up on the possibility that the board's disposition for each case is influenced by the complainant's gender. It goes without saying why the complainant's ethnicity might also have an impact on the model being unfair for the latter reason.

# In[6]:


df_nypd.columns


# We're left with the following columns:

# In[7]:


df_nypd.isna().sum() / df_nypd.shape[0]


# In[8]:


for cl,s in zip(df_nypd.isna(), df_nypd.isna().sum()):
    if s!=0:
        print(f" The column  [{cl}] \t\t has {s}  messing values")
print(f"There is a total of {df_nypd.isna().sum().sum()}" " of the missing values ")
    


# There is a shocking 12.5% of the missing protected attribute "complainant_gender". Normally, **we would impute this feature, `but this violates fairness and adds potentially false values to the data`**, which can impact the resulting model. Since we still have a significant amount of nonmissing data within the protected attribute, we can drop the ~4000 missing tuples.

# ## ML Task without Fairness Algorithms

# In[9]:


# storring accyracies for plotting later
accuracies = []
opportunities = []


# In[10]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


# In[11]:


# remove rows with null values
df_nypd.dropna(axis=0, how='any', inplace=True)
# cleanup complainant_gender with some random values like Transman (FTM)
df_nypd.drop(df_nypd.loc[~df_nypd['complainant_gender'].isin(['Male','Female'])].index, inplace=True)


# In[12]:


print(f" Now there is {df_nypd.isna().sum().sum()} messing values")


# In[13]:


# breakdown of the frequencies in the target variable
target_variable = df_nypd['board_disposition'].value_counts()
# encode the target variable
df_nypd.loc[df_nypd.board_disposition == 'Unsubstantiated', 'board_disposition'] = 0
df_nypd.loc[df_nypd.board_disposition == 'Exonerated', 'board_disposition'] = 0
df_nypd.loc[df_nypd.board_disposition == 'Substantiated (Charges)', 'board_disposition'] = 1
df_nypd.loc[df_nypd.board_disposition == 'Substantiated (Formalized Training)', 'board_disposition'] = 1
df_nypd.loc[df_nypd.board_disposition == 'Substantiated (Command Discipline A)', 'board_disposition'] = 1
df_nypd.loc[df_nypd.board_disposition == 'Substantiated (Command Discipline B)', 'board_disposition'] = 1
df_nypd.loc[df_nypd.board_disposition == 'Substantiated (Command Discipline)', 'board_disposition'] = 1
df_nypd.loc[df_nypd.board_disposition == 'Substantiated (Command Lvl Instructions)', 'board_disposition'] = 1
df_nypd.loc[df_nypd.board_disposition == 'Substantiated (Instructions)', 'board_disposition'] = 1
df_nypd.loc[df_nypd.board_disposition == 'Substantiated (No Recommendations)', 'board_disposition'] = 1


# In[14]:


# plt.figure(figsize=(8,6))
plt.hist(df_nypd['board_disposition']);


# In[15]:


df_nypd.drop(labels=['unique_mos_id','first_name','last_name',
                     'complaint_id','allegation','fado_type','contact_reason',
                     'outcome_description','rank_incident','rank_now'],axis=1, inplace=True)


# In[16]:


# Complainant Ethnicity with White are treated as privelleged and others as Unprivelleged
df_nypd.loc[df_nypd.complainant_ethnicity.isin(['White']), 'complainant_ethnicity'] = 1
df_nypd.loc[df_nypd.complainant_ethnicity.isin(['Black','Asian','Refused','Hispanic','Unknown',
                                                'Other Race','American Indian']), 'complainant_ethnicity'] = 0


# In[17]:


df_nypd.dtypes


# In[18]:


# cat = ["mos_ethnicity", "mos_gender", "fado_type", "allegation", "contact_reason"]
# cat = ["mos_ethnicity_American Indian",
# "mos_ethnicity_Asian",
# "mos_ethnicity_Black",
# "mos_ethnicity_Hispanic",
# "mos_ethnicity_White",
# "mos_gender_F",
# "mos_gender_M"]
cat = [
    "command_now",
"command_at_incident",       
"rank_abbrev_incident",      
"rank_abbrev_now",           
"mos_ethnicity",             
"mos_gender"
# ,"complainant_ethnicity"
#     ,"complainant_gender"
]


# In[19]:


train, test = train_test_split(df_nypd, test_size=0.2)


# In[20]:


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
    y_train = train.board_disposition.values.astype('int')
    
    mod = LogisticRegression(C = 1.0, class_weight='balanced',)
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
    y_test = test.board_disposition.values.astype('int')
    
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
    
    
    return pred, mod.score(test_feats, y_test),opportunity


# In[21]:


initial_accuracy = model(train, test, cat)[1]
initial_opp = model(train, test, cat)[2]

opportunities.append(initial_opp)
accuracies.append(initial_accuracy)
print("initial_accuracy",initial_accuracy)


# Now, a AI360 dataset object needs to be constructed from our original dataset. To do this, we need to define a few things, including which values in columns are considered "privileged". Since being White could give priviledge to either a complainant (board disposition could be more likely to substantiate the officer compared to if the accuser is a minority) or an officer (board disposition could be less likely to substantiate the officer if they are White), we can put white down as a priviledged value for both columns. "Male" in the protected attribute is also considered priviledged for obvious reasons.

# In[22]:


protected = ["complainant_gender", "complainant_ethnicity"]
privileged = [["Male"], ["White"]]


# We also need to define all of the categorical features, so that they can be OneHot-Encoded.

# In[23]:


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


# In[24]:


train.dtypes


# In[25]:


aif_train = make_ai360_object(train, "board_disposition", protected, privileged, cat, lambda x: x == False)
aif_test = make_ai360_object(test, "board_disposition", protected, privileged, cat, lambda x: x == False)


# Now we must define what classes combinations are considered priviledged and which are considered unpriviledged.

# In[26]:


privileged_groups = [{'complainant_gender': 1, 'complainant_ethnicity': 1}]
unprivileged_groups = [{'complainant_gender': 1, 'complainant_ethnicity': 0},
                        {'complainant_gender': 0, 'complainant_ethnicity': 1},
                        {'complainant_gender': 0, 'complainant_ethnicity': 0}]


# For our preprocessing technique, we will use reweighing.

# In[27]:


rw = Reweighing(unprivileged_groups=unprivileged_groups,
                   privileged_groups=privileged_groups)


# In[28]:


rw.fit(aif_train)


# In[29]:


transf_train = rw.transform(aif_train)
transf_test = rw.transform(aif_test)


# In[30]:


transftrain_df, _ = transf_train.convert_to_dataframe(de_dummy_code=True)
transftest_df, _ = transf_test.convert_to_dataframe(de_dummy_code=True)


# In[31]:


transftrain_df["weights"] = transf_train.instance_weights
transftest_df["weights"] = transf_test.instance_weights


# In[32]:


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
    y_train = train.board_disposition.values.astype('int')
    
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
    
    y_test = test.board_disposition.values.astype('int')
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
    
    return pred, acc, opportunity


# In[33]:


subpred, pre_accuracy, pre_opp = pre_proc_model(transftrain_df, transftest_df, cat)
accuracies.append(pre_accuracy)
opportunities.append(pre_opp)
pre_accuracy


# In[34]:


from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
def in_proc_model(df):
    #%% Mitigating Bias with AIF 360
    nypd_dataset = BinaryLabelDataset(favorable_label=1,
                                                unfavorable_label=0,
                                                df=df,
                                                label_names=['board_disposition'],
                                                protected_attribute_names=['complainant_ethnicity'])
    nypd_train, nypd_test = nypd_dataset.split([0.9], shuffle=True)
    privileged_groups = [{'complainant_ethnicity': 1}]
    unprivileged_groups = [{'complainant_ethnicity': 0}]
    
    
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    #%% Training the aif360 inprocessing algorithm
    debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                              unprivileged_groups=unprivileged_groups,
                              scope_name='debiased_classifier',
                              debias=True,
                              sess=sess)
    debiased_model.fit(nypd_train)
    
    #%%  Prediction from the inprocessing algorithm
    dataset_debiasing_test = debiased_model.predict(nypd_test)
    #%% Classification metrics
#     classified_metric_debiasing_test = ClassificationMetric(nypd_test,
#                                                      dataset_debiasing_test,
#                                                      unprivileged_groups=unprivileged_groups,
#                                                      privileged_groups=privileged_groups)

#     print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
    
    
#     sess.close()
    
#     #%% Disparate impact after using inprocessing algorithm
#     print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())

    # convert test aif360 df back to pandas df 
    test_df, _ = dataset_debiasing_test.convert_to_dataframe(de_dummy_code=True)
    original_test_df, _ = nypd_test.convert_to_dataframe(de_dummy_code=True)
    
    pred = test_df['board_disposition'].values
    y_test = original_test_df.board_disposition.values.astype('int')
    
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
    
    return pred, acc,opportunity
    


# In[35]:


df_nypd_cp = df_nypd.copy()

categorical_columns = ['month_received','year_received','month_closed','year_closed','mos_age_incident','complainant_age_incident','precinct','shield_no','command_now', 'command_at_incident','rank_abbrev_incident','mos_ethnicity','mos_gender','complainant_gender','rank_abbrev_now']
for column in categorical_columns:
    onehotencoded_features = pd.get_dummies(df_nypd_cp[column], prefix=column)
    df_nypd_cp = df_nypd_cp.drop(column, axis=1)
    df_nypd_cp = df_nypd_cp.join(onehotencoded_features)


# In[36]:


in_proo_pred ,in_acc,in_opp = in_proc_model(df_nypd_cp)
accuracies.append(in_acc)
opportunities.append(in_opp)


# In[37]:


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
    y_train = train.board_disposition.values.astype('int')
    
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
    y_test = test.board_disposition.values.astype('int')
    
    pred = mod.predict(test_feats)
    
    post = CalibratedEqOddsPostprocessing(
        unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    
    # convert to pandas df with concatenated test features and actual label outcomes
    with_actual = test_feats.assign(board_disposition = y_test)
    
    # revert to non-OHE categorical features
    actual_revert = pd.DataFrame(
        np.hstack(
            [with_actual[["complainant_gender", "complainant_ethnicity", "board_disposition"]].to_numpy(),
            test[cat].to_numpy()]),
        columns = ["complainant_gender", "complainant_ethnicity", "board_disposition"] + cat)
    
    actual_revert['complainant_ethnicity'] = test['complainant_ethnicity'].values
    actual_revert['complainant_gender'] = test['complainant_gender'].values
    actual_revert['board_disposition'] = actual_revert['board_disposition'].astype(bool)
    
    # convert to aif360 df
    actual_test = make_ai360_object(
        actual_revert, "board_disposition", protected, privileged, cat, lambda x: x == False)
    
    # convert to pandas df with concatenated test features and predictions
    with_pred = test_feats.assign(board_disposition = pred)
    
    # revert to non-OHE categorical features
    pred_revert = pd.DataFrame(
        np.hstack(
            [with_pred[["complainant_gender", "complainant_ethnicity", "board_disposition"]].to_numpy(),
            test[cat].to_numpy()]),
        columns = ["complainant_gender", "complainant_ethnicity", "board_disposition"] + cat)
    
    pred_revert['complainant_ethnicity'] = test['complainant_ethnicity'].values
    pred_revert['complainant_gender'] = test['complainant_gender'].values
    pred_revert['board_disposition'] = with_pred['board_disposition'].astype(bool)

    # convert to aif360 df
    pred_test = make_ai360_object(
        pred_revert, "board_disposition", protected, privileged, cat, lambda x: x == False)
    
    # pass into postproc
    post = post.fit_predict(actual_test, pred_test)
    
    # convert test aif360 df back to pandas df 
    testallegations, _ = post.convert_to_dataframe(de_dummy_code=True)
    
    pred = testallegations['board_disposition'].values
    
    
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
    
    
    return pred, acc,opportunity


# In[38]:


subpred, post_accuracy,post_opp = post_proc_model(train, test, cat)
accuracies.append(post_accuracy)
opportunities.append(post_opp)
post_accuracy


# In[39]:


accuracies


# ### Visualizing the accuracy

# In[40]:


import matplotlib.pyplot as plt


# In[41]:


labels = ['no intervention', 'pre-processing (Reweighing)', 
          'in-processing (Adversarial Debiasing)', 
          'post-processing (Calibrated Equality Odds)']


# In[72]:


plt.figure(figsize = (20, 10))
plt.title('Accuracies after each intervention techniques applied',fontsize=25)
plt.xlabel('Intervention technique',fontsize=25)
plt.ylabel('Accuracy',fontsize=25)
plt.plot(labels, accuracies, marker='.', markersize = 50)
plt.gca().yaxis.set_tick_params(labelsize=25)
plt.gca().xaxis.set_tick_params(labelsize=15)


# In[76]:


plt.figure(figsize = (20, 10))
plt.title('opportunities after each intervention techniques applied',fontsize=25)
plt.xlabel('Intervention technique',fontsize=25)
plt.ylabel('opportunity',fontsize=25)
plt.plot(labels, np.abs(opportunities), marker='.', markersize = 50)
plt.gca().yaxis.set_tick_params(labelsize=25)
plt.gca().xaxis.set_tick_params(labelsize=15)


# In[61]:


for lab,ac, op in zip(labels,accuracies,np.abs(opportunities)):
    print(f"+ {lab}   give:\t\t accuracy =  {round(ac,5)}  | \t equalty_opportunity=  {round(op,5)}")

