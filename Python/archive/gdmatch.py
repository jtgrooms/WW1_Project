#######################################
## World War 1 Machine Learning Matches
## Written by Jared Grooms
## 05/18/2023
#######################################

## Import needed libraries 
##########################
import os
import time
import pandas as pd
import numpy as np
import jellyfish as jf
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
#################################

## Here load in your training data and make sure that the variable types are correct
####################################################################################
# change the working directory 
os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//data//additional_features")

## load in the count data and convert to string types 
first1_counts_data = pd.read_csv("first1_counts.csv")
first1_counts_data.iloc[:,0] = first1_counts_data.iloc[:,0].astype(str)
first2_counts_data = pd.read_csv("first2_counts.csv")
first2_counts_data.iloc[:,0] = first2_counts_data.iloc[:,0].astype(str)
last1_counts_data = pd.read_csv("last1_counts.csv")
last1_counts_data.iloc[:,0] = last1_counts_data.iloc[:,0].astype(str)
last2_counts_data = pd.read_csv("last2_counts.csv")
last2_counts_data.iloc[:,0] = last2_counts_data.iloc[:,0].astype(str)
township1_counts_data = pd.read_csv("township1_counts.csv")
township1_counts_data.iloc[:,0] = township1_counts_data.iloc[:,0].astype(str)
township2_counts_data = pd.read_csv("township2_counts.csv")
township2_counts_data.iloc[:,0] = township2_counts_data.iloc[:,0].astype(str)
county1_counts_data = pd.read_csv("county1_counts.csv")
county1_counts_data.iloc[:,0] = county1_counts_data.iloc[:,0].astype(str)
county2_counts_data = pd.read_csv("county2_counts.csv")
county2_counts_data.iloc[:,0] = county2_counts_data.iloc[:,0].astype(str)

# change the working directory 
os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//data//machine_learning_data")

## load in the training data 
data = pd.read_csv("training_data.csv")

## convert all the covariates into strings
data.iloc[:, 1:9] = data.iloc[:, 1:9].astype(str)

## drop any missing values
data = data.dropna(subset=["lat1", "lat2", "lon1", "lon2", "distance"], how = "any")

print(data.head())
#######################################################################################

## This is where you will create the features of your model. I put everything into a function since you'll likely be calling this on both your training and unknown data
########################################################################################################################################################################
## define a function to return the features of the model
## these are different measures of how similar a string is
## numeric measures are needed to fit a machine learning model
def engineer_features(df):
    df['first1'] = df['first1'].str.lower()
    df['last1'] = df['last1'].str.lower()
    df['first2'] = df['first2'].str.lower()
    df['last2'] = df['last2'].str.lower()
    df['township1'] = df['township1'].str.lower()
    df['township2'] = df['township2'].str.lower()
    df['county1'] = df['county1'].str.lower()
    df['county2'] = df['county2'].str.lower()

    ## make a middle inital variable
    df["middle1"] = df["first1"].str.split(" ").str[1]
    df["middle1"].fillna(df["last1"].str.split(" ").str[0], inplace = True)

    df["middle2"] = df["first2"].str.split(" ").str[1]
    df["middle2"].fillna(df["last2"].str.split(" ").str[0], inplace = True)

    ## if middle initals aren't piecked up from firstname or lastname then we need to set middle back to missing
    df["middle1"].loc[df["middle1"] == df["last1"]] = " "
    df["middle2"].loc[df["middle2"] == df["last2"]] = " "

    ## we're going to drop the potential middle names from the first and last name columns 
    df["first1"] = df["first1"].str.split(" ").str[0]
    df["first2"] = df["first2"].str.split(" ").str[0]

    df["last1_reversed"] = df["last1"].str.split().apply(reversed).apply(' '.join)
    df["last1_new"] = df["last1_reversed"].str.split(" ").str[0]
    df["last1_new_reversed"] = df["last1_new"].str.split().apply(reversed).apply(' '.join)
    df["last1"] = df["last1_new_reversed"]

    df["last2_reversed"] = df["last2"].str.split().apply(reversed).apply(' '.join)
    df["last2_new"] = df["last2_reversed"].str.split(" ").str[0]
    df["last2_new_reversed"] = df["last2_new"].str.split().apply(reversed).apply(' '.join)
    df["last2"] = df["last2_new_reversed"]


    ## make features that are dummy variables for if the strings match exactly or not
    df['first_match'] = (df['first1'] == df['first2']).astype(int)
    df['last_match'] = (df['last1'] == df['last2']).astype(int)
    df['township_match'] = (df['township1'] == df['township2']).astype(int)
    df['county_match'] = (df['county1'] == df['county2']).astype(int)
    df['middle_match'] = (df['middle1'] == df['middle2']).astype(int)

    ##
    df['levenshtein_distance_first'] = df.apply(
        lambda x: jf.levenshtein_distance(x['first1'],
                                          x['first2']), axis=1)
    df['levenshtein_distance_last'] = df.apply(
        lambda x: jf.levenshtein_distance(x['last1'],
                                          x['last2']), axis=1)

    df['levenshtein_distance_township'] = df.apply(
        lambda x: jf.levenshtein_distance(x['township1'],
                                          x['township2']), axis=1)
    df['levenshtein_distance_county'] = df.apply(
        lambda x: jf.levenshtein_distance(x['county1'],
                                          x['county2']), axis=1)
    df['levenshtein_distance_middle'] = df.apply(
        lambda x: jf.levenshtein_distance(x['middle1'],
                                          x['middle2']), axis=1)
    ##
    df['damerau_levenshtein_first'] = df.apply(
        lambda x: jf.damerau_levenshtein_distance(x['first1'],
                                                  x['first2']), axis=1)
    df['damerau_levenshtein_last'] = df.apply(
        lambda x: jf.damerau_levenshtein_distance(x['last1'],
                                                  x['last2']), axis=1)
    df['damerau_levenshtein_township'] = df.apply(
        lambda x: jf.damerau_levenshtein_distance(x['township1'],
                                                  x['township2']), axis=1)
    df['damerau_levenshtein_county'] = df.apply(
        lambda x: jf.damerau_levenshtein_distance(x['county1'],
                                                  x['county2']), axis=1)
    df['damerau_levenshtein_middle'] = df.apply(
        lambda x: jf.damerau_levenshtein_distance(x['middle1'],
                                                  x['middle2']), axis=1)
    ##
    df['hamming_first'] = df.apply(
        lambda x: jf.hamming_distance(x['first1'],
                                      x['first2']), axis=1)
    df['hamming_last'] = df.apply(
        lambda x: jf.hamming_distance(x['last1'],
                                      x['last2']), axis=1)
    df['hamming_township'] = df.apply(
        lambda x: jf.hamming_distance(x['township1'],
                                      x['township2']), axis=1)
    df['hamming_county'] = df.apply(
        lambda x: jf.hamming_distance(x['county1'],
                                      x['county2']), axis=1)
    df['hamming_middle'] = df.apply(
        lambda x: jf.hamming_distance(x['middle1'],
                                      x['middle2']), axis=1)
    ##
    df['jaro_similarity_first'] = df.apply(
        lambda x: jf.jaro_similarity(x['first1'],
                                     x['first2']), axis=1)
    df['jaro_similarity_last'] = df.apply(
        lambda x: jf.jaro_similarity(x['last1'],
                                     x['last2']), axis=1)
    df['jaro_similarity_township'] = df.apply(
        lambda x: jf.jaro_similarity(x['township1'],
                                     x['township2']), axis=1)
    df['jaro_similarity_county'] = df.apply(
        lambda x: jf.jaro_similarity(x['county1'],
                                     x['county2']), axis=1)
    df['jaro_similarity_middle'] = df.apply(
        lambda x: jf.jaro_similarity(x['middle1'],
                                     x['middle2']), axis=1)
    ##
    df['jaro_winkler_similarity_first'] = df.apply(
        lambda x: jf.jaro_winkler_similarity(x['first1'],
                                             x['first2']), axis=1)
    df['jaro_winkler_similarity_last'] = df.apply(
        lambda x: jf.jaro_winkler_similarity(x['last1'],
                                             x['last2']), axis=1)
    df['jaro_winkler_similarity_township'] = df.apply(
        lambda x: jf.jaro_winkler_similarity(x['township1'],
                                             x['township2']), axis=1)
    df['jaro_winkler_similarity_county'] = df.apply(
        lambda x: jf.jaro_winkler_similarity(x['county1'],
                                             x['county2']), axis=1)
    df['jaro_winkler_similarity_middle'] = df.apply(
        lambda x: jf.jaro_winkler_similarity(x['middle1'],
                                             x['middle2']), axis=1)
    ##
    df['ratio_first'] = df.apply(
        lambda x: fuzz.ratio(x['first1'],
                             x['first2']), axis=1)
    df['ratio_last'] = df.apply(
        lambda x: fuzz.ratio(x['last1'],
                             x['last2']), axis=1)
    df['ratio_township'] = df.apply(
        lambda x: fuzz.ratio(x['township1'],
                             x['township2']), axis=1)
    df['ratio_county'] = df.apply(
        lambda x: fuzz.ratio(x['county1'],
                             x['county2']), axis=1)
    df['ratio_middle'] = df.apply(
        lambda x: fuzz.ratio(x['middle1'],
                             x['middle2']), axis=1)
    ##
    df['partial_ratio_first'] = df.apply(
        lambda x: fuzz.partial_ratio(x['first1'],
                                     x['first2']), axis=1)
    df['partial_ratio_last'] = df.apply(
        lambda x: fuzz.partial_ratio(x['last1'],
                                     x['last2']), axis=1)
    df['partial_ratio_township'] = df.apply(
        lambda x: fuzz.partial_ratio(x['township1'],
                                     x['township2']), axis=1)
    df['partial_ratio_county'] = df.apply(
        lambda x: fuzz.partial_ratio(x['county1'],
                                     x['county2']), axis=1)
    df['partial_ratio_middle'] = df.apply(
        lambda x: fuzz.partial_ratio(x['middle1'],
                                     x['middle2']), axis=1)
    ##
    df['token_sort_ratio_first'] = df.apply(
        lambda x: fuzz.token_sort_ratio(x['first1'],
                                        x['first2']), axis=1)
    df['token_sort_ratio_last'] = df.apply(
        lambda x: fuzz.token_sort_ratio(x['last1'],
                                        x['last2']), axis=1)
    df['token_sort_ratio_township'] = df.apply(
        lambda x: fuzz.token_sort_ratio(x['township1'],
                                        x['township2']), axis=1)
    df['token_sort_ratio_county'] = df.apply(
        lambda x: fuzz.token_sort_ratio(x['county1'],
                                        x['county2']), axis=1)
    df['token_sort_ratio_middle'] = df.apply(
        lambda x: fuzz.token_sort_ratio(x['middle1'],
                                        x['middle2']), axis=1)
    ##
    df['token_set_ratio_first'] = df.apply(
        lambda x: fuzz.token_set_ratio(x['first1'],
                                       x['first2']), axis=1)
    df['token_set_ratio_last'] = df.apply(
        lambda x: fuzz.token_set_ratio(x['last1'],
                                       x['last2']), axis=1)
    df['token_set_ratio_township'] = df.apply(
        lambda x: fuzz.token_set_ratio(x['township1'],
                                       x['township2']), axis=1)
    df['token_set_ratio_county'] = df.apply(
        lambda x: fuzz.token_set_ratio(x['county1'],
                                       x['county2']), axis=1)
    df['token_set_ratio_middle'] = df.apply(
        lambda x: fuzz.token_set_ratio(x['middle1'],
                                       x['middle2']), axis=1)
    ##
    df['w_ratio_first'] = df.apply(
        lambda x: fuzz.WRatio(x['first1'],
                              x['first2']), axis=1)
    df['w_ratio_last'] = df.apply(
        lambda x: fuzz.WRatio(x['last1'],
                              x['last2']), axis=1)
    df['w_ratio_township'] = df.apply(
        lambda x: fuzz.WRatio(x['township1'],
                              x['township2']), axis=1)
    df['w_ratio_county'] = df.apply(
        lambda x: fuzz.WRatio(x['county1'],
                              x['county2']), axis=1)
    df['w_ratio_middle'] = df.apply(
        lambda x: fuzz.WRatio(x['middle1'],
                              x['middle2']), axis=1)
    ##
    df['uq_ratio_first'] = df.apply(
        lambda x: fuzz.UQRatio(x['first1'],
                               x['first2']), axis=1)
    df['uq_ratio_last'] = df.apply(
        lambda x: fuzz.UQRatio(x['last1'],
                               x['last2']), axis=1)
    df['uq_ratio_township'] = df.apply(
        lambda x: fuzz.UQRatio(x['township1'],
                               x['township2']), axis=1)
    df['uq_ratio_county'] = df.apply(
        lambda x: fuzz.UQRatio(x['county1'],
                               x['county2']), axis=1)
    df['uq_ratio_middle'] = df.apply(
        lambda x: fuzz.UQRatio(x['middle1'],
                               x['middle2']), axis=1)
    ##
    df['q_ratio_first'] = df.apply(
        lambda x: fuzz.QRatio(x['first1'],
                              x['first2']), axis=1)
    df['q_ratio_last'] = df.apply(
        lambda x: fuzz.QRatio(x['last1'],
                              x['last2']), axis=1)
    df['q_ratio_township'] = df.apply(
        lambda x: fuzz.QRatio(x['township1'],
                              x['township2']), axis=1)
    df['q_ratio_county'] = df.apply(
        lambda x: fuzz.QRatio(x['county1'],
                              x['county2']), axis=1)
    df['q_ratio_middle'] = df.apply(
        lambda x: fuzz.QRatio(x['middle1'],
                              x['middle2']), axis=1)

    ## merge in the count columns 
    df = pd.merge(df, first1_counts_data, on = "first1", how = "left")
    df = pd.merge(df, first2_counts_data, on = "first2", how = "left")
    df = pd.merge(df, last1_counts_data, on = "last1", how = "left")
    df = pd.merge(df, last2_counts_data, on = "last2", how = "left")
    df = pd.merge(df, township1_counts_data, on = "township1", how = "left")
    df = pd.merge(df, township2_counts_data, on = "township2", how = "left")
    df = pd.merge(df, county1_counts_data, on = "county1", how = "left")
    df = pd.merge(df, county2_counts_data, on = "county2", how = "left")

    ## fill in missing values for count data 
    df["first1_count"] = df["first1_count"].fillna(0)
    df["first2_count"] = df["first2_count"].fillna(0)
    df["last1_count"] = df["last1_count"].fillna(0)
    df["last2_count"] = df["last2_count"].fillna(0)
    df["township1_count"] = df["township1_count"].fillna(0)
    df["township2_count"] = df["township2_count"].fillna(0)
    df["county1_count"] = df["county1_count"].fillna(0)
    df["county2_count"] = df["county2_count"].fillna(0)

    ##
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(value=0, inplace=True)
    return df

## compute the features of the data
data = engineer_features(data)
print(data.head())
####################################################################################################################################################

## Calculate the colinearity of your features. This isn't really necessary but is still something I look at
###########################################################################################################

def calculate_vif(df, features):
    vif, tolerance = {}, {}
    # all the features that you want to examine
    for feature in features:
        # extract all the other features you will regress against
        X = [f for f in features if f != feature]
        X, y = df[X], df[feature]
        # extract r-squared from the fit
        r2 = LinearRegression().fit(X, y).score(X, y)

        # calculate tolerance
        tolerance[feature] = 1 - r2
        # calculate VIF
        vif[feature] = 1 / (tolerance[feature])
    # return VIF DataFrame
    return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})

vif = calculate_vif(df=data, features = [
          "levenshtein_distance_first", "levenshtein_distance_last", "levenshtein_distance_township", "levenshtein_distance_county", "levenshtein_distance_middle",
          "damerau_levenshtein_first", "damerau_levenshtein_last", "damerau_levenshtein_township", "damerau_levenshtein_county", "damerau_levenshtein_middle",
          "hamming_first", "hamming_last", "hamming_township", "hamming_county", "hamming_middle",
          "jaro_similarity_first", "jaro_similarity_last", "jaro_similarity_township", "jaro_similarity_county", "jaro_similarity_middle",
          "jaro_winkler_similarity_first", "jaro_winkler_similarity_last", "jaro_winkler_similarity_township", "jaro_winkler_similarity_county", "jaro_winkler_similarity_middle",
          "ratio_first", "ratio_last", "ratio_township", "ratio_county", "ratio_middle",
          "partial_ratio_first", "partial_ratio_last", "partial_ratio_township", "partial_ratio_county", "partial_ratio_middle",
          "token_sort_ratio_first", "token_sort_ratio_last", "token_sort_ratio_township", "token_sort_ratio_county", "token_sort_ratio_middle",
          "token_set_ratio_first", "token_set_ratio_last", "token_set_ratio_township", "token_set_ratio_county", "token_set_ratio_middle",
          "w_ratio_first", "w_ratio_last", "w_ratio_township", "w_ratio_county", "w_ratio_middle",
          "uq_ratio_first", "uq_ratio_last", "uq_ratio_township", "uq_ratio_county", "uq_ratio_middle",
          "q_ratio_first","q_ratio_last", "q_ratio_township", "q_ratio_county", "q_ratio_middle",
          "distance","first_match", "last_match", "county_match", "township_match", "middle_match",
          "first1_count", "first2_count", "last1_count", "last2_count", "township1_count", "township2_count", "county1_count", "county2_count"])
print(vif)

## Make an X Matrix and a y matrix
##################################
X_final = data[[
          "levenshtein_distance_first", "levenshtein_distance_last", "levenshtein_distance_township", "levenshtein_distance_county", "levenshtein_distance_middle",
          "damerau_levenshtein_first", "damerau_levenshtein_last", "damerau_levenshtein_township", "damerau_levenshtein_county", "damerau_levenshtein_middle",
          "hamming_first", "hamming_last", "hamming_township", "hamming_county", "hamming_middle",
          "jaro_similarity_first", "jaro_similarity_last", "jaro_similarity_township", "jaro_similarity_county", "jaro_similarity_middle",
          "jaro_winkler_similarity_first", "jaro_winkler_similarity_last", "jaro_winkler_similarity_township", "jaro_winkler_similarity_county", "jaro_winkler_similarity_middle",
          "ratio_first", "ratio_last", "ratio_township", "ratio_county", "ratio_middle",
          "partial_ratio_first", "partial_ratio_last", "partial_ratio_township", "partial_ratio_county", "partial_ratio_middle",
          "token_sort_ratio_first", "token_sort_ratio_last", "token_sort_ratio_township", "token_sort_ratio_county", "token_sort_ratio_middle",
          "token_set_ratio_first", "token_set_ratio_last", "token_set_ratio_township", "token_set_ratio_county", "token_set_ratio_middle",
          "w_ratio_first", "w_ratio_last", "w_ratio_township", "w_ratio_county", "w_ratio_middle",
          "uq_ratio_first", "uq_ratio_last", "uq_ratio_township", "uq_ratio_county", "uq_ratio_middle",
          "q_ratio_first","q_ratio_last", "q_ratio_township", "q_ratio_county", "q_ratio_middle",
          "distance","first_match", "last_match", "county_match", "township_match", "middle_match",
          "first1_count", "first2_count", "last1_count", "last2_count", "township1_count", "township2_count", "county1_count", "county2_count"]]
y_final = data['match'].values
#####################################

## Split your data
##################
## split the data into train and test
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y_final, test_size=0.3, random_state = 42)
###################

## Train each of the models you are interested in and see how they are performing
#################################################################################
## define a function to return the confusion matrix
def get_confusion_matrix_values(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])

df_results_final = pd.DataFrame(columns=['model', 'accuracy', 'mae', 'precision',
                                   'recall','f1','roc','run_time','tp','fp',
                                   'tn','fn'])

## loop through each model and make predictions on the test data
classifier = {"RandomForestClassifier":RandomForestClassifier(random_state = 42),
              "XGBClassifier":XGBClassifier(n_estimators=1000, learning_rate=0.1, seed = 42),
              "GradientBoostingClassifier":GradientBoostingClassifier(random_state = 42),
              "NeuralNet":MLPClassifier(),
              "Stacked":LogisticRegression(random_state = 42)}
models = {}
hats = {}
stacked_test_data = X_test_final

for key in classifier:

    if key != "Stacked":
        ## use cross validation to determine best parameters for each model 
        #if key == "XGBClassifier":
        #    param_grid = {"max_depth":[2,3,4,5,6], 
        #    'learning_rate':[.001, .005, .01, .05, .1, .15, .2], 
        #    "n_estimators":[20,40,60,80,100]}
        #if key == "RandomForestClassifier":
        #    param_grid = {"max_depth":[2,3,4,5,6],  
        #    "n_estimators":[20,40,60,80,100]}
        #if key == "KNeighborsClassifier":
        #    param_grid = {"n_neighbors":[2,3,4,5,6]}
        #if key == "GradientBoostingClassifier":
        #    param_grid = {"max_depth":[2,3,4,5,6], 
        #    'learning_rate':[.001, .005, .01, .05, .1, .15, .2],  
        #    "n_estimators":[20,40,60,80,100]}

        start_time = time.time()
        ## set the classifier 
        my_classifier = classifier[key]

        ## set the grid search
        #my_classifier_search = GridSearchCV(my_classifier,param_grid,cv=5,return_train_score=True)
        ## train model over the whole grid to look for best parameters 
        #model=my_classifier_search.fit(X_train_final,y_train_final)
        model =  my_classifier.fit(X_train_final, y_train_final) 
        models[key] = model
        ## show the best paramters 
        #if key == "XGBClassifier":
        #    print(key + ": Best max_depth: ",model.best_estimator_.get_params()['max_depth'])
        #    print(key + ": Best learning rate: ",model.best_estimator_.get_params()['learning_rate'])
        #    print(key + ": Best n_estimators: ",model.best_estimator_.get_params()['n_estimators']) 
        #if key == "RandomForestClassifier":
        #    print(key + ": Best max_depth: ",model.best_estimator_.get_params()['max_depth'])
        #    print(key + ": Best n_estimators: ",model.best_estimator_.get_params()['n_estimators'])
        #if key == "KNeighborsClassifier":
        #    print(key + ": Best n_neighbors: ",model.best_estimator_.get_params()['n_neighbors'])
        #if key == "GradientBoostingClassifier":
        #    print(key + ": Best max_depth: ",model.best_estimator_.get_params()['max_depth'])
        #    print(key + ": Best n_estimators: ",model.best_estimator_.get_params()['n_estimators'])
        #    print(key + ": Best learning_rate: ",model.best_estimator_.get_params()['learning_rate'])
        ### make predictions 
        y_pred = model.predict(X_test_final)
        hats[key] = y_pred
        stacked_test_data = np.column_stack((stacked_test_data, hats[key]))

        run_time = format(round((time.time() - start_time)/60,2))
        print(run_time)

        mae = mean_absolute_error(y_test_final, y_pred)
        accuracy = accuracy_score(y_test_final, y_pred)
        precision = precision_score(y_test_final, y_pred, zero_division=0)
        recall = recall_score(y_test_final, y_pred)
        f1 = f1_score(y_test_final, y_pred, zero_division=0)
        roc = roc_auc_score(y_test_final, y_pred)
        classification = classification_report(y_test_final, y_pred, zero_division=0)
        tp, fp, fn, tn = get_confusion_matrix_values(y_test_final, y_pred)

        row = {'model': key,
            'accuracy': accuracy,
            'mae': mae,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc': roc,
            'run_time': run_time,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            }

        row_df = pd.DataFrame([row])
        df_results_final = pd.concat([df_results_final, row_df], ignore_index=True)
    if key == "Stacked":
        
        start_time = time.time()

        ## set the classifier 
        my_classifier = classifier[key]
        ## fit the classifier 
        meta_model =  my_classifier.fit(stacked_test_data, y_test_final) 
        ## predict 
        stacked_pred = meta_model.predict(stacked_test_data)
        ## compute and print the time to run 
        run_time = format(round((time.time() - start_time)/60,2))
        print(run_time)

        ## compute performance statistics 
        mae = mean_absolute_error(y_test_final, stacked_pred)
        accuracy = accuracy_score(y_test_final, stacked_pred)
        precision = precision_score(y_test_final, stacked_pred, zero_division=0)
        recall = recall_score(y_test_final, stacked_pred)
        f1 = f1_score(y_test_final, stacked_pred, zero_division=0)
        roc = roc_auc_score(y_test_final, stacked_pred)
        classification = classification_report(y_test_final, stacked_pred, zero_division=0)
        tp, fp, fn, tn = get_confusion_matrix_values(y_test_final, stacked_pred)

        row = {'model': key,
            'accuracy': accuracy,
            'mae': mae,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc': roc,
            'run_time': run_time,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            }

        row_df = pd.DataFrame([row])
        df_results_final = pd.concat([df_results_final, row_df], ignore_index=True)


## look at the results
print(df_results_final)
#################################################################################

## Part 2 - Use the XGBoost Model to match Evan's Data #########################################################################################

## Pick the model you want to use for your analysis
###################################################
## lets use XGBoost for the final match 
# model = XGBClassifier(n_estimators=1000, learning_rate=0.1).fit(X_train_final, y_train_final)
###################################################

## loop through all group data
## this dataset includes all of evan's data matched to every observation of the census or the scraped RLL data 
## census and RLL data include observation's from Oregon and not from Portland
## observations were also blocked to only include comparisons with the same first inital and that were within 150 miles of each other for the census
## blocking was not used for scraped data 
## in our training data the 75th percentile on distance for true matches was within 12 miles of each other. The 95th percentile was 150 miles

## Load in the unknown data and make sure the variables are the right types
###########################################################################
file_string = "rll_oregon_groups.csv"
oregon = pd.read_csv(file_string)

oregon = oregon.dropna(subset=["lat1", "lat2", "lon1", "lon2", "distance"], how = "any")

## convert all the covariates into strings
oregon.iloc[:, 0:7] = oregon.iloc[:, 0:7].astype(str)
oregon.iloc[:, 9] = oregon.iloc[:, 9].astype(str)

## engineer the features for the unknown data using your defined function
#########################################################################
## engineer the featurs for evans data
oregon = engineer_features(oregon)
print(oregon.head())
#########################################################################

## calculate the vif for the unknown data
#########################################
vif = calculate_vif(df=oregon, features=[
          "levenshtein_distance_first", "levenshtein_distance_last", "levenshtein_distance_township", "levenshtein_distance_county", "levenshtein_distance_middle",
          "damerau_levenshtein_first", "damerau_levenshtein_last", "damerau_levenshtein_township", "damerau_levenshtein_county", "damerau_levenshtein_middle",
          "hamming_first", "hamming_last", "hamming_township", "hamming_county", "hamming_middle",
          "jaro_similarity_first", "jaro_similarity_last", "jaro_similarity_township", "jaro_similarity_county", "jaro_similarity_middle",
          "jaro_winkler_similarity_first", "jaro_winkler_similarity_last", "jaro_winkler_similarity_township", "jaro_winkler_similarity_county", "jaro_winkler_similarity_middle",
          "ratio_first", "ratio_last", "ratio_township", "ratio_county", "ratio_middle",
          "partial_ratio_first", "partial_ratio_last", "partial_ratio_township", "partial_ratio_county", "partial_ratio_middle",
          "token_sort_ratio_first", "token_sort_ratio_last", "token_sort_ratio_township", "token_sort_ratio_county", "token_sort_ratio_middle",
          "token_set_ratio_first", "token_set_ratio_last", "token_set_ratio_township", "token_set_ratio_county", "token_set_ratio_middle",
          "w_ratio_first", "w_ratio_last", "w_ratio_township", "w_ratio_county", "w_ratio_middle",
          "uq_ratio_first", "uq_ratio_last", "uq_ratio_township", "uq_ratio_county", "uq_ratio_middle",
          "q_ratio_first","q_ratio_last", "q_ratio_township", "q_ratio_county", "q_ratio_middle",
          "distance","first_match", "last_match", "county_match", "township_match", "middle_match",
          "first1_count", "first2_count", "last1_count", "last2_count", "township1_count", "township2_count", "county1_count", "county2_count"])

# print(vif)
#########################################

## Make an X matrix and a y matrix
##################################
oregon_x = oregon[[
          "levenshtein_distance_first", "levenshtein_distance_last", "levenshtein_distance_township", "levenshtein_distance_county", "levenshtein_distance_middle",
          "damerau_levenshtein_first", "damerau_levenshtein_last", "damerau_levenshtein_township", "damerau_levenshtein_county", "damerau_levenshtein_middle",
          "hamming_first", "hamming_last", "hamming_township", "hamming_county", "hamming_middle",
          "jaro_similarity_first", "jaro_similarity_last", "jaro_similarity_township", "jaro_similarity_county", "jaro_similarity_middle",
          "jaro_winkler_similarity_first", "jaro_winkler_similarity_last", "jaro_winkler_similarity_township", "jaro_winkler_similarity_county", "jaro_winkler_similarity_middle",
          "ratio_first", "ratio_last", "ratio_township", "ratio_county", "ratio_middle",
          "partial_ratio_first", "partial_ratio_last", "partial_ratio_township", "partial_ratio_county", "partial_ratio_middle",
          "token_sort_ratio_first", "token_sort_ratio_last", "token_sort_ratio_township", "token_sort_ratio_county", "token_sort_ratio_middle",
          "token_set_ratio_first", "token_set_ratio_last", "token_set_ratio_township", "token_set_ratio_county", "token_set_ratio_middle",
          "w_ratio_first", "w_ratio_last", "w_ratio_township", "w_ratio_county", "w_ratio_middle",
          "uq_ratio_first", "uq_ratio_last", "uq_ratio_township", "uq_ratio_county", "uq_ratio_middle",
          "q_ratio_first","q_ratio_last", "q_ratio_township", "q_ratio_county", "q_ratio_middle",
          "distance","first_match", "last_match", "county_match", "township_match", "middle_match",
          "first1_count", "first2_count", "last1_count", "last2_count", "township1_count", "township2_count", "county1_count", "county2_count"]]
###################################

## Use this to know what index the probability of a success is
## substitiute match for your response variable and 1 for whatever value represents a success
#############################################################################################
## check what index match = 1 is at
# print(model.classes_)
#############################################################################################

## save your probabilities and add a column for the probability and the max probability of each potential match, keep only observations equal to the max probability of a match
###############################################################################################################################################################################

## choose the model 
key = "XGBClassifier"
oregon_predictions = models[key].predict_proba(oregon_x)[:, 1]
# print(oregon_predictions[0:10])

oregon["score"] = oregon_predictions
# print(oregon.head())

oregon['score_max'] = oregon.groupby(["id"])['score'].transform(max)
# print(oregon.head())

new_oregon_data = oregon[oregon["score"] == oregon["score_max"]]
# print(new_oregon_data.head())
################################################################################################################################################################################

## keep only the columns that you want in your final output and save your data 
##############################################################################
if file_string == "rll_oregon_groups.csv":
    new_oregon_data = new_oregon_data[["first1", "first2", "middle1", "middle2", "last1", "last2", 
                                       "township1", "township2", "county1", "county2", 
                                       "distance", "state", "lat1", "lat2", "lon1", "lon2", 
                                       "first_match", "last_match", "middle_match", "township_match", "county_match",
                                       "first1_count", "first2_count", "last1_count", "last2_count", "township1_count", "township2_count", "county1_count", "county2_count", 
                                       "id", "ark1910", "pid", "score_max"]]
if file_string == "census_oregon_groups.csv":
    new_oregon_data = new_oregon_data[["first1", "first2", "middle1", "middle2", "last1", "last2", 
                                       "township1", "township2", "county1", "county2", 
                                       "distance", "state", "lat1", "lat2", "lon1", "lon2", 
                                       "first_match", "last_match", "middle_match", "township_match", "county_match",
                                       "first1_count", "first2_count", "last1_count", "last2_count", "township1_count", "township2_count", "county1_count", "county2_count", 
                                       "id", "ark1910", "score_max"]]

new_oregon_data = new_oregon_data.rename(columns = {"score_max":"score"})

#bool_series = new_oregon_data["id"].duplicated(keep = False)
#new_oregon_data["dup"] = 0
#new_oregon_data.loc[new_oregon_data[bool_series]["dup"], new_oregon_data["dup"]] = 1

new_oregon_data['dup']=new_oregon_data.groupby('id')['id'].transform('count')

## change the working directory 
os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//first_matches")

if file_string == "rll_oregon_groups.csv":
    new_oregon_data.to_csv("rll_oregon_scores.csv", index=False)
if file_string == "census_oregon_groups.csv":
    new_oregon_data.to_csv("census_oregon_scores.csv", index=False)
#############################################################################



