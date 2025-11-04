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
from sklearn.ensemble import StackingClassifier
#################################

## Define Functions
## This is where you will create the features of your model
###########################################################
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

    ## do these steps for machine learning data not for the training data 
    if middle_engineer:
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

## set booleans for different parts of the project ###############################################################################

## This should always run when you are running parts 2-4 
part1 = True ## Trains the model 

## These only need to be run once when done correctly 
part2 = True ## makes predictions on matches between Evan's data and the RLL scraped data 
part3 = True ## makes predictions on the nonperfect matches left over from Evan's data and the census data 
part4 = True ## use the stacked model to make new predictions on the duplicate data from both the scraped data and the census data 

## This should be run when you have finished running all models and are ready to assemble your final matches 
part5 = True ## append the leftover data from the scraped and census matches and keep the best ones 

##################################################################################################################################

if part1:

    ## set a boolean to determine if we are testing parameters 
    test = False

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

    ## set a dummy for engineering middle names 
    middle_engineer = True
    if middle_engineer:
        # change the working directory 
        os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//data//machine_learning_data")
        ## load in the training data 
        data = pd.read_csv("training_data.csv")
        ## convert all the covariates into strings
        data.iloc[:, 1:8] = data.iloc[:, 1:10].astype(str)
        ## drop any missing values
        data = data.dropna(subset=["distance"], how = "any")
        ## compute the features of the data
        data = engineer_features(data)
        print("Training data engineered!")

    middle_engineer = False
    if not middle_engineer:
        # change the working directory 
        os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//matches//first_matches//final_matches//hand_matched_training_data")
        ## load in the training data 
        hand_matched = pd.read_csv("hand_matched_training_data.csv")
        ## convert all the covariates into strings
        hand_matched.iloc[:, 1:10] = hand_matched.iloc[:, 1:10].astype(str)
        ## drop any missing values
        hand_matched = hand_matched.dropna(subset=["distance"], how = "any")
        ## compute the features of the data
        hand_matched = engineer_features(hand_matched)
        print("Hand matched training data engineered!")

    training_data = data.append(hand_matched, ignore_index = True)
    training_data_save_string = "updated_training_data.csv"
    training_data.to_csv(training_data_save_string, index = False)

    ####################################################################################################################################################

    ## Make an X Matrix and a y matrix
    ##################################
    X_final = training_data[[
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
    y_final = training_data['match'].values
    #####################################

    ## Split your data
    ##################
    ## split the data into train and test
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y_final, test_size=0.3, random_state = 42)
    ###################

    ## test new parameters 
    if test:

        ## Last Run Times #######################################
        ## Random Forest Classifier: 117.07
        ## XGB Classifier: 608.46
        ## Gradient Boosting Classifier: Could not finish running
        #########################################################

        ## initialize each base clasifier 
        classifier = [("RandomForestClassifier", RandomForestClassifier(random_state = 42)),
                    ("XGBClassifier", XGBClassifier(seed = 42)),
                    ("GradientBoostingClassifier", GradientBoostingClassifier(random_state = 42))]

        ## make empty dictionaries to save the best parameters for each model 
        xgb_params = {}
        random_forest_params = {}
        gb_boosting_params = {}

        for name, model in classifier:
            ## use cross validation to determine best parameters for each model 
            if name == "XGBClassifier":

                start_time = time.time()
                ## set the grid of parameters to test over
                param_grid = {"max_depth":[2,3,4,5,6,7,8], 
                'learning_rate':[.001, .005, .01, .05, .1, .15, .2], 
                "n_estimators":range(100, 1000, 100)}

                ## set the classifier
                my_classifier = model

                ## set the grid search
                my_classifier_search = GridSearchCV(my_classifier,param_grid,cv=5,return_train_score=True)

                ## train model over the whole grid to look for best parameters 
                my_model = my_classifier_search.fit(X_train_final,y_train_final)

                ## save best parameters 
                xgb_params["max_depth"] = my_model.best_estimator_.get_params()['max_depth']
                xgb_params["learning_rate"] = my_model.best_estimator_.get_params()['learning_rate']
                xgb_params["n_estimators"] = my_model.best_estimator_.get_params()['n_estimators']
                run_time = format(round((time.time() - start_time)/60,2))
                print(name, run_time)
                print("max_depth: ", xgb_params["max_depth"])
                print("learning_rate: ", xgb_params["learning_rate"])
                print("n_estimators: ", xgb_params["n_estimators"])

            if name == "RandomForestClassifier":

                start_time = time.time()
                ## set the grid of parameters to test over
                param_grid = {"max_depth":[2,3,4,5,6,7,8],  
                "n_estimators":range(100, 1000, 100)}

                ## set the classifier
                my_classifier = model

                ## set the grid search
                my_classifier_search = GridSearchCV(my_classifier,param_grid,cv=5,return_train_score=True)

                ## train model over the whole grid to look for best parameters 
                my_model=my_classifier_search.fit(X_train_final,y_train_final)

                ## save best parameters 
                random_forest_params["max_depth"] = my_model.best_estimator_.get_params()['max_depth']
                random_forest_params["n_estimators"] = my_model.best_estimator_.get_params()['n_estimators']
                run_time = format(round((time.time() - start_time)/60,2))
                print(name, run_time)
                print("max_depth: ", random_forest_params["max_depth"])
                print("n_estimators: ", random_forest_params["n_estimators"])

            if name == "GradientBoostingClassifier":

                start_time = time.time()
                ## set the grid of parameters to test over
                param_grid = {"max_depth":[2,3,4,5,6], 
                'learning_rate':[.001, .005, .01, .05, .1, .15, .2],  
                "n_estimators":range(100, 1000, 200)}

                ## set the classifier
                my_classifier = model

                ## set the grid search
                my_classifier_search = GridSearchCV(my_classifier,param_grid,cv=5,return_train_score=True)

                ## train model over the whole grid to look for best parameters 
                my_model = my_classifier_search.fit(X_train_final,y_train_final)

                ## save best parameters 
                gb_boosting_params["max_depth"] = my_model.best_estimator_.get_params()['max_depth']
                gb_boosting_params["n_estimators"] = my_model.best_estimator_.get_params()['n_estimators']
                gb_boosting_params["learning_rate"] = my_model.best_estimator_.get_params()['learning_rate']
                run_time = format(round((time.time() - start_time)/60,2))
                print(name, run_time)
                print("max_depth: ", gb_boosting_params["max_depth"])
                print("learning_rate: ", gb_boosting_params["learning_rate"])
                print("n_estimators: ", gb_boosting_params["n_estimators"])

        ## reinitialize each base clasifier with best paramters
        classifier = [("RandomForestClassifier", RandomForestClassifier(max_depth = random_forest_params["max_depth"],
                    n_estimators = random_forest_params["n_estimators"],
                    random_state = 42)),
                    ("XGBClassifier", XGBClassifier(n_estimators = xgb_params["n_estimators"], 
                    learning_rate = xgb_params["learning_rate"],
                    max_depth = xgb_params["max_depth"],
                    seed = 42)),
                    ("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators = gb_boosting_params["n_estimators"],
                    learning_rate = gb_boosting_params["learning_rate"],
                    max_depth = gb_boosting_params["max_depth"],
                    random_state = 42))]

    else:
        ## here are the results of the last grid search 
        ## we will reinitialize all the models with the best paramters
        classifier = [("RandomForestClassifier", RandomForestClassifier(max_depth = 8,
                    n_estimators = 200,
                    random_state = 42)),
                    ("XGBClassifier", XGBClassifier(n_estimators = 600, 
                    learning_rate = .1,
                    max_depth = 2,
                    seed = 42)),
                    ("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators = 100,
                    learning_rate = .15,
                    max_depth = 3,
                    random_state = 42))]
        ## Current parameters for Gradient Boosting model come from a previous smaller grid search iteration than the one currently being used
        ## Performance likely shouldn't be effected greatly 
        ## each time you rerun the grid search replace the best parameters with new bests if any are found 

    ## initialize the stacked model 
    ## estimaters are the three models we fit, Random Forest, XGBoost, and Gradient Boost
    ## Use Logistic regression as the final estimater since this is classification
    ## stack method uses probalities instead of predictions 
    stacked = StackingClassifier(
        estimators = classifier,
        final_estimator = LogisticRegression(),
        cv = 5,
        stack_method = "predict_proba")

    ## make an empty dictionary to save the trained models to 
    models = {}

    ## Train each of the models you are interested in and see how they are performing
    #################################################################################
    ## define a function to return the confusion matrix
    def get_confusion_matrix_values(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])

    df_results_final = pd.DataFrame(columns=['model', 'accuracy', 'mae', 'precision',
                                        'recall','f1','roc','run_time','tp','fp',
                                        'tn','fn'])

    for name, model in classifier:

        start_time = time.time()
        current_model = model.fit(X_train_final, y_train_final)
        prediction = current_model.predict(X_test_final)
        models[name] = current_model
    
        ## compute performance statistics 
        mae = mean_absolute_error(y_test_final, prediction)
        accuracy = accuracy_score(y_test_final, prediction)
        precision = precision_score(y_test_final, prediction, zero_division=0)
        recall = recall_score(y_test_final, prediction)
        f1 = f1_score(y_test_final, prediction, zero_division=0)
        roc = roc_auc_score(y_test_final, prediction)
        classification = classification_report(y_test_final, prediction, zero_division=0)
        tp, fp, fn, tn = get_confusion_matrix_values(y_test_final, prediction)
        run_time = format(round((time.time() - start_time)/60,2))

        row = {'model': name,
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

    start_time = time.time()

    stacked_model = stacked.fit(X_train_final, y_train_final)    
    stacked_prediction = stacked_model.predict(X_test_final)
    models["Stacked"] = stacked_model

    run_time = format(round((time.time() - start_time)/60,2))
    print("Stacked", run_time)

    ## compute performance statistics 
    mae = mean_absolute_error(y_test_final, stacked_prediction)
    accuracy = accuracy_score(y_test_final, stacked_prediction)
    precision = precision_score(y_test_final, stacked_prediction, zero_division=0)
    recall = recall_score(y_test_final, stacked_prediction)
    f1 = f1_score(y_test_final, stacked_prediction, zero_division=0)
    roc = roc_auc_score(y_test_final, stacked_prediction)
    classification = classification_report(y_test_final, stacked_prediction, zero_division=0)
    tp, fp, fn, tn = get_confusion_matrix_values(y_test_final, stacked_prediction)

    row = {'model': "Stacked",
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

    print(df_results_final)

###############################################################################################################################

if part2:

    middle_engineer = True

    ## set the state we are making predictions on, we will have to do this twice for the rll and the census because we used different naming conventions
    rll_states = ["ALABAMA", "ARIZONA", "ARKANSAS", "CALIFORNIA",
    "COLORADO", "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA",
    "IDAHO", "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY",
    "LOUISIANA", "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN",
    "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA",
    "NEVADA", "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO",
    "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA",
    "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA", "TENNESSEE", 
    "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA",
    "WISCONSIN", "WYOMING", "NEW YORK", "PENNSYLVANIA"]
    rll_states = ["OREGON"]

    ## loop through all states
    for rll_state in rll_states:

        # change the working directory 
        os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//data//machine_learning_data")

        ## Part 2 - Use the XGBoost Model to match Evan's Data to the scraped data ####################################################

        ## loop through all group data
        ## this dataset includes all of evan's data matched to every observation of the census or the scraped RLL data 
        ## census and RLL data include observation's from Oregon and not from Portland
        ## observations were also blocked to only include comparisons with the same first inital and that were within 150 miles of each other for the census
        ## blocking was not used for scraped data 
        ## in our training data the 75th percentile on distance for true matches was within 12 miles of each other. The 95th percentile was 150 miles

        ## Load in the unknown data and make sure the variables are the right types
        ###########################################################################

        ## start with the scraped data since we are prioritizing observations that link directly to the family tree 
        file_string = "rll_" + rll_state + "_groups.csv"
        rll_data = pd.read_csv(file_string)

        rll_data = rll_data.dropna(subset=["lat1", "lat2", "lon1", "lon2", "distance"], how = "any")

        ## convert all the covariates into strings
        rll_data.iloc[:, 0:7] = rll_data.iloc[:, 0:7].astype(str)
        rll_data.iloc[:, 9] = rll_data.iloc[:, 9].astype(str)

        ## engineer the features for the unknown data using your defined function
        #########################################################################
        ## engineer the featurs for evans data
        rll_data = engineer_features(rll_data)
        # print(oregon.head())
        print("Evan's data engineered!")
        #########################################################################

        ## Make an X matrix
        ##################################
        rll_data_x = rll_data[[
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

        ## save your probabilities and add a column for the probability and the max probability of each potential match, keep only observations equal to the max probability of a match
        ###############################################################################################################################################################################

        ## choose the model ##
        name = "XGBClassifier"
        ######################
        ## make predicitions on Evan's data using your chosen model 
        rll_data_predictions = models[name].predict_proba(rll_data_x)[:, 1]
        # print(oregon_predictions[0:10])

        rll_data["score"] = rll_data_predictions
        # print(oregon.head())

        rll_data['score_max'] = rll_data.groupby(["id"])['score'].transform(max)
        # print(oregon.head())

        new_rll_data = rll_data[rll_data["score"] == rll_data["score_max"]]
        # print(new_oregon_data.head())
        ################################################################################################################################################################################

        ## keep only the columns that you want in your final output and save your data 
        ##############################################################################
        new_rll_data = new_rll_data[["first1", "first2", "middle1", "middle2", "last1", "last2", 
                                            "township1", "township2", "county1", "county2", 
                                            "distance", "state", "lat1", "lat2", "lon1", "lon2", 
                                            "first_match", "last_match", "middle_match", "township_match", "county_match",
                                            "first1_count", "first2_count", "last1_count", "last2_count", "township1_count", "township2_count", "county1_count", "county2_count", 
                                            "id", "ark1910", "pid", "score_max", "evan_id"]]
        new_rll_data = new_rll_data.rename(columns = {"score_max":"score"})

        new_rll_data['dup']=new_rll_data.groupby('id')['id'].transform('count')

        ## change the working directory 
        os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//matches//second_matches")

        ## save all of the first matches 
        first_matches_file_string = "rll_" + rll_state + "_first_matches.csv"
        new_rll_data.to_csv(first_matches_file_string, index=False)

        ## save all of the duplicate matches
        rll_data_dup_matches = new_rll_data.loc[new_rll_data["dup"] > 1]
        file_dup_string_rll = "dup_matches//rll_" + rll_state + "_dup_matches.csv"
        rll_data_dup_matches.to_csv(file_dup_string_rll, index = False)

        ## save all of the perfect matches
        rll_data_perfect_matches = new_rll_data.loc[(new_rll_data["score"] >= .999998) & (new_rll_data["dup"] == 1)]
        file_perfect_string_rll = "perfect_matches//rll_" + rll_state + "_perfect_matches.csv"
        rll_data_perfect_matches.to_csv(file_perfect_string_rll, index = False)

        ## save macthes that aren't perfect and aren't duplicated
        rll_data_other_matches = new_rll_data.loc[(new_rll_data["score"] < .999998) & (new_rll_data["dup"] == 1)]
        file_other_string_rll = "other_matches//rll_" + rll_state + "_other_matches.csv"
        rll_data_other_matches.to_csv(file_other_string_rll, index = False)

        print_string = rll_state + " rll matches complete!"
        print(print_string)


###############################################################################################################################

if part3: 

    middle_engineer = True

    ## Part 3 - Use the XGBoost Model to match the other data that were not duplicates or perfect matches into the census #########

    ## name the state we are matching 
    ## there are two lists for each state because the naming conventions of the files were created differently 
    ## leave out New York and Pennsylvanai for right now, they are potentially too big 
    census_states = ["Alabama", "Arizona", "Arkansas", "California",
    "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
    "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
    "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming", "New York", "Pennsylvania"]
    census_states = ["Oregon"]
    rll_states = ["ALABAMA", "ARIZONA", "ARKANSAS", "CALIFORNIA",
    "COLORADO", "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA",
    "IDAHO", "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY",
    "LOUISIANA", "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN",
    "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA",
    "NEVADA", "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO",
    "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA",
    "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA", "TENNESSEE", 
    "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA",
    "WISCONSIN", "WYOMING", "NEW YORK", "PENNSYLVANIA"]
    rll_states = ["OREGON"]
    ## loop through each state 
    for (census_state, rll_state) in zip(census_states, rll_states):

        ## load in the other data 
        os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//matches//second_matches//other_matches")
        file_other_string = "rll_" + rll_state + "_other_matches.csv"
        other = pd.read_csv(file_other_string)

        ## load in the census matches 
        os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//data//machine_learning_data")

        ## name the file path 
        file_string = "census_" + census_state + "_groups.csv"
        census = pd.read_csv(file_string)

        ## drop the missing values 
        census = census.dropna(subset=["lat1", "lat2", "lon1", "lon2", "distance"], how = "any")

        ## convert all the covariates into strings
        census.iloc[:, 0:7] = census.iloc[:, 0:7].astype(str)
        census.iloc[:, 9] = census.iloc[:, 9].astype(str)

        ## keep only the ids that match those from the other ids
        census = census[census['id'].isin(other['id'])]
        ## look at unique values to make sure we saved the right ids
        # print(census_other["id"].unique())

        ## now lets engineer our features 
        census = engineer_features(census)
        # print(oregon.head())
        print("Evan's data engineered!")
        ##################################

        ## Make an X matrix
        ##################################
        census_x = census[[
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

        ## now lets select our model 
        model_name = "XGBClassifier"

        ######################
        ## make predicitions on Evan's data using your chosen model 
        predictions = models[model_name].predict_proba(census_x)[:, 1]
        # print(oregon_predictions[0:10])

        census["score"] = predictions
        # print(census.head())

        census['score_max'] = census.groupby(["id"])['score'].transform(max)
        # print(census.head())

        census = census[census["score"] == census["score_max"]]
        # print(census.head())

        ## keep important variables 
        census = census[["first1", "first2", "middle1", "middle2", "last1", "last2", 
                                                "township1", "township2", "county1", "county2", 
                                                "distance", "state", "lat1", "lat2", "lon1", "lon2", 
                                                "first_match", "last_match", "middle_match", "township_match", "county_match",
                                                "first1_count", "first2_count", "last1_count", "last2_count", "township1_count", "township2_count", "county1_count", "county2_count", 
                                                "id", "ark1910", "score_max", "evan_id"]]

        ## find the number of times every observation appears in the data 
        census['dup']=census.groupby('id')['id'].transform('count')

        ## rename score_max
        census = census.rename(columns = {"score_max":"score"})

        ## change the working directory 
        os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//matches//second_matches")

        ## save all of the first matches 
        first_matches_census_save_string = "census_" + census_state + "_first_matches.csv"
        census.to_csv(first_matches_census_save_string, index=False)

        ## save the perfect matches 
        census_data_perfect_matches = census.loc[(census["score"] >= .999999) & (census["dup"] == 1)]
        file_perfect_string_census = "perfect_matches//census_" + census_state + "_perfect_matches.csv"
        census_data_perfect_matches.to_csv(file_perfect_string_census, index = False)

        ## save the duplicate matches 
        census_data_dup_matches = census.loc[census["dup"] > 1]
        file_dup_string_census = "dup_matches//census_" + census_state +"_dup_matches.csv"
        census_data_dup_matches.to_csv(file_dup_string_census, index = False)

        ## save macthes that aren't perfect and aren't duplicated
        census_data_other_matches = census.loc[(census["score"] < .999999) & (census["dup"] == 1)]
        file_other_string_census = "other_matches//census_" + census_state + "_other_matches.csv"
        census_data_other_matches.to_csv(file_other_string_census, index = False)

        print_string = census_state + " other matches into census complete!"
        print(print_string)

if part4:

    middle_engineer = True

    ## Part 4 - Use the stacked model to make new predictions on the duplicate matches #########

    census_states = ["Alabama", "Arizona", "Arkansas", "California",
    "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
    "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
    "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming", "New York", "Pennsylvania"]
    census_states = ["Oregon"]
    rll_states = ["ALABAMA", "ARIZONA", "ARKANSAS", "CALIFORNIA",
    "COLORADO", "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA",
    "IDAHO", "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY",
    "LOUISIANA", "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN",
    "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA",
    "NEVADA", "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO",
    "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA",
    "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA", "TENNESSEE", 
    "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA",
    "WISCONSIN", "WYOMING", "NEW YORK", "PENNSYLVANIA"]
    rll_states = ["OREGON"]
    ## loop through all states
    for (census_state, rll_state) in zip(census_states, rll_states):

        datasets = ["rll", "census"]

        for data in datasets:

            if data == "census":
                data_string = "dup_matches//" + data + "_" + census_state + "_dup_matches.csv"
                file_string = data + "_" + census_state + "_groups.csv"
            if data == "rll":
                data_string = "dup_matches//" + data + "_" + rll_state + "_dup_matches.csv"
                file_string = data + "_" + rll_state + "_groups.csv"

            ## load in the duplicates 
            os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//matches//second_matches")
            dups_data = pd.read_csv(data_string)
            num_census_dups = len(dups_data.index)

            if num_census_dups > 0:

                ## load in the machine learning data  
                os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//data//machine_learning_data")
                ml_data = pd.read_csv(file_string)

                ## drop the missing values 
                ml_data = ml_data.dropna(subset=["distance"], how = "any")

                ## convert all the covariates into strings
                ml_data.iloc[:, 0:7] = ml_data.iloc[:, 0:7].astype(str)
                ml_data.iloc[:, 9] = ml_data.iloc[:, 9].astype(str)

                ## keep only the ids that match those from the other ids
                ml_data_dups = ml_data[ml_data['id'].isin(dups_data['id'])]
                ## look at unique values to make sure we saved the right ids
                print(ml_data_dups["id"].unique())

                ## now lets engineer our features 
                ml_data_dups = engineer_features(ml_data_dups)
                # print(oregon.head())
                print("Evan's data engineered!")
                ##################################

                ## Make an X matrix
                ##################################
                ml_data_dups_x = ml_data_dups[[
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

                ## now lets select our model 
                model_name = "Stacked" 

                ######################
                ## make predicitions on Evan's data using your chosen model 
                predictions = models[model_name].predict_proba(ml_data_dups_x)[:, 1]
                # print(predictions[0:10])

                ml_data_dups["score"] = predictions
                # print(ml_data_dups.head())

                ml_data_dups['score_max'] = ml_data_dups.groupby(["id"])['score'].transform(max)
                # print(ml_data_dups.head())

                ml_data_dups = ml_data_dups[ml_data_dups["score"] == ml_data_dups["score_max"]]
                # print(ml_data_dups.head())

                ## keep important variables 
                ml_data_dups = ml_data_dups[["first1", "first2", "middle1", "middle2", "last1", "last2", 
                                                        "township1", "township2", "county1", "county2", 
                                                        "distance", "state", "lat1", "lat2", "lon1", "lon2", 
                                                        "first_match", "last_match", "middle_match", "township_match", "county_match",
                                                        "first1_count", "first2_count", "last1_count", "last2_count", "township1_count", "township2_count", "county1_count", "county2_count", 
                                                        "id", "ark1910", "score_max", "evan_id"]]

                ## rename score_max
                ml_data_dups = ml_data_dups.rename(columns = {"score_max":"score"})

                ml_data_dups['dup']=ml_data_dups.groupby('id')['id'].transform('count')

                ## change the working directory 
                os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//matches//second_matches")

                if data == "rll":
                    ## name the save file string 
                    save_string = "dup_matches//" + data + "_" + rll_state + "_dups_stacked_matches.csv"
                    ## save the new matches 
                    ml_data_dups.to_csv(save_string, index = False)
                if data == "census":
                    ## name the save file string 
                    save_string = "dup_matches//" + data + "_" + census_state + "_dups_stacked_matches.csv"
                    ## save the new matches 
                    ml_data_dups.to_csv(save_string, index = False)

                ## save the resolved and unresolved duplicates 
                if data == "rll":
                    save_string = "dup_matches//resolved_dups//" + data + "_" + rll_state + "_resolved_dup_matches.csv"
                    ml_data_dups_resolved = ml_data_dups.loc[ml_data_dups["dup"] == 1]
                    ml_data_dups_resolved.to_csv(save_string, index = False)

                    save_string = "dup_matches//unresolved_dups//" + data + "_" + rll_state + "_unresolved_dup_matches.csv"
                    ml_data_dups_unresolved = ml_data_dups.loc[ml_data_dups["dup"] > 1]
                    ml_data_dups_unresolved.to_csv(save_string, index = False)

                if data == "census":
                    save_string = "dup_matches//resolved_dups//" + data + "_" + census_state + "_resolved_dup_matches.csv"
                    ml_data_dups_resolved = ml_data_dups.loc[ml_data_dups["dup"] == 1]
                    ml_data_dups_resolved.to_csv(save_string, index = False)

                    save_string = "dup_matches//unresolved_dups//" + data + "_" + census_state + "_unresolved_dup_matches.csv"
                    ml_data_dups_unresolved = ml_data_dups.loc[ml_data_dups["dup"] > 1]
                    ml_data_dups_unresolved.to_csv(save_string, index = False)
        print_string = census_state + " duplicated applied to stacked model!"
        print(print_string)


if part5:

    ## Part 5 - Append all the leftover matches between the scraped and census data and keep the best matches 

    census_states = ["Alabama", "Arizona", "Arkansas", "California",
    "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
    "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
    "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming", "New York", "Pennsylvania"]
    census_states = ["Oregon"]
    rll_states = ["ALABAMA", "ARIZONA", "ARKANSAS", "CALIFORNIA",
    "COLORADO", "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA",
    "IDAHO", "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY",
    "LOUISIANA", "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN",
    "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA",
    "NEVADA", "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO",
    "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA",
    "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA", "TENNESSEE", 
    "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WEST VIRGINIA",
    "WISCONSIN", "WYOMING", "NEW YORK", "PENNSYLVANIA"]
    rll_states = ["OREGON"]
    ## loop through all states 
    for (census_state, rll_state) in zip(census_states, rll_states):

        ## load in the leftover census matches 
        os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//matches//second_matches")
        file_other_string_census = "other_matches//census_" + census_state + "_other_matches.csv"
        census_remaining = pd.read_csv(file_other_string_census)
        ## make a new column to indicate which observations are not rll 
        census_remaining["rll"] = 0

        ## load in the rll matches that were leftover after part2
        file_other_string_rll = "other_matches//rll_" + rll_state + "_other_matches.csv"
        rll_remaining = pd.read_csv(file_other_string_rll)
        ## make a new column to indicate which observations are rll 
        rll_remaining["rll"] = 1

        ## load in the perfect census matches and the duplicate census matches 
        file_perfect_string_census = "perfect_matches//census_" + census_state + "_perfect_matches.csv"
        census_perfect = pd.read_csv(file_perfect_string_census)
        census_perfect["rll"] = 0
        file_dup_string_census = "dup_matches//census_" + census_state + "_dup_matches.csv"
        census_duplicate = pd.read_csv(file_dup_string_census)

        ## keep only the observations from the remaining rll data that are not ids in the perfect or duplicate census
        ## we have to do it this way because the full census matches dataset does not contain potential matches for every observation of Evan's data
        common_perfect = rll_remaining.merge(census_perfect, on = ["id"])
        rll_remaining_2 = rll_remaining[(~rll_remaining.id.isin(common_perfect.id))]

        common_dup = rll_remaining.merge(census_duplicate, on = ["id"])
        rll_remaining_3 = rll_remaining_2[(~rll_remaining_2.id.isin(common_dup.id))]

        ## concatenate the two dataframes together 
        all_remaining = pd.concat([rll_remaining_3, census_remaining])

        ## group by score and keep the best score between the two data sets 
        all_remaining['score_max'] = all_remaining.groupby(["id"])['score'].transform(max)
        all_remaining = all_remaining[all_remaining["score"] == all_remaining["score_max"]]

        ## generate a variable to see if there are duplicates left over 
        all_remaining['dup'] = all_remaining.groupby('id')['id'].transform('count')

        ## we are prioritizing the rll data so if there are duplicates lets keep the rll observation since it has a direct link to the family tree
        all_remaining = all_remaining.drop(all_remaining[(all_remaining['rll'] == 0) & (all_remaining["dup"] > 1)].index) 

        # all_remaining.to_csv("test.csv")

        ## append the perfect matches and the resolved duplicates 
        file_perfect_string_rll = "perfect_matches//rll_" + rll_state + "_perfect_matches.csv"
        rll_perfect = pd.read_csv(file_perfect_string_rll)
        rll_perfect["rll"] = 1

        final_matches = pd.concat([all_remaining, rll_perfect])
        final_matches = pd.concat([final_matches, census_perfect])

        os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//matches//second_matches//dup_matches//resolved_dups")
        datasets = ["rll", "census"]
        for data in datasets:
            if data == "rll":
                data_string = data + "_" + rll_state + "_resolved_dup_matches.csv"
            if data == "census":
                data_string = data + "_" + census_state + "_resolved_dup_matches.csv"
            if os.path.isfile(data_string):
                resolved = pd.read_csv(data_string)
                if data == "rll":
                    resolved["rll"] = 1 
                else:
                    resolved["rll"] = 0
                final_matches = pd.concat([final_matches, resolved])

        final_matches = final_matches.drop(["score_max", "dup", "rll"], axis = 1)
        final_matches_sorted = final_matches.sort_values(by = "score", ascending = False)

        ## save the data
        os.chdir("V://FHSS-JoePriceResearch//RA_work_folders//Jared_Grooms//German_Discrimination//matches//second_matches//final_matches")

        ## save a clean copy of the data 
        final_matches_clean = final_matches_sorted[["first1", "first2", "last1", "last2", "middle1", "middle2",
                                                    "township1", "township2", "county1", "county2", 
                                                    "distance", "lat1", "lat2", "lon1", "lon2",
                                                    "id", "ark1910", "pid", "score", "evan_id"]]

        final_matches_save_string = "final_matches_" + census_state + ".csv"
        final_matches_clean.to_csv(final_matches_save_string, index = False)
        print_string = census_state + " final matches complete!"
        print(print_string)

    
