# World War 1 Death Matching Project

## Description

The purpose of this project is to take a list of names of US soldiers that died in World War 1 and match them to individuals found on Family Search. Machine Learning methods such as regression trees or boosting algorithms are used in Python to find the most likely matches between WW1 data and US Census data or scraped data from Family Search. The scraped data is always prioritized because it is a list of people we already know died in World War 1, increasing the chances of being a match. 

- *This project was done during my time with the BYU Record Linking Lab and uses propriety data. The modeling steps and process used are free to discuss, but real data to make the code run is not publicly available. Further steps will also be taken on this project after I end my time with the Record Linking Lab, but the current progress of the project is available to share.* 

### Phase 1
This project took place over 3 phases. The first phase was using cross validation to determine the best ML model for the project. Multiple models were trained on data from Oregon until the output looked appropriate. The code for phase 1 is now contained in the python archive folder and is named gdmatch.py. 

### Phase 2
Phase 2 was completed by running a new version of gdmatch.py on the states Alabama, Oregon, and Michigan in order to find the three best matches for each observation of the WW1 data from those states. This code is named gdmatch_kfold_validation_many_matches and is the current code used for phases 2 and 3. The reason for phase 2 is that the current model being used was still struggling on prioritizing matches that had stronger similarities between county and township names. So the decision was made to hand verify the best match between the three best for each observation in order to feed our model better future training data. 

There is a further explanation of the original code written for this phase, gdmatch_kfold_validation, further down in the readme. This file keeps only the best match, resolves “ties”, and relies on the most current training data. It is also important to understand that even though it keeps the “best” match, that does not guarantee it is a match. The code is written to keep match scores, rather than a binary response, so it could still be up to further discretion where the cutoff score should be to determine if the match is correct or not. This is the same for gdmatch_kfold_validation_many_matches. Though this code keeps the best 3 matches does not guarantee any one of those matches is true. From working personally with the data, it seems anything with a score of .999998 or better is a “perfect” match and anything greater than about .97 generally seems like a match.

### Phase 3
Phase 3 then takes the original training data and the new hand-matched training data and reruns the model for all states. Since this phase is also using gdmatch_kfold_validation_many_matches it outputs the top 3 matches for every World War 1 observation. Future work can be done to scrape death dates for all three mathces for every observation and write a rule to choose the match with the most accurate death year (these observations should have died during World War 1).  

## Code

#### Python Code 
*Gdmatch_kfold_validation* – Original python code written for the project. This code was used to obtain only the best match for every World War 1 observation from either the US Census or the Family Search scraped data. It comes in 5 parts, and it first prioritizes the scraped data. It also resolves cases of multiple best matches using a stacked model that is trained on output from a gradient boosting model, xgboost model, and random forest model 
		
*Gdmatch_kfold_validation_many_matches* – Current code for the project. This code doesn’t narrow down the matches to 1 best match but to the three best matches. It also prioritizes the Family Search scraped data first, meaning if we find a perfect match from Family Search, then we keep that match, plus the next two best from the Family Search data and we don’t check the US Census. If there wasn’t a perfect match from Family Search then we look at the top three matches from both the census and Family Search and keep those. This code is flexible, and you can set it to which stage of the matching you are on, either first or second. This will ensure you are using the right training data and are saving to the right file paths. It comes in 4 parts and does not worry about resolving multiple best matches because we are keeping the best 3. Refer to lines 332-346 for settings.

#### Stata Code
*Append Matches* – Take all final matches from the second round of matching and append them together in order to save each pid and scrape death data from Family Search using DataFinder. 

*Build Frequency Features* – Builds all the additional features such as frequency counts from the census for first name, last name, county, township. 

*Build Machine Learning Data* – Builds the final machine learning data by making an individual file for all possible matches of every observation from the World War 1 data to the US Census or the Family Search scraped data. Then block the matches on distance, first initial, and last initial and append all the individual files to have a final machine learning data set for every state for both the US Census matches and the Family Search matches. 

*Make Census State Data* – Builds census observations from each state that qualify as potential matches to the World War 1 data. 

*Make Scraped State Data* – Builds Family Search scraped data observations from each state that qualify as potential matches to the World War 1 data.

*Make Training Data* – Builds the original training data for the model using the 1910 – 1920 census data. 

*Make WW1 State Data* - Builds World War 1 observations from each state that qualify as potential matches to the US Census or the Family Search scraped data.

