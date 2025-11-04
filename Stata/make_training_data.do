
// use hand matched data from the first round of machine learning 
import delimited using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\first_matches\final_matches\hand_matched_training_data\final_matches_Alabama - final_matches_Alabama.csv", clear

// drop the empty rows 
drop if first1 == ""

// replace missing result values 
replace result = 0 if result == .

// keep important variables 
drop ark1910 pid score evan_id rll auditor 
rename result match

// reorder the variables 
order match first1 first2 middle1 middle2 last1 last2 township1 township2 county1 county2 distance

// save the data 
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\first_matches\final_matches\hand_matched_training_data\Alabama_training_temp.dta", replace

// now do the other two states 
import delimited using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\first_matches\final_matches\hand_matched_training_data\final_matches_Michigan - final_matches_Michigan.csv", clear

// drop the empty rows 
drop if first1 == ""

// replace missing result values 
replace result = 0 if result == .

// keep important variables 
drop ark1910 pid score evan_id rll auditor 
rename result match

// reorder the variables 
order match first1 first2 middle1 middle2 last1 last2 township1 township2 county1 county2 distance

// save the data 
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\first_matches\final_matches\hand_matched_training_data\Michigan_training_temp.dta", replace

import delimited using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\first_matches\final_matches\hand_matched_training_data\final_matches_Oregon - final_matches_Oregon.csv", clear

// drop the empty rows 
drop if first1 == ""

// replace missing result values 
replace v18 = 0 if v18 == .

// keep important variables 
drop ark1910 pid score evan_id rll v17 
rename v18 match

// reorder the variables 
order match first1 first2 middle1 middle2 last1 last2 township1 township2 county1 county2 distance

// save the data 
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\first_matches\final_matches\hand_matched_training_data\Oregon_training_temp.dta", replace

// now append all the training data together 
append using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\first_matches\final_matches\hand_matched_training_data\Michigan_training_temp.dta"
append using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\first_matches\final_matches\hand_matched_training_data\Alabama_training_temp.dta"

export delimited using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\first_matches\final_matches\hand_matched_training_data\hand_matched_training_data.csv", replace
















