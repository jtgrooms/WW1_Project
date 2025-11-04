// Build Machine Learning Data
// Written by Jared Grooms 
// 05/18/2023

////////////////////////////////////////////////////////////////////////////////////////////
// Produce the data set for Evan that is filtered for towns in Oregon not including Portland 
////////////////////////////////////////////////////////////////////////////////////////////

use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\us_updated.dta", clear
keep firstname surname name status_str residence state_name_spaces geocodehere_county state_name status geocodehere_lat geocodehere_lon
// Rename variables 
rename name county
rename residence township
// Lowercase all observations in firstname surname county township 
for X in any firstname surname county township: replace X = lower(X)
// Count all observations by firstname surname county township 
bys firstname surname county township: gen count = _N
// Replace status_str if you find two occurances of an observation 
replace status_str="killed/accident" if count==2
replace status = 1 if count == 2
// Count observations by firstname surname county township from 1 to n
bys firstname surname county township: gen num = _n
// drop duplicates 
keep if num==1
// drop num and count 
drop num count
// make a new id variable 
gen id2 = _n
// save the data 
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\temp\temp_evan_full", replace

// Limit evan data to just oregon /////////////////////////////////////////////////////////////
keep if state_name_spaces == "Oregon"

// There are many observations missing a first name in this data 
// Lets write some code to fill in the first name filed with the first element of the surname field
// I will split the surname and replace firstname with the first word of surname if there was more than one word in surname. I don't want to replace an empty firstname with a surname 
split surname 
replace firstname = surname2 if firstname == "" & surname2 != "" 
replace surname = surname1
rename surname temp 
drop surname*
rename temp surname 

// I found multiple instances where our county variable was empty but the geocodehere_county variable was not empty 
// lets replace county with geocodehere_county when it is empty 
replace county = geocodehere_county if county == "" & geocodehere_county != ""
drop geocodehere_county

// I found a similar problem with state_name_spaces, so lets fix it 
replace state_name_spaces = state_name if state_name_spaces == "" & state_name != ""
drop state_name

// Lets make some indicator variables to track the percentage of empty observations 
local vars firstname surname county township state_name_spaces
foreach var in `vars' {
	gen `var'_i = 1 if `var' != ""
	replace `var'_i = 0 if `var' == ""
}
// We have the most missing values in county, all other variables are negligible 
// I think cause of death could have potential for linking in the future
// Lets create a string variable based off the values in the integer variable status
gen cause = "ACCIDENT" if status == 4
replace cause = "DISEASE" if status == 3
replace cause = "KILLED IN ACTION" if status == 1
replace cause = "WOUNDS" if status == 2
replace cause = "OTHER" if status == 9

// save this data for now 
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\temp\temp_evan_oregon", replace

// Reformat evan's data for the machine learning model ////////////////////////////////////////
keep firstname surname county township state_name_spaces geocodehere_lat geocodehere_lon

rename firstname first1
rename surname last1
rename county county1
rename township township1
rename state_name_spaces state1
rename geocodehere_lat lat1
rename geocodehere_lon lon1

drop if strpos(township1, "portland") | strpos(township1, "porland")

save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\temp\evan_oregon", replace



/////////////////////////////////////////////////////////////////////////////////////////////////
// Produce the data set for the cenus that is filtered for towns in Oregon not including Portland 
/////////////////////////////////////////////////////////////////////////////////////////////////


// 1910 Census data
// Load in US Census data from the family search refined files 
// Use birth year
use "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_pr_birth_year.dta", clear
// Keep if born between 1885 and 1900 
keep if pr_birth_year>=1885 & pr_birth_year<=1900
// Merge in sex data 
merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_is_male.dta", nogen keep(1 3) 
// Keep if male 
keep if is_male==1
// Drop sex variable 
drop is_male
// Merge in state data 
merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_event_state", nogen keep(1 3) 
// Keep if living in Oregon 
keep if event_state=="Oregon"
// drop state data 
drop event_state
// merge in pids 
merge 1:m ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\fs_hints\ark1910_pid_hints", nogen keep(1 3) 
// clean the pids, i.e drop duplicates 
gsort ark1910 -attached -match_score
drop if ark1910==ark1910[_n-1]
// merge in surnames 
merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_pr_name_surn.dta", nogen keep(1 3)
// merge in given names 
merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_pr_name_gn.dta", nogen keep(1 3)
// merge in birth places
merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_pr_birth_place.dta", nogen keep(1 3)
// merge in county data 
merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_event_county", nogen keep(1 3)
// merge in township data 
merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_event_township", nogen keep(1 3)
// Mutate string variables to be lowercased 
gen surname = lower(pr_name_surn)
gen firstname = lower(pr_name_gn)
gen township = lower(event_township)
gen county = lower(event_county)
// rename township 
rename township township2
// keep mutated variables 
keep surname firstname township2 county pid attached match_score pr_birth_year ark1910
// make an id variable from 1 to n
gen id1 = _n
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\temp\temp_census_oregon", replace

// Reformat for machine learning /////////////////////////////////////////////////////////////////////
keep firstname surname township2 county ark1910
rename firstname first2
rename surname last2
rename county county2
gen state2 = "Oregon"

drop if strpos(township2, "portland") | strpos(township2, "porland")

merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\anc_fs\ark1910_histid1910.dta", nogen keep(3)

merge 1:1 histid1910 using "V:\FHSS-JoePriceResearch\data\census_refined\ipums\1910\histid1910_latlon.dta", nogen keep(3)

rename lat lat2
rename lon lon2 
drop histid1910

gen long id = _n

save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\temp\census_oregon", replace



///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// merge the two data sets and save 1 csv for each observation of evan's data every observation from the census
///////////////////////////////////////////////////////////////////////////////////////////////////////////////


do "V:\FHSS-JoePriceResearch\tools\server_stata_installs\geodist_serverinstall.do"

// use evans data 
use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\temp\evan_oregon.dta", clear

// duplicate evey observation equal to the number in the census 
expand 77617

// sort by all unique variables 
sort first1 last1 county1 township1 state1 lat1 lon1 

// generate an id variable 
by first1 last1 county1 township1 state1 lat1 lon1: gen long id = _n

// merge the two data sets 
merge m:1 id using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\temp\census_oregon.dta", nogen

// sort again by unique variables 
sort first1 last1 county1 township1 state1 lat1 lon1 id

// generate an id variable for each data set 
egen group = group(first1 last1 township1 state1 lat1 lon1)

// compute the distances of all mcomparisons
geodist lat1 lon1 lat2 lon2, gen(distance) miles 

// generate the first character of both first names and drop observations that don't match
gen first_char1 = substr(first1, 1, 1)
gen first_char2 = substr(first2, 1, 1)
drop if first_char1 != first_char2

// block the comparisons 
drop if distance == .
drop if distance > 150

// drop unneeded variables
drop first_char1 first_char2 id 

// rename the id variable
rename group id

// fix the state variable
gen state = state1
drop state1 state2

// order the variables how we like them
order first1 first2 last1 last2 township1 township2 county1 county2 distance state id ark1910

save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\census_oregon_groups.dta", replace

export delimited "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\census_oregon_groups.csv", replace



/////////////////////////////////////////////////////////////////////////////////////////////////
// Produce the data set for the RLL data that is filtered for towns in Oregon not including Portland 
/////////////////////////////////////////////////////////////////////////////////////////////////


// I am not entirely sure how this data set was made. Refer to merge_data.do in the merge_data folder potentially 
use "V:\FHSS-JoePriceResearch\papers\current\wwi_deaths\data\merge_data\temp_rll_oregon_v2.dta", clear

// Reformat the data for machine learning //////////////////////////////////////////////////////////////////////////
rename firstname first2
rename surname last2 
rename county county2 

keep first2 last2 county2 pid township2

gen state2 = "Oregon"

drop if strpos(township2, "portland") | strpos(township2, "porland")

merge m:1 pid using "V:\FHSS-JoePriceResearch\data\census_refined\fs\clean_ark_pid_crosswalks\ark1910_pid_clean.dta", nogen keep(1 3)

merge m:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\anc_fs\ark1910_histid1910.dta", nogen keep(1 3)

merge m:1 histid1910 using "V:\FHSS-JoePriceResearch\data\census_refined\ipums\1910\histid1910_latlon.dta", nogen keep(1 3)

drop attached match_score histid1910 
rename lat lat2 
rename lon lon2

gen long id = _n

save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\temp\rll_oregon.dta", replace

*/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// merge the two data sets and save 1 csv for each observation of evan's data to every observation from the RLL data
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


do "V:\FHSS-JoePriceResearch\tools\server_stata_installs\geodist_serverinstall.do"

// use evans data 
use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\temp\evan_oregon.dta", clear

// duplicate evey observation equal to the number in the census 
expand 507

// sort by all unique variables 
sort first1 last1 county1 township1 state1 lat1 lon1 

// generate an id variable 
by first1 last1 county1 township1 state1 lat1 lon1: gen long id = _n

// merge the two data sets 
merge m:1 id using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\temp\rll_oregon.dta", nogen

// sort again by unique variables 
sort first1 last1 county1 township1 state1 lat1 lon1 id

// generate an id variable for each data set 
egen group = group(first1 last1 township1 state1 lat1 lon1)

// compute the distances of all mcomparisons
geodist lat1 lon1 lat2 lon2, gen(distance) miles 

// generate the first character of both first names and drop observations that don't match
gen first_char1 = substr(first1, 1, 1)
gen first_char2 = substr(first2, 1, 1)
* drop if first_char1 != first_char2

// drop unneeded variables
drop first_char1 first_char2 id

// rename the id variable
rename group id

// fix the state variable
gen state = state1
drop state1 state2

// order the variables how we like them
order first1 first2 last1 last2 township1 township2 county1 county2 distance state id ark1910

save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\rll_oregon_groups.dta", replace

export delimited "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\rll_oregon_groups.csv", replace

*/



