
// load in the data 
local states `" "Alabama" "Arizona" "Arkansas" "California" "Colorado" "Connecticut" "Delaware" "Florida" "Georgia" "Idaho" "Illinois" "Indiana" "Iowa" "Kansas" "Kentucky" "Louisiana" "Maine" "Maryland" "Massachusetts" "Michigan" "Minnesota" "Mississippi" "Missouri" "Montana" "Nebraska" "Nevada" "New Hampshire" "New Jersey" "New Mexico" "New York" "North Carolina" "North Dakota" "Ohio" "Oklahoma" "Oregon" "Pennsylvania" "Rhode Island" "South Carolina" "South Dakota" "Tennessee" "Texas" "Utah" "Vermont" "Virginia" "Washington" "West Virginia" "Wisconsin" "Wyoming" "'

foreach state of local states {

	import delimited using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\second_matches\final_matches\final_matches_`state'.csv", clear
	save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\second_matches\final_matches\final_matches_`state'.dta", replace
	
}

use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\second_matches\final_matches\final_matches_Alabama.dta", clear

local states `" "Arizona" "Arkansas" "California" "Colorado" "Connecticut" "Delaware" "Florida" "Georgia" "Idaho" "Illinois" "Indiana" "Iowa" "Kansas" "Kentucky" "Louisiana" "Maine" "Maryland" "Massachusetts" "Michigan" "Minnesota" "Mississippi" "Missouri" "Montana" "Nebraska" "Nevada" "New Hampshire" "New Jersey" "New Mexico" "New York" "North Carolina" "North Dakota" "Ohio" "Oklahoma" "Oregon" "Pennsylvania" "Rhode Island" "South Carolina" "South Dakota" "Tennessee" "Texas" "Utah" "Vermont" "Virginia" "Washington" "West Virginia" "Wisconsin" "Wyoming" "'

foreach state of local states {

	append using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\second_matches\final_matches\final_matches_`state'.dta"
	
}

// merge in the missing pids 
rename pid pid1910
merge m:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\clean_ark_pid_crosswalks\ark1910_pid_clean.dta", nogen keep(1 3)
drop match_score attached
replace pid1910 = pid if pid1910 == ""
drop pid 
rename pid1910 pid
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\second_matches\final_matches\final_matches_appended.dta", replace
keep pid
drop if pid == ""
export delimited using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\second_matches\final_matches\final_matches_pids.csv", replace

// scrape for pids now

// load in the scraped pids 
import delimited using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\second_matches\final_matches\pids_scraped.csv", varn(1) clear 
keep pid gender deathyear deathplace
export delimited using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\second_matches\final_matches\pids_scraped.csv", replace

// merge onto the appended data 
merge 1:m pid using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\second_matches\final_matches\final_matches_appended.dta", nogen keep(1 3)

save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\second_matches\final_matches\final_matches_full_scraped_data.dta", replace

keep if deathyear == ""
keep pid

save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\second_matches\final_matches\pids_missing_deathyears.dta", replace

export delimited using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\matches\second_matches\final_matches\pids_missing_deathyears.csv", replace










