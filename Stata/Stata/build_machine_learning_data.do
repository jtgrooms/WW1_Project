// Build Machine Learning Data
// Written by Jared Grooms 
// 06/28/2023

///////////////////////////////////////////////////////////////////////////////////////////////////////
// merge the two data sets and save 1 csv for each observation of evan's data every observation from the census for each state
///////////////////////////////////////////////////////////////////////////////////////////////////////

// set a loal for states for evans data 
local states `" "Alabama" "Arizona" "Arkansas" "California" "Colorado" "Connecticut" "Delaware" "Florida" "Georgia" "Idaho" "Illinois" "Indiana" "Iowa" "Kansas" "Kentucky" "Louisiana" "Maine" "Maryland" "Massachusetts" "Michigan" "Minnesota" "Mississippi" "Missouri" "Montana" "Nebraska" "Nevada" "New Hampshire" "New Jersey" "New Mexico" "New York" "North Carolina" "North Dakota" "Ohio" "Oklahoma" "Pennsylvania" "Rhode Island" "South Carolina" "South Dakota" "Tennessee" "Texas" "Utah" "Vermont" "Virginia" "Washington" "West Virginia" "Wisconsin" "Wyoming" "'
local states "Oregon"

/*
// fix the state variable for every state file because I messed up earlier
foreach state of local states {
	
	use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_census_states\temp_census_`state'.dta"
	drop state2
	gen state2 = "`state'"
	replace state2 = lower(state2)
	save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_census_states\temp_census_`state'.dta", replace 
	
}
*/
/*
// make an id variable for each of the evan data sets because I forgot
foreach state of local states {
	use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_ww1_states\temp_evan_`state'.dta", clear
	gen id = _n
	save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_ww1_states\temp_evan_`state'.dta", replace
}
*/

// split evan's data into individual datasets before assigning possible matches, we have been running out of memory 
do "V:\FHSS-JoePriceResearch\tools\server_stata_installs\geodist_serverinstall.do"

foreach state of local states {
	di "`state'"
	// load in the census data to determine how many observations are inside 
	use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_census_states\temp_census_`state'.dta", clear
	count
	local censusobs = `r(N)'
	
	// load in evan's data to find the number of observations so we know how many times to loop  
	use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_ww1_states\temp_evan_`state'.dta", clear
	count 
	local evanobs = `r(N)'
	// loop through each unique observation of evan's world war 1 data, match it to every observation of the census, block it, then save it
	forvalue i = 1/`evanobs' {
		di `i' "/" `evanobs'
		use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_ww1_states\temp_evan_`state'.dta", clear
		keep if id == `i'
		// duplicate evey observation equal to the number in the census 
		expand `censusobs'
		// make a new id variable to match to every unique observation of the census 
		drop id 
		gen id = _n 
		// merge the two data sets 
		merge 1:1 id using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_census_states\temp_census_`state'.dta", nogen keep(3)

		// compute the distances of all mcomparisons
		geodist lat1 lon1 lat2 lon2, gen(distance) miles 
		
		// compute the number of matches for the current observation 
		quietly count 
		local tempobs = `r(N)'

		// block on first letter of first name 
		if `tempobs' > 10 {
			// generate the first character of both first names and drop observations that don't match
			replace first1 = subinstr(first1, " ", "", 1) if substr(first1, 1, 1) == " "
			replace first1 = subinstr(first1, " ", "", 1) if substr(first1, 1, 1) == " "
			gen last_char1 = substr(last1, 1, 1)
			gen last_char2 = substr(last2, 1, 1)
			gen first_char1 = substr(first1, 1, 1)
			gen first_char2 = substr(first2, 1, 1)
			drop if first_char1 != first_char2
		}
		
		// compute the number of matches for the current observation 
		quietly count 
		local tempobs = `r(N)'
		
		// block on distance over 150 
		if `tempobs' > 10 { 
			drop if distance == .
			drop if distance > 150
		}
		
		// compute the number of matches for the current observation 
		quietly count 
		local tempobs = `r(N)'
		
		// block on first letter of last name  
		if `tempobs' > 10 { 
			drop if last_char1 != last_char2
		}
		
		// compute the number of matches for the current observation 
		quietly count 
		local tempobs = `r(N)'
		
		// block on distance over 125 
		if `tempobs' > 10 { 
			drop if distance > 125
		}

		// drop unneeded variables
		drop last_char1 last_char2 first_char1 first_char2 

		// fix the state variable
		gen state = state1
		drop state1 state2

		// order the variables how we like them
		order first1 first2 last1 last2 township1 township2 county1 county2 distance state id ark1910 

		save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\first_census_merge\census_`state'_`i'", replace

	}
	// append them all together 
	use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\first_census_merge\census_`state'_1", clear
	forvalue x = 2/`evanobs' {
		append using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\first_census_merge\census_`state'_`x'"
	}
	save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\machine_learning_data\census_`state'_groups.dta", replace
	export delimited using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\machine_learning_data\census_`state'_groups.csv", replace 
	
}
*/

////////////////////////////////////////////////////////////////////////////////////////////
// merge the two data sets and save 1 csv for each observation of evan's data to every observation from the RLL data
////////////////////////////////////////////////////////////////////////////////////////////


local states `" "ALABAMA" "ARIZONA" "ARKANSAS" "CALIFORNIA" "COLORADO" "CONNECTICUT" "DELAWARE" "FLORIDA" "GEORGIA" "IDAHO" "ILLINOIS" "INDIANA" "IOWA" "KANSAS" "KENTUCKY" "LOUISIANA" "MAINE" "MARYLAND" "MASSACHUSETTS" "MICHIGAN" "MINNESOTA" "MISSISSIPPI" "MISSOURI" "MONTANA" "NEBRASKA" "NEVADA" "NEW HAMPSHIRE" "NEW JERSEY" "NEW MEXICO" "NEW YORK" "NORTH CAROLINA" "NORTH DAKOTA" "OHIO" "OKLAHOMA" "PENNSYLVANIA" "RHODE ISLAND" "SOUTH CAROLINA" "SOUTH DAKOTA" "TENNESSEE" "TEXAS" "UTAH" "VERMONT" "VIRGINIA" "WASHINGTON" "WEST VIRGINIA" "WISCONSIN" "WYOMING" "'  
local states "Oregon"

/*
// fix the state variable for every state file because I messed up earlier
foreach state of local states {
	
	use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_scraped_states\temp_rll_`state'.dta",clear
	drop state2
	gen state2 = "`state'"
	replace state2 = lower(state2)
	save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_scraped_states\temp_rll_`state'.dta", replace 
	
}
*/

*do "V:\FHSS-JoePriceResearch\tools\server_stata_installs\geodist_serverinstall.do"

foreach state of local states {
	
	di "`state'"
	// load in the scraped data to determine how many observations are inside 
	use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_scraped_states\temp_rll_`state'.dta", clear
	count
	local scrapedobs = `r(N)'
	
	// load in evan's data to find the number of observations so we know how many times to loop
	use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_ww1_states\temp_evan_`state'.dta", clear
	count 
	local evanobs = `r(N)'
	// loop through each unique observation of evan's world war 1 data, match it to every observation of the scraped data, block it, then save it
	forvalue i = 1/`evanobs' {
		di `i' "/" `evanobs'
		use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_ww1_states\temp_evan_`state'.dta", clear
		keep if id == `i'
		// duplicate evey observation equal to the number in the census 
		expand `scrapedobs'
		// make a new id variable to match to every unique observation of the census 
		drop id 
		gen id = _n
		// merge the two data sets 
		merge 1:1 id using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_scraped_states\temp_rll_`state'.dta", nogen keep(3)
		
		// compute the distances of all mcomparisons
		geodist lat1 lon1 lat2 lon2, gen(distance) miles
		
		// compute the number of matches for the current observation 
		quietly count 
		local tempobs = `r(N)'

		// block on first letter of first name 
		if `tempobs' > 10 {
			// generate the first character of both first names and drop observations that don't match
			replace first1 = subinstr(first1, " ", "", 1) if substr(first1, 1, 1) == " "
			replace first1 = subinstr(first1, " ", "", 1) if substr(first1, 1, 1) == " "
			gen last_char1 = substr(last1, 1, 1)
			gen last_char2 = substr(last2, 1, 1)
			gen first_char1 = substr(first1, 1, 1)
			gen first_char2 = substr(first2, 1, 1)
			drop if first_char1 != first_char2
		}
		
		// compute the number of matches for the current observation 
		quietly count 
		local tempobs = `r(N)'
		
		// block on distance over 150 
		if `tempobs' > 10 { 
			drop if distance == .
			drop if distance > 150
		}
		
		// compute the number of matches for the current observation 
		quietly count 
		local tempobs = `r(N)'
		
		// block on first letter of last name  
		if `tempobs' > 10 { 
			drop if last_char1 != last_char2
		}
		
		// compute the number of matches for the current observation 
		quietly count 
		local tempobs = `r(N)'
		
		// block on distance over 125 
		if `tempobs' > 10 { 
			drop if distance > 125
		}

		// drop unneeded variables
		drop last_char1 last_char2 first_char1 first_char2 
		
		// fix the state variable
		gen state = state1
		drop state1 state2
		
		// order the variables how we like them
		order first1 first2 last1 last2 township1 township2 county1 county2 distance state id ark1910 pid
		
		save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\first_scraped_merge\rll_`state'_`i'", replace
		
	}
	// append them all together 
	use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\first_scraped_merge\rll_`state'_1", clear
	forvalue x = 2/`evanobs' {
		append using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\first_scraped_merge\rll_`state'_`x'"
	}
	save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\machine_learning_data\rll_`state'_groups.dta", replace
	export delimited using "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\machine_learning_data\rll_`state'_groups.csv", replace 
	
}

*/


























