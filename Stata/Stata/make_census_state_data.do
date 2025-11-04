

// make a local for all states 
local states `" "Alabama" "Alaska" "Arizona" "Arkansas" "California" "Colorado" "Connecticut" "Delaware" "Florida" "Georgia" "Hawaii" "Idaho" "Illinois" "Indiana" "Iowa" "Kansas" "Kentucky" "Louisiana" "Maine" "Maryland" "Massachusetts" "Michigan" "Minnesota" "Mississippi" "Missouri" "Montana" "Nebraska" "Nevada" "New Hampshire" "New Jersey" "New Mexico" "New York" "North Carolina" "North Dakota" "Ohio" "Oklahoma" "Pennsylvania" "Rhode Island" "South Carolina" "South Dakota" "Tennessee" "Texas" "Utah" "Vermont" "Virginia" "Washington" "West Virginia" "Wisconsin" "Wyoming" "'
di `states'
local states "Oregon"

foreach state of local states {
	
	if "`state'" == "Oregon"{
		local townships "portland"
	}
	if "`state'" == "New York" {
		local townships `" "new york" "buffalo" "rochester" "syracuse" "albany" "yonkers" "troy" "utica" "schenectady" "'
	}
	if "`state'" == "Illinois" {
		local townships `" "chicago" "peoria" "east st. louis"  "'
	}
	if "`state'" == "Pennsylvania" {
		local townships `" "philadelphia" "pittsburgh" "scranton" "reading" "youngstown" "wilkes-barre" "erie" "harrisburg" "johnstown" "'
	}
	if "`state'" == "Missouri" {
		local townships `" "st. louis" "kansas city" "st. joseph"  "'
	}
	if "`state'" == "Massachusetts" {
		local townships `" "boston" "worcester" "fall river" "lowell" "cambridge" "new bedford" "lynn" "springfield" "lawrence" "somerville" "holyoke" "brockton" "'
	}
	if "`state'" == "Ohio" {
		local townships `" "cleveland" "cincinnati" "columbus" "toledo" "dayton" "akron"  "'
	}
	if "`state'" == "Maryland" {
		local townships "baltimore"
	}
	if "`state'" == "Michigan" {
		local townships `" "detroit" "grand rapids"  "'
	}
	if "`state'" == "California" {
		local townships `" "san francisco" "los angeles" "oakland"  "'
	}
	if "`state'" == "Wisconsin" {
		local townships "milwaukee"
	}
	if "`state'" == "New Jersey" {
		local townships `" "newark" "jersey city" "paterson" "trenton" "camden" "elizabeth" "hoboken" "bayonne" "passaic" "'
	}
	if "`state'" == "Louisiana" {
		local townships "new orleans"
	}
	if "`state'" == "Minnesota" {
		local townships `" "minneapolis" "st. paul" "duluth"  "'
	}
	if "`state'" == "Washington" {
		local townships `" "seattle" "spokane" "tacoma""'
	}
	if "`state'" == "Indiana" {
		local townships `" "indianapolis" "evansville" "fort wayne" "terre haute" "south bend" "'
	}
	if "`state'" == "Rhode Island" {
		local townships "providence"
	}
	if "`state'" == "Kentucky" {
		local townships "louisville"
	}
	if "`state'" == "Colorado" {
		local townships "denver"
	}
	if "`state'" == "Maine" {
		local townships "portland"
	}
	if "`state'" == "Georgia" {
		local townships `" "atlanta" "savannah"  "'
	}
	if "`state'" == "Connecticut" {
		local townships `" "new haven" "bridgeport" "hartford" "waterbury"  "'
	}
	if "`state' "== "Alabama" {
		local townships "birmingham"
	}
	if "`state'" == "Tennessee" {
		local townships `" "memphis" "nashville"  "'
	}
	if "`state'" == "Virginia" {
		local townships `" "richmond" "norfolk" "'
	}
	if "`state'" == "Nebraska" {
		local townships "omaha" 
	}
	if "`state'" == "Texas" {
		local townships `" "san antonio" "dallas" "houston" "fort worth" "'
	}
	if "`state'" == "Utah" {
		local townships "salt lake city" 
	}
	if "`state'" == "Delaware" {
		local townships "wilmington" 
	}
	if "`state'" == "Iowa" {
		local townships "des moines"
	}
	if "`state'" == "Kansas" {
		local townships "kansas city"
	}
	if "`state'" == "Oklahoma" {
		local townships "oklahoma city" 
	}
	if "`state'" == "South Carolina" {
		local townships "charleston"
	}
	if "`state'" == "Florida" {
		local townships "jacksonville"
	}
	
	di "`state'"
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
	keep if event_state == "`state'"
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
	
	foreach town of local townships {
		drop if strpos(township2, "`town'") > 0
	}
	
	// make an id variable from 1 to n
	gen id1 = _n
	
	// Reformat for machine learning 
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
	
	save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_census_states\temp_census_`state'", replace
	
}
