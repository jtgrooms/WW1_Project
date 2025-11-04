
// build world war 1 data for every state

// make a local for states 
local states `" "Alabama" "Arizona" "Arkansas" "California" "Colorado" "Connecticut" "Delaware" "Florida" "Georgia" "Idaho" "Illinois" "Indiana" "Iowa" "Kansas" "Kentucky" "Louisiana" "Maine" "Maryland" "Massachusetts" "Michigan" "Minnesota" "Mississippi" "Missouri" "Montana" "Nebraska" "Nevada" "New Hampshire" "New Jersey" "New Mexico" "New York" "North Carolina" "North Dakota" "Ohio" "Oklahoma" "Pennsylvania" "Rhode Island" "South Carolina" "South Dakota" "Tennessee" "Texas" "Utah" "Vermont" "Virginia" "Washington" "West Virginia" "Wisconsin" "Wyoming" "'
local states "Oregon"

foreach state of local states {
	
	// make a local for the 100 biggest cities in the us in 1910 
	if "`state'" == "Oregon" {
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
	
	use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\us_updated.dta", clear
	keep firstname surname name status_str residence state_name_spaces geocodehere_county state_name status geocodehere_lat geocodehere_lon id 
	rename id evan_id
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

	// Limit evan data to just oregon /////////////////////////////////////////////////////////////
	keep if state_name_spaces == "`state'"

	// There are many observations missing a first name in this data 
	// Lets write some code to fill in the first name filed with the first element of the surname field
	// I will split the surname and replace firstname with the first word of surname if there was more than one word in surname. I don't want to replace an empty firstname with a surname 
	split surname 
	cap replace firstname = surname2 if firstname == "" & surname2 != "" 
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
	
	// Reformat evan's data for the machine learning model ////////////////////////////////////////
	keep firstname surname county township state_name_spaces geocodehere_lat geocodehere_lon evan_id

	rename firstname first1
	rename surname last1
	rename county county1
	rename township township1
	rename state_name_spaces state1
	rename geocodehere_lat lat1
	rename geocodehere_lon lon1

	foreach town of local townships{
		drop if strpos(township1, "`town'") > 0
	}

	// save this data for now 
	save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_ww1_states\temp_evan_`state'", replace
	
}

