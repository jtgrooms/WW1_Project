
// make a local for all states 
local states `" "ALABAMA" "ARIZONA" "ARKANSAS" "CALIFORNIA" "COLORADO" "CONNECTICUT" "DELAWARE" "FLORIDA" "GEORGIA" "IDAHO" "ILLINOIS" "INDIANA" "IOWA" "KANSAS" "KENTUCKY" "LOUISIANA" "MAINE" "MARYLAND" "MASSACHUSETTS" "MICHIGAN" "MINNESOTA" "MISSISSIPPI" "MISSOURI" "MONTANA" "NEBRASKA" "NEVADA" "NEW HAMPSHIRE" "NEW JERSEY" "NEW MEXICO" "NEW YORK" "NORTH CAROLINA" "NORTH DAKOTA" "OHIO" "OKLAHOMA" "PENNSYLVANIA" "RHODE ISLAND" "SOUTH CAROLINA" "SOUTH DAKOTA" "TENNESSEE" "TEXAS" "UTAH" "VERMONT" "VIRGINIA" "WASHINGTON" "WEST VIRGINIA" "WISCONSIN" "WYOMING" "' 
di `states'
local states "OREGON"

// load in all the soldiers for each state and save seperately 
foreach state of local states {
	
	// set a local for the 100 biggest cities in the US in 1910
	if "`state'" == "OREGON" {
		local townships "portland"
	}
	if "`state'" == "NEW YORK" {
		local townships `" "new york" "buffalo" "rochester" "syracuse" "albany" "yonkers" "troy" "utica" "schenectady" "'
	}
	if "`state'" == "ILLINOIS" {
		local townships `" "chicago" "peoria" "east st. louis"  "'
	}
	if "`state'" == "PENNSYLVANIA" {
		local townships `" "philadelphia" "pittsburgh" "scranton" "reading" "youngstown" "wilkes-barre" "erie" "harrisburg" "johnstown" "'
	}
	if "`state'" == "MISSOURI" {
		local townships `" "st. louis" "kansas city" "st. joseph"  "'
	}
	if "`state'" == "MASSACHUSETTS" {
		local townships `" "boston" "worcester" "fall river" "lowell" "cambridge" "new bedford" "lynn" "springfield" "lawrence" "somerville" "holyoke" "brockton" "'
	}
	if "`state'" == "OHIO" {
		local townships `" "cleveland" "cincinnati" "columbus" "toledo" "dayton" "akron"  "'
	}
	if "`state'" == "MARYLAND" {
		local townships "baltimore"
	}
	if "`state'" == "MICHIGAN" {
		local townships `" "detroit" "grand rapids"  "'
	}
	if "`state'" == "CALIFORNIA" {
		local townships `" "san francisco" "los angeles" "oakland"  "'
	}
	if "`state'" == "WISCONSIN" {
		local townships "milwaukee"
	}
	if "`state'" == "NEW JERSEY" {
		local townships `" "newark" "jersey city" "paterson" "trenton" "camden" "elizabeth" "hoboken" "bayonne" "passaic" "'
	}
	if "`state'" == "LOUISIANA" {
		local townships "new orleans"
	}
	if "`state'" == "MINNESOTA" {
		local townships `" "minneapolis" "st. paul" "duluth"  "'
	}
	if "`state'" == "WASHINGTON" {
		local townships `" "seattle" "spokane" "tacoma""'
	}
	if "`state'" == "INDIANA" {
		local townships `" "indianapolis" "evansville" "fort wayne" "terre haute" "south bend" "'
	}
	if "`state'" == "RHODE ISLAND" {
		local townships "providence"
	}
	if "`state'" == "KENTUCKY" {
		local townships "louisville"
	}
	if "`state'" == "COLORADO" {
		local townships "denver"
	}
	if "`state'" == "MAINE" {
		local townships "portland"
	}
	if "`state'" == "GEORGIA" {
		local townships `" "atlanta" "savannah"  "'
	}
	if "`state'" == "CONNECTICUT" {
		local townships `" "new haven" "bridgeport" "hartford" "waterbury"  "'
	}
	if "`state' "== "ALABAMA" {
		local townships "birmingham"
	}
	if "`state'" == "TENNESSEE" {
		local townships `" "memphis" "nashville"  "'
	}
	if "`state'" == "VIRGINIA" {
		local townships `" "richmond" "norfolk" "'
	}
	if "`state'" == "NEBRASKA" {
		local townships "omaha" 
	}
	if "`state'" == "TEXAS" {
		local townships `" "san antonio" "dallas" "houston" "fort worth" "'
	}
	if "`state'" == "UTAH" {
		local townships "salt lake city" 
	}
	if "`state'" == "DELAWARE" {
		local townships "wilmington" 
	}
	if "`state'" == "IOWA" {
		local townships "des moines"
	}
	if "`state'" == "KANSAS" {
		local townships "kansas city"
	}
	if "`state'" == "OKLAHOMA" {
		local townships "oklahoma city" 
	}
	if "`state'" == "SOUTH CAROLINA" {
		local townships "charleston"
	}
	if "`state'" == "FLORIDA" {
		local townships "jacksonville"
	}
	*/
	
	// load in the data for that state 
	use "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\soldier_pids2.dta" if state == "`state'", clear
	count 
	// keep important variables 
	keep cause givenname name event_place deathdate pid match_score
	// generate a new variable called rev that reverses the order of the string name 
	gen rev = reverse(name)
	// split rev at every space into sepatate variables 
	split rev
	// generate a variable called surname that is rev1 reversed
	gen surname = reverse(rev1)
	// rename givenname 
	rename givenname firstname
	// reverse the string place 
	gen place = reverse(event_place)
	// split on a comma 
	split place, p(",")
	// make a variable called county that is place3 reversed 
	gen county = reverse(place3)
	// make a variable called township2 that is place4 reversed 
	gen township2 = reverse(place4)
	// lowercase string variables firstname surname county township2 
	for X in any firstname surname county township2: replace X = lower(X)
	// make an id variable 
	gen id1 = _n
	// keep important variables 
	keep firstname surname county township2 deathdate pid match_score cause event_place id1
	// now lets drop any observations that are in the 100 biggest US citites 
	foreach town of local townships {
		drop if strpos(township2, "`town'") > 0
	}
	
	// Reformat the data for machine learning 
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
	
	// save the raw state data excluding the biggest cities in 1910 
	save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\base_data\raw_scraped_states\temp_rll_`state'.dta", replace 
	
}


















