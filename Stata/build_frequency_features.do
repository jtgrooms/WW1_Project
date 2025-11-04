
// Build more census features 

// find the frequency of every census county 
use "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_event_county.dta", clear
drop ark1910 
replace event_county = strlower(event_county)
sort event_county
by event_county: gen county1_count = _N
by event_county: gen dup = cond(_N == 1, 0, _n)
drop if dup > 1
drop dup 
rename event_county county1
drop if missing(county1)
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\county1_counts.dta", replace
export delimited "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\county1_counts.csv", replace
rename county1 county2
rename county1_count county2_count
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\county2_counts.dta", replace
export delimited "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\county2_counts.csv", replace

// find the frequency of every census township 
use "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_event_township.dta", clear
drop ark1910 
replace event_township = strlower(event_township)
sort event_township
by event_township: gen township1_count = _N
by event_township: gen dup = cond(_N == 1, 0, _n)
drop if dup > 1
drop dup 
rename event_township township1
drop if missing(township1)
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\township1_counts.dta", replace
export delimited "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\township1_counts.csv", replace
rename township1 township2
rename township1_count township2_count
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\township2_counts.dta", replace
export delimited "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\township2_counts.csv", replace

// find the requency of every census firstname
ssc install chimchar

use "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_pr_birth_year.dta", clear
keep if pr_birth_year>=1885 & pr_birth_year<=1900
drop pr_birth_year 
merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_is_male.dta", nogen keep(1 3)
keep if is_male==1
drop is_male 
*merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_event_state", nogen keep(1 3) 
*keep if event_state=="Oregon"
*drop event_state
// merge in given names 
merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_pr_name_gn.dta", nogen keep(1 3)
drop ark1910 
replace pr_name_gn = strlower(pr_name_gn)
split pr_name_gn, parse(" ")
do "V:\FHSS-JoePriceResearch\tools\server_stata_installs\chimchar_serverinstall.do"
chimchar pr_name_gn1, numremove
gen first1 = pr_name_gn1 
keep first1
sort first1
by first1: gen first1_count = _N
by first1: gen dup = cond(_N == 1, 0, _n)
drop if dup > 1
drop dup 
drop if missing(first1)
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\first1_counts.dta", replace
export delimited "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\first1_counts.csv", replace
rename first1 first2
rename first1_count first2_count
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\first2_counts.dta", replace
export delimited "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\first2_counts.csv", replace

// find the frequency of every census lastname 
use "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_pr_birth_year.dta", clear
keep if pr_birth_year>=1885 & pr_birth_year<=1900
drop pr_birth_year 
merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_is_male.dta", nogen keep(1 3)
keep if is_male==1
drop is_male 
*merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_event_state", nogen keep(1 3) 
*keep if event_state=="Oregon"
*drop event_state
// merge in given names 
merge 1:1 ark1910 using "V:\FHSS-JoePriceResearch\data\census_refined\fs\1910\ark1910_pr_name_surn.dta", nogen keep(1 3)
drop ark1910 
replace pr_name_surn = strlower(pr_name_surn)
gen rev = reverse(pr_name_surn)
split rev, parse(" ")
chimchar rev1, numremove
gen last1 = rev1 
replace last1 = reverse(last)
keep last1 
sort last1
by last1: gen last1_count = _N
by last1: gen dup = cond(_N == 1, 0, _n)
drop if dup > 1
drop dup 
drop if missing(last1)
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\last1_counts.dta", replace
export delimited "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\last1_counts.csv", replace
rename last1 last2 
rename last1_count last2_count
save "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\last2_counts.dta", replace
export delimited "V:\FHSS-JoePriceResearch\RA_work_folders\Jared_Grooms\German_Discrimination\data\last2_counts.csv", replace


