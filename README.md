# quick_project_student_grades

Task Details
Predicting student performance with the demographic and socioeconomic information.

Expected Submission
The solution should contain Notebook with prediction accuracy.
An informative solution should also include statistical testing and information about what contributes the prediction.
Of course the script should be easy to read and understand.

Okay, gender vs scores is easy - but lets try to understand racial groups vs scores
Also parental education vs scores - violin
Also lunch vs scores - done

Find the total pop mean, find the mean per gender, education group...

Then whichever one is more interesting, lets try to see if we can use that and scores to machine learn some information - lets say try to predict a new students scores via their info. 

Seperate the groups into groups of ethnicity and see if there is a difference between them (ANOVA)

Seperate the groups into groups of completed prep and see if they're different (2 sample T-squared)

Predict if a round of new students will pass the test: (machine learn some shit - multilinear and knearest to avoid doing regularization)
Probably seperate the ethnicity groups out into different samples, but numberize their parents education and test prep. 

Turn parents into some college and above vs no college
Turn race into best scoring race vs other
And test prep course vs none

###############---------------------Learnings
Bar plots-
	Scores are largely single mode, right skewed. 

	Sex vs scores:
	math scores : mean = 66.1, with a male score 68.72 and female 63.633
	reading scores : mean = 69.2, with a male score 65.5 and female 72.6
	writing scores : mean = 68.1, with a male score 63.3 and female 75.5

	Lunch vs scores:
	math scores : mean = 66.1, with a lunch score 70.03 and free 58.92
	reading scores : mean = 69.2, with a lunch score 71.65 and free 64.65
	writing scores : mean = 68.1, with a lunch score 70.8 and free 63

	Prep Course vs scores:
	math scores : mean = 66.1, with a test prep course score 69.7 and no prep course 64.1
	reading scores : mean = 69.2, with a test prep course score 73.9 and no prep course 66.5
	writing scores : mean = 68.1, with a test prep course score 74.4 and no prep course 64.5

##### Lets do some T-tests to see if lunch or gender are different from the population, set threshold to 0.05. 

Violin plots - 
	Race/Ethnicity vs Scores
	Math: greater sample mean 66.1, groupA: 61.6,groupB: 63.5,groupC: 64.5, groupD: 67.4,groupE: 73.8
	Group A's interquartile is lower than the other groups, with a lower mean, group E seems to perform the best over all in maths. GroupC has the largest right skew (giving seemingly the lowest scores)

	Reading: greater sample mean (69.2), groupA: 64.7,groupB: 67.4, groupC: 69.1,groupD: 70.0,groupE: 73.0
	GroupA against has a lower interquartile range, though B is close. E performance the best again, with the highest mean. B appears to bne bimodal with peaks around 65 and a smaller peak around 80. 

	Writing: greater sample mean (68.1), groupA: 62.7, groupB: 65.6, groupC: 67.8, groupD: 70.1, groupE: 71.4
	While A is the lowest, and B is a close second, D and E are very comparable here. 


	Parental Education vs Scores
	Math: greater mean (66.1), associate's: 67.9, bachelor's: 69.4, highschool: 62.1, master's: 69.8, some college: 67.1,some high school: 63.5
	Reading: greater mean (69.2), associate's: 70.9, bachelor's: 73.0, highschool: 64.7, master's: 75.4, some college: 69.5, some high school: 67.0
	Writing: greater mean (68.1), associate's: 69.9, bachelor's: 73.4, high school: 62.4, master's: 75.7,some college: 68.8, some high school: 64.9


###############---------------------T-test
We want to see if the scores from this sample population match the expected mean (lets say 75) for those who took the prep course. This group was seperated and their sample scores for all tests were compared to the expected means. We find that math appears to different - though we're not sure if that means this group did better or worse (have to do a right left tail Test for that). The other two tests have pvalues > 0.05 so we're not about to reject the null

We also did two 2 sample T tests. One for gender, one for lunch status

gender:

Lunch status: 

lets do some machine learning!
