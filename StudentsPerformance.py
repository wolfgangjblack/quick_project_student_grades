import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp,ttest_ind, f_oneway



df = pd.read_csv('StudentsPerformance.csv')
##Columns
print(df.columns)
#gender - Misnomer, biological sex 
#race/ethnicity - Unnamed Groups A,B,C,D,E
#parental level of education - some highschool to masters degree. 
#       I wonder what the difference is between 'some college' and associates
#       I'll assume some college is < associates
#lunch - standard or free/reduced (indicates poverty levels free/reduced set them at the poverty line) 
#test preperation course - compelted or nah
#math score
#reading score
#writing score

##--------------------------------------Understanding our data
##                             Basic scores
fig = plt.figure(0)
sns.histplot(data = df, x = 'math score', hue = 'gender', multiple = 'stack')
title = './Math score dist Gender'
fig.savefig(fname = title,format = 'png')

fig1 = plt.figure(1)
sns.histplot(data = df, x = 'reading score', hue = 'gender', multiple = 'stack')
plt.savefig('Reading score dist Gender')
fig2 = plt.figure(2)
sns.histplot(data = df, x = 'writing score', hue = 'gender', multiple = 'stack')
plt.savefig('writing score dist Gender')
## How about scores via poverty
fig3 = plt.figure(3)
sns.histplot(data = df, x = 'math score', hue = 'lunch', multiple = 'stack')
plt.savefig('Math score dist Lunch Status')
fig4 = plt.figure(4)
sns.histplot(data = df, x = 'reading score', hue = 'lunch', multiple = 'stack')
plt.savefig('reading score dist Lunch Status')
fig5 = plt.figure(5)
sns.histplot(data = df, x = 'writing score', hue = 'lunch', multiple = 'stack')
plt.savefig('writing score dist Lunch Status')

print(df['math score'].mean(), df.groupby('test preparation course')['math score'].mean())
print(df['reading score'].mean(), df.groupby('test preparation course')['reading score'].mean())
print(df['writing score'].mean(), df.groupby('test preparation course')['writing score'].mean())


##Finally the everloved violin plot- this will be to show the distributions of 
# #scores vs race/ethnicity

fig6 = plt.figure(6)
sns.violinplot(data = df, x='race/ethnicity', y = 'math score')
plt.savefig('Violin plot math score per race')


fig7 = plt.figure(7)
sns.violinplot(data = df, x='race/ethnicity', y = 'reading score')
plt.savefig('Violin plot reading score per race')
fig8 = plt.figure(8)
sns.violinplot(data = df, x='race/ethnicity', y = 'writing score')
plt.savefig('Violin plot writing score per race')
print(df['math score'].mean(), df.groupby('race/ethnicity')['math score'].mean())
print(df['reading score'].mean(), df.groupby('race/ethnicity')['reading score'].mean())
print(df['writing score'].mean(), df.groupby('race/ethnicity')['writing score'].mean())

#Moar violin plots to show parental education vs scores
fig9 = plt.figure(9)
sns.violinplot(data = df, x='parental level of education', y = 'math score')
plt.xticks(rotation=45)
plt.savefig('Violin plot math score per parental education')
fig10 = plt.figure(10)
sns.violinplot(data = df, x='parental level of education', y = 'reading score')
plt.xticks(rotation=45)
plt.savefig('Violin plot reading score per parental education')

fig11 = plt.figure(11)
sns.violinplot(data = df, x= 'parental level of education', y = 'writing score')
plt.xticks(rotation=45)
plt.savefig('Violin plot writing score per parental education')

print(df['math score'].mean(), df.groupby('parental level of education')['math score'].mean())
print(df['reading score'].mean(), df.groupby('parental level of education')['reading score'].mean())
print(df['writing score'].mean(), df.groupby('parental level of education')['writing score'].mean())


# count plots, I want to see the break down of Students and their financials

fig12 = plt.figure(12)
sns.countplot(data = df, y = 'race/ethnicity', hue = 'lunch')
plt.savefig('Count plot race vs lunch ')

fig13 = plt.figure(13)
sns.countplot(data = df, y='parental level of education', hue = 'lunch')
plt.savefig('Count plot parental education vs lunch ')
#Interesting to see how the parental edcuation level affects the childs overall
#poverty level. However, need to be careful because we don't see easy relatable data here
#For instance, it appears that at a glance some college has more free/reduced lunches
#thansome highschool by ~20 -but we also have at least 40 more samples...

#T-Test
#Lets see if the test prep kids mean is different from the population mean. 

#I'm going to assume a larger population mean of 75 on all exams

test_prep = df.loc[df['test preparation course'] == 'completed']
t,pval_testprep_math = ttest_1samp(test_prep['math score'],75)
t,pval_testprep_read = ttest_1samp(test_prep['reading score'],75)
t,pval_testprep_write = ttest_1samp(test_prep['writing score'],75)

print('The pvalues for the test prep group are for math, reading, and writing, respectively :'+\
      str(pval_testprep_math)+' , '+str(pval_testprep_read)+' , '+str(pval_testprep_write))
    
#we can reject null hypothesis for the math score, and say these exams do not follow the math population well, however, 
# we can not say that for reading and writing

#2-sample T-test: Lets see if gender plays a role! the null hyp here is that 
# seperating the genders reveals they're actually from the same populaton and therefor gender doesn't have a role to play here

t2,pval_sex_math = ttest_ind(df.loc[df['gender']== 'male']['math score'], df.loc[df['gender']=='female']['math score'])
t2,pval_sex_read = ttest_ind(df.loc[df['gender']== 'male']['reading score'], df.loc[df['gender']=='female']['reading score'])
t2,pval_sex_write = ttest_ind(df.loc[df['gender']== 'male']['writing score'], df.loc[df['gender']=='female']['writing score'])

print(' ')
print('For gender, we\'re interested in seeing if gender impacts scores. If the pval per test is lower than our threshold of 0.05, we can say the genders perform differently.')
print(pval_sex_math,pval_sex_read,pval_sex_write)
print('These are all different, hence sex seems to influence the scores')
print(' ')
t2,pval_lunch_math = ttest_ind(df.loc[df['lunch']== 'standard']['math score'], df.loc[df['lunch'] != 'standard']['math score'])
t2,pval_lunch_read = ttest_ind(df.loc[df['lunch']== 'standard']['reading score'], df.loc[df['lunch'] != 'standard']['reading score'])
t2,pval_lunch_write = ttest_ind(df.loc[df['lunch']== 'standard']['writing score'], df.loc[df['lunch'] != 'standard']['writing score'])

print('We\'re also interested in seeing if lunch status impacts scores. If the pval per test is lower than our threshold of 0.05, we can say that students who can pay for standard lunch perform differently than students who can not.')
print(pval_lunch_math,pval_lunch_read,pval_lunch_write)
print('These are all different, hence lunch status seems to influence the scores')
print(' ')

##Machine Learning to stretch those muscles 
#Lets make 2 models for Multilinear Regression
#First model will have all features, 2nd model will scale down and try to get similar answer as before

#Do some data wrangling - change parental level education to a spectrum
#Drop races, we COULD put them on a spectrum based on what races performed
# better on some average of the exams.. but it feels bad man
#

#Multilinear regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

    
def education_to_number(string):
    if string == "bachelor's degree":
        var = 4
    elif string == 'some college':
        var = 2
    elif string == "master's degree":
        var = 5
    elif string == "associate's degree":
        var = 3
    elif string == 'high school':
        var = 1
    elif string =='some high school':
        var = 0
    return var

df['level_number']=df['parental level of education'].apply(lambda level: education_to_number(level))
df['gender_num'] = df.gender.apply(lambda gender: 0 if gender == 'female' else 1)
df['lunch_num'] = df['lunch'].apply(lambda lunch: 1 if lunch == 'standard' else 0)
df['preparation_num'] = df['test preparation course'].apply(lambda val: 1 if val == 'completed' else 0)

x = df[['gender_num', 'level_number', 'lunch_num','preparation_num']]
y = df['writing score']
# # Did math and reading, but the scores were lower. 
x_train, x_test, y_train,y_test = train_test_split(x,y,train_size = 0.8, test_size = 0.2, random_state = 100)

mlr = LinearRegression()
mlr_model = mlr.fit(x_train,y_train)
y_predict = mlr.predict(x_test)

print("Train score:")
print(mlr.score(x_train, y_train))

print("Test score:")
print(mlr.score(x_test, y_test))
plt.figure(20)
plt.scatter(y_test, y_predict,alpha = 0.2)
plt.savefig('Scatter plot of predicted scores vs actual scores - writing (Full features)')
print(mlr_model.coef_)
#So this modle isn't very strong. Probably this data isn't enough to really tell us if they'll do good on individual exams
# However, lets say we're happy with a mean squared error of .38 (explained ~38% of scores). 
# Seeing the coefficents, we can say that we gender, lunch, and prep explain most of the score -lets 
# see what happens when we eliminate parental education level.  

x = df[['gender_num', 'lunch_num','preparation_num']]
y = df['writing score']
# # Did math and reading, but the scores were lower. 
x_train, x_test, y_train,y_test = train_test_split(x,y,train_size = 0.8, test_size = 0.2, random_state = 100)

mlr = LinearRegression()
mlr_model = mlr.fit(x_train,y_train)
y_predict = mlr.predict(x_test)

print("Train score:")
print(mlr.score(x_train, y_train))

print("Test score:")
print(mlr.score(x_test, y_test))
plt.figure(21)
plt.scatter(y_test, y_predict, alpha = 0.2)
plt.savefig('Scatter plot of predicted scores vs actual scores - writing (reduced Features)')
#This score is .321, so we didn't lose a ton here dropping parental level. If our data was MASSIVE
# we could save time/resources skipping this model. 

















