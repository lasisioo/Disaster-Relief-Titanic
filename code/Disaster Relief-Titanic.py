## Disaster Relief - Titanic


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import feature_selection
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


#use "conda install psycopg2" on terminal
from sqlalchemy import create_engine
import pandas as pd
connect_param = 'postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com:5432/titanic'
engine = create_engine(connect_param)


df = pd.read_sql("train", engine)
df.head()
del df["index"]
df.head()
n[4]:


df.shape


# ## Exploratory Data Analysis

df.describe()

# The data frame consists of 891 entries and 12 columns.


df.shape


sns.heatmap(df.corr())


# The heat map provides relationships between variables in the dataframe. 
# From the map, we that class is negatively correlated to survival while Fare is postively correlated. 
# This is makes sense intuitively because poeple from Classes 2 and 3 payed less fares and were less likely to survive 
# the rich people first!


sns.violinplot(x="Pclass", y="Age", hue="Survived", data=df, palette="muted", figsize=(18, 6), split=True)
# The next step in the process is exploring the data for better understanding. 
# From the first violinplot, we can see the distribution of age across classes for both those that survived and those that didn't. 
#In class 1, people that survived were, on average, younger than people that didn't. 
#In classes 2 and 3, we see that the average of those that survived was similar to that of those that didn't. Also, there more children in classes 2 and 3, and most of them survived.


sns.violinplot(x="Pclass", y="Survived", hue="Sex", data=df, split=True)
# The plot above shows a distribution of males and females that survived across the three classes. 
#There are several things to observe from this plot. First, the survival rate is lower in classes 2 and 3. 
#Women in all classes survived more than men. More women survived than didn't in classes 1 and 2. Very few women in classes 1 and 2 died. It appears to be the case that the proportion of women that survived in class 3 is similar to that of women that didn't survive in class 3. Finally, the lower the class the more men that died.


sns.violinplot(x="Embarked", y="Survived", hue="Sex", data=df, split=True)
# The last plot shows a distribution of males and females that survival based on their port of embarkment. 
#Fewer men from Queenstown survived and fewer women from Queenstown died. 
#In general, more women survived than men, and more men died than women.


## Part 3: Data Wrangling


df.isnull().any()
df.count()

#How many missing Age values are there?
print "Number of missing Age values:", sum(df["Age"].isnull().values.ravel())


#How is it distributed?
df.Age.plot(kind = 'hist', bins=15)
plt.title("Distribution of Age", size = 14)
plt.xlabel("Age")
plt.ylabel("Frequency")


#We see that the distribution is mainly centered around the mean(and there are few outliers). 
#For this reason, I will replace the empty fields with the overall mean age.
age_pipe = make_pipeline(Imputer(strategy="mean"))
df["Age"] = pd.DataFrame(age_pipe.fit_transform(df[["Age"]]))


#How many missing fields are in the "Embarked" column?
sum(df["Embarked"].isnull().values.ravel())


df.Embarked.value_counts()


#A large proportion of the passengers embarked from Southampton. 
#Since we only have only two empty fields in "Embarked" column, I will replace them with the majority i.e. Southampton.
df.Embarked = df.Embarked.fillna('S')


df.isnull().any()
#I will not be using the "Cabin" column so I would worry about it's missing values.


#I want to create dummy variables for Sex, Embarked, and Pclass.
#Since I won't be needing it anymore, I dropped the "Sex" column in df2.
dummydf = pd.get_dummies(df["Sex"])
dummydf2 = pd.get_dummies(df["Embarked"])
dummydf3 = pd.get_dummies(df["Pclass"], prefix ="Class")
df2 = df[["Survived", "Pclass", "Age", "SibSp", "Parch", 
          "Fare", "Embarked"]].join(dummydf)
df3 = df2[df2.columns].join(dummydf2)
df4 = df3[df3.columns].join(dummydf3)
df4.head()


#To decide which columns to delete, I want to see which values are most frequent.
#I will be using the original dataframe (df) for this.
print df.Sex.value_counts()
print df.Embarked.value_counts()
print df.Pclass.value_counts()


#I also want to drop the "Embarked" and "Pclass" columns.
df4.drop(df4[["Pclass", "Embarked", "male", "S", "Class_3"]], axis=1, inplace=True)


df4.head()


df4.dtypes


#I want to make "Age" values integers instead of floats (It makes more sense that way).
df4[["Age"]] = df4[["Age"]].astype(int)
df4.dtypes


sns.heatmap(df4.corr())


df4.shape


# The heatmap show correlations between the variables we will be using in the regression. 
# From the graph, we see that being female is the most positively related to survival.

## Part 4: Logistic Regression and Model Validation


#I want to define the variables for my regression analysis.
#My dependent variable (y) is the "Survived" column.
#My independent variables (x) are "Age", "Parch", "SibSp", "Fare", "Female", "C", "Q", "Class_1", and "Class_2".
X = df4[df4.columns[1:]]
y = df4[df4.columns[0]]


#The regression.
lm = LogisticRegression()

result = lm.fit(X,y)
predictions = lm.predict(X)
print "Score:",result.score(X,y)


#To determine the coefficients for the correlations and the intercept:
print result.coef_
print result.intercept_


# To determine the p-value for each coefficient:
from sklearn.feature_selection import chi2
scores, pvalues = chi2(X, y)
pvalues


# To determine the odds for each coefficeient, I have to take the exponent of the coefficient.
# The same goes for the intercept.
print np.exp(result.coef_)
print np.exp(result.intercept_)


# From our results, we see that all coefficients but SibSp and Q are statistically signifant at a 5% significance level. 
# Age, SibSp and Parch generally decrease the odds of survival while Fare, Female, C, Q, Class_1, 
# and Class_2 generally increase the odds of survival. For this model, our baseline is a male from Southhampton in the 3rd class. 
# This person has a 0.34 to 1 odds of survival. 
# If this person happened to be female, their odds of survival increases to 4.46 to 1 (13.261 x 0.336). 
# If this person were to be female and in the 1st class (keeping other factors the same), their odds increases to 30.59 to 1 (13.261 x 0.336 x 6.859).  


# I want group the age into bins to see which age group impacted survival the most. 
def binAge(age): 
    if age > 60:
        return "61_and_above"
    elif age >= 46:
        return "46-60"
    elif age >= 31:
        return "31-45"
    elif age >= 16:
        return "16-30"
    
    return "16_and_under"
df4["Age"] = df4.Age.map(lambda age: binAge(age) )
df5 = df4
dummies5 = pd.get_dummies( df5["Age"], prefix = "Age" )
newData = df5.join(dummies5)
newData.head()


# Which age group is the most frequent?
print newData.Age.value_counts()


# I want to drop the "Age" and "16-30" columns.
newData.drop(newData[["Age", "Age_16-30"]], axis=1, inplace=True)


newData.shape


sns.heatmap(newData.corr())


from patsy import dmatrices


newData.dtypes


# Defining veriables for my new regession:
x = newData[newData.columns[1:]]
Y = newData[newData.columns[0]]

lm2 = LogisticRegression()

result2 = lm2.fit(x,Y)
predictions2 = lm2.predict(x)
print "Score:",result2.score(x,Y)


# To determine the coefficients for the correlations and the intercept:
print result2.coef_
print result2.intercept_


# To determine the p-value for each coefficient:
from sklearn.feature_selection import chi2
scores, pvalues = chi2(x, Y)
pvalues


pvalues = pvalues.tolist()
pvalues.append("N/A")


# To determine the odds for each coefficeient, I have to take the exponent of the coefficient.
# The same goes for the intercept.
print np.exp(result2.coef_)
print np.exp(result2.intercept_)


a = np.exp(result2.coef_[0]).tolist()
b = np.exp(result2.intercept_).tolist()
a.append(b[0])


c = x.columns.tolist()
c.append("Intercept")


d = pd.DataFrame(zip(c, a, pvalues), columns=["CoeffName", "Coeff", "pvalues"])
d


# From our results, we see SibSp, Q and all age groups except "Age_16 and under" statistically insignificant at a 5% significance level. 
# For this model, our baseline is a male from Southhampton in the 3rd class between the ages of 16-30. 
# Similar to the first model, SibSp and Parch generally decrease the odds of survival while Fare, Female, C, Q, Class_1, Class_2, 
# and Age_16 and under generally increase the odds of survival. 
# From this model, we find that only children 16 and under in the group had a significantly higher chance of survival.


x_train, x_test, Y_train, Y_test = train_test_split(x, Y, train_size=0.70, random_state=15)

lm3 = LogisticRegression()

result3 = lm3.fit(x_train, Y_train)
predictions3 = lm3.predict(x_test)
print "Score:",result3.score(x_test,Y_test)


pb = result3.predict_proba(x_test)
x_test["ProbabilityOfZero"], x_test["ProbabilityOfOne"] = zip(*pb)

x_test["actualSurvived"] = Y_test
x_test['predictedSurvived'] = result3.predict( x_test[ x_test.columns[0:12] ] )
dFrame = x_test
#dFrame['predictedSurvived'] = result3.predict( dFrame[ dFrame.columns[0:12] ] )
dFrame.head()


from sklearn import cross_validation


x_train, x_test, Y_train, Y_test = cross_validation.train_test_split(x, Y, train_size=0.70, random_state=15)
lm4 = LogisticRegression()

result4 = lm4.fit(x_train,Y_train)
predictions = lm4.predict(x_test)
print "Score:", result4.score(x_test, Y_test)


print pd.crosstab(
                    dFrame["actualSurvived"],
                    dFrame["predictedSurvived"], 
                    rownames=["actual"]
                 )


from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc


# Compute ROC curve and ROC area for Adaboost
logfpr = dict()
logtpr = dict()
logroc_auc = dict()
logfpr, logtpr, _ = roc_curve(Y_test, x_test["ProbabilityOfOne"])
logroc_auc = auc(logfpr, logtpr)

# Plot of a ROC curve 
plt.figure(figsize=(15,15))
plt.plot(logfpr,logtpr,label='AUC = %0.2f' % logroc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('Receiver Operating Characteristic\n', fontsize=30)
plt.legend(loc="lower right", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

plt.show()


from sklearn.metrics import classification_report


target    = dFrame["actualSurvived"].tolist()
predicted = dFrame["predictedSurvived"].tolist()
target_names = ["Not Survived", "Survived"]

print(classification_report(target, predicted, target_names=target_names))


# The precision (TP/TP+FP) shows retrieved instances that are relevant 
# i.e. the proportion of people that actually survived from the total number people that our model predicted survived. 
# Recall (TP/TP+FN) indicates the proportion of people that our model predicted survived from the total number of people that actually survived. 
# The F1-score indicates how well it can predict a passenger surviving relative to predicting a passenger not surviving. 
# For this model, the precision, recall and F1-score are all 0.79.


print metrics.accuracy_score(Y_test, predictions3)
print metrics.roc_auc_score(Y_test, pb[:, 1])


# From the analysis above, I can imply that people on the titanic were more likely to survive if they were female, 
# aged 16 or below, and in the 1st class. 
# These results were derived from certain assumptions what I will need to review for future work. 
# For instance, I assumed the age of 177 passengers (20% of the dataset) to be the average age of the remaining 714 
# this is a huge percentage relative to the dataset. 
# Another problem is that I assumed everyone was in their assigned class while the boat was sinking. 
# It may have been the case that some people from the 1st class cabin (whose cabins were vertically further away from the ocean) were actually at the lower levels of the boat. I really don't see the 1st class passengers chilling at the lower level with the "commoners," but who's to say? Still, from the data we collect, I was able to develop so insights about the chances of survival. With a pretty decent precision and recall rate, I can apply this model to help manage situations in the event of a plane crash. Imagine a scenario when a plane crash occurred, and were are unable to find all the passengers and crew members. We can collect information from each passenger's boarding pass and driver's license/passport, we can use this model to predict (to a good extent) the chance of survival. Obviously, certain other factors will have to be considered given that the disaster in a plane crash and not a ship sinking. 
