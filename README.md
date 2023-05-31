# Heart Failure
This project titled Heart Failure Takes into consideration a number of features by which it can be predicted whether a person will suffer heart failure (leading to death or not). Heart failure is of course not desirable in our world and should be prevented.

The model may help prevent death by heart failure by examining whether a patients characteristics (predictors) may lead to death. If the current one points to death, then adjusting the patients habits like smoking and others may lead to a longer life.


### Data Source:
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)

### Data Features:
1. age: age of the patient (years)
2. anaemia: decrease of red blood cells or hemoglobin (boolean)
3. high blood pressure: if the patient has hypertension (boolean)
4. creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
5. diabetes: if the patient has diabetes (boolean)
6. ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
7. platelets: platelets in the blood (kiloplatelets/mL)
8. sex: woman or man (binary)
9. serum creatinine: level of serum creatinine in the blood (mg/dL)
10. serum sodium: level of serum sodium in the blood (mEq/L)
11. smoking: if the patient smokes or not (boolean)
12. time: follow-up period (days)

### Data Target:
 death event: if the patient deceased during the follow-up period (boolean)

### Regression Techniques used:
1. [Linear Regression](https://www.oxfordreference.com/display/10.1093/oi/authority.20110803100107226;jsessionid=BAD370C49344F63EAF545090E2E032DE)
2. [K-Nearest Neighbor (KNN)](https://online.stat.psu.edu/stat508/lesson/k)
3. [Support Vector Machine (SVM)](https://www.researchgate.net/publication/221621494_Support_Vector_Machines_Theory_and_Applications/link/0912f50fd2564392c6000000/download)
4. [Decision Tree (DT)](https://online.stat.psu.edu/stat857/node/236/)

### Evaluation Metric:
[Accuracy score](https://developers.google.com/machine-learning/crash-course/classification/accuracy)
was used to evaluate the models.


### The best Model:
When the using the metric Mean Squared Error, the model with the lowest mean squared model is the best among all models
under consideration. However, while using the Coefficient of Determination, the model with the highest Coefficient of
Variation is preferable. Using both metrics, the Decision Tree came up as the best model

[View Code on Kaggle](https://www.kaggle.com/oluade111/heart-failure)
