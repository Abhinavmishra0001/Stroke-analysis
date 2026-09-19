# How I explain this project in an interview

## 30-second version

I analysed a healthcare stroke dataset with about 5,000 patient records. First I cleaned missing BMI values and created charts to understand the data. Then I trained Logistic Regression and Random Forest models to predict stroke risk. Because stroke cases were rare, I used recall and PR-AUC instead of accuracy alone. Finally, I used K-Means clustering to explore groups of patients with similar health characteristics.

## Simple project flow

1. **Load the data** with Pandas.
2. **Clean it** by converting BMI to numbers and filling missing BMI values with the median.
3. **Explore it** using Matplotlib charts for stroke counts and age patterns.
4. **Split it** into training data (75%) and testing data (25%).
5. **Train two models** and compare their test-set results.
6. **Cluster patients** with K-Means, then use PCA to show those groups in a 2D chart.

## Questions I can answer

**What is supervised learning here?**  
The `stroke` column is the answer we already know. The models learn patterns from the other columns to predict that answer for a new record.

**What is unsupervised learning here?**  
K-Means does not use the `stroke` answer to form groups. It finds patients that look similar based on age, glucose, BMI, hypertension, and heart disease.

**Why did you use two models?**  
Logistic Regression is simple and easy to interpret. Random Forest can learn more complex relationships. Comparing them is more useful than assuming one model is best.

**Why is accuracy not enough?**  
Most people in the data did not have a stroke. A model could predict “no stroke” for everyone and still get high accuracy. Recall tells me how many actual stroke cases the model found, and PR-AUC is better suited to rare positive cases.

**What is data leakage?**  
It happens if the model accidentally sees information from the test set while training. I used a scikit-learn pipeline so the imputer, scaler, and encoder learn only from training data.

**What would you improve next?**  
I would try cross-validation, tune the prediction threshold based on the business goal, test for fairness across groups, and validate with new external data.

## Important honest note

This is a learning and portfolio project, not a medical tool. A clinical prediction model requires medical expertise, additional validation, safety checks, and governance.
