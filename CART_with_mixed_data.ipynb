import numpy as np
import pandas as pd

churn = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
churn["TotalCharges"] = pd.to_numeric(churn.TotalCharges, errors = "coerce")
churn.dropna(inplace = True)

churn_data = churn.copy()
label = churn_data["Churn"].map({"Yes" : 1, "No" : 0})
churn_data.drop(["customerID", "Churn"], axis = 1, inplace = True)

churn_data["MultipleLines"].replace({"No phone service" : "No"}, inplace = True)
for column in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
    churn_data[column].replace({"No internet service": "No"}, inplace = True)
for column in ["PaperlessBilling", "PaymentMethod"]:
    churn_data.drop(column , axis =1, inplace = True)
churn_data.describe(include = "all")

churn_data["OnlineSecurity" + "_cat"] = churn_data["OnlineSecurity"].map({"Yes" : 1, "No": 0})
churn_data.drop("OnlineSecurity", axis = 1, inplace = True)
churn_data

for column in churn_data.iloc[:, [2,3,5,6,8,9,10,11,12]]:
    churn_data[column + "_cat"] = churn_data[column].map({"Yes" : 1, "No": 0})
    churn_data.drop(column, axis = 1, inplace = True)

churn_data["gender_cat"] = churn_data["gender"].map({"Male" : 1, "Female" : 0})
churn_data.drop("gender", axis = 1, inplace = True)

churn_data["Contract_cat"] = churn_data["Contract"].map({"Month-to-month" : 1, "One year" : 2, "Two year": 3})
churn_data.drop("Contract", axis = 1, inplace = True)
churn_with_dummies = pd.get_dummies(data = churn_data, columns = ["InternetService"])
churn_with_dummies.head()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydotplus
X_train, X_test, y_train, y_test = train_test_split(churn_with_dummies, label, test_size = 0.3, random_state = 42)
clf = DecisionTreeClassifier(max_depth = 8, min_samples_leaf = 40)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Độ chính xác là: {} %".format(accuracy_score(y_pred, y_test)*100))
