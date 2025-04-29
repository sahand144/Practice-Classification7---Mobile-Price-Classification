path = r"...path...\Mobile Price Classification.zip"

import zipfile as zip
import pandas as pd
import matplotlib.pyplot as plt

# lets extract the files
with zip.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall()
    test_filename = zip_ref.namelist()[0]
    train_filename = zip_ref.namelist()[1]
    with zip_ref.open(test_filename) as file:
        test_df = pd.read_csv(file)
    with zip_ref.open(train_filename) as file:
        train_df = pd.read_csv(file)

#Inspect the data
print(train_df.head())
print(test_df.head())
#Drop the useless column id
test_df.drop(columns=['id'], inplace=True)

print(train_df.shape)
print(test_df.shape)

print(train_df.dtypes)
print(test_df.dtypes)

#look at the missing values
print(train_df.isnull().sum())#no missing values
print(test_df.isnull().sum())#no missing values

#look at the unique values
print(train_df.nunique())
print(test_df.nunique())
print(train_df['price_range'].unique())

#look at the descriptive statistics with help of describe() method and plot the histogram of the data
print(train_df.describe())
print(test_df.describe())


#look at the distribution of the data
print(train_df.hist(figsize=(10,10)))
plt.show()
print(test_df.hist(figsize=(10,10)))
plt.show()


#look at the correlation matrix with help of seaborn heatmap plot
import seaborn as sns
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm')
plt.show()
sns.heatmap(test_df.corr(), annot=True, cmap='coolwarm')
plt.show()

#Define X,y
X = train_df.drop('price_range', axis=1)
y = train_df['price_range']

#define a function to extract any two features which have high correlation with each other and return the list of features
def get_highly_correlated_features(df, threshold=0.6):
    # Compute correlation matrix
    corr_matrix = df.corr()    
    # Extract upper triangle of correlation matrix to avoid duplicates
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr_pairs.append((col1, col2))
    return high_corr_pairs

print(get_highly_correlated_features(X))
#we can see that the features which have high correlation with each other are: 'fc' and 'pc'
#so we need to drop one of them
X.drop(columns=['pc'], inplace=True)

#now we need to apply standardization to the data but i prefer to apply a pipeline to the data for better performance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#now we need to split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#now we need to apply the pipeline to the data
pipeline_lr = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
pipeline_lr.fit(X_train, y_train)  

#now we need to predict the testing set
y_pred_lr = pipeline_lr.predict(X_test)

#now we need to evaluate the model
print(accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))

#now we need to visualize the result with help of confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred_lr ), annot=True, cmap='coolwarm')
plt.show()

#now lets try to improve the model with help of grid search cv
from sklearn.model_selection import GridSearchCV
parameters = {'model__C': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(pipeline_lr, parameters, cv=5)
grid_search.fit(X_train, y_train)

#now we need to predict the testing set
y_pred_lr_gs = grid_search.predict(X_test)

#now we need to evaluate the model
print(accuracy_score(y_test, y_pred_lr_gs))
print(classification_report(y_test, y_pred_lr_gs))
print(confusion_matrix(y_test, y_pred_lr_gs))

#now we need to visualize the result with help of confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred_lr_gs), annot=True, cmap='coolwarm')
plt.show()


#now lets try to improve the model with help of random forest classifier without grid search cv
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

#now we need to predict the testing set
y_pred_rf = model_rf.predict(X_test)

#now we need to evaluate the model
print(accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))


#now lets try to improve the model with help of random forest classifier with grid search cv
parameters = {'n_estimators': [10, 50, 100, 200], 'max_depth': [10, 20, 30, 40, 50]}
grid_search = GridSearchCV(model_rf, parameters, cv=5)
grid_search.fit(X_train, y_train)

#now we need to predict the testing set
y_pred_rf_gs = grid_search.predict(X_test)

#now we need to evaluate the model
print(accuracy_score(y_test, y_pred_rf_gs))
print(classification_report(y_test, y_pred_rf_gs))
print(confusion_matrix(y_test, y_pred_rf_gs)) 

#now we need to visualize the result with help of confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf_gs), annot=True, cmap='coolwarm')
plt.show()

#now we need to apply the pipeline to the data
pipeline_rf = Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier())])
pipeline_rf.fit(X_train, y_train)

#now we need to predict the testing set
y_pred_rf_pipeline = pipeline_rf.predict(X_test)

#now we need to evaluate the model
print(accuracy_score(y_test, y_pred_rf_pipeline))
print(classification_report(y_test, y_pred_rf_pipeline))
print(confusion_matrix(y_test, y_pred_rf_pipeline))

