# %% [markdown]
# # Solution for Portuguese banking institution by 2022/P/1123

# %% [markdown]
# ### Install required packages

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix



# %% [markdown]
# #### Load data

# %%

df=pd.read_csv('banking.csv')

df.shape

# %% [markdown]
# #### Check the types of the data set

# %%
df.info()

# %% [markdown]
# #### Describing the data set
# 

# %%
df.describe()


# %% [markdown]
# ### check  for missing values

# %%

df.isnull().sum()

# %%
df.head(10)

# %% [markdown]
# ### Handling unkown values
# 

# %%
unknown_counts = df.apply(lambda x: x[x == 'unknown'].count())
unknown_counts

# %% [markdown]
# As we see here there are some unkown values in columns job,marital,eduction,default,housing and loan .As the next step we gonna replace the unkown values with mode value of the each column

# %%
for column in ['job','marital','education','default','housing','loan']:
    mode_val = df[column].mode()[0]
    df[column] = df[column].replace('unknown',mode_val)
unknown_counts = df.apply(lambda x: x[x == 'unknown'].count())
unknown_counts
df.shape

# %%
df.info()

# %% [markdown]
# we drop duration to achive realistic predictive model . Including it would artificially inflate the model’s performance, making the evaluation results unreliable for real-world scenarios.

# %%
df=df.drop(columns=['duration'])

# %% [markdown]
# ### Removing Outliers

# %%

int_float_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(['y'])
for column in int_float_cols:
 
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

  
    IQR = Q3 - Q1

 
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR


    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df.shape
  


# %% [markdown]
# ## Q Q Plot and Histogram

# %%
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

for column in numerical_features:
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    stats.probplot(df[column], dist="norm", plot=plt,)
    plt.title(f'Q-Q Plot of {column}')

    plt.subplot(1, 2, 2)
    sns.histplot(df[column], bins=30, color='green', kde=True)
    plt.title(f'Histogram of {column}')
    


plt.tight_layout()
plt.show()




# %% [markdown]
# Due to skeweness we need to get rid of them to make our distribution looks normal.I use log transformation for this.Log trnasformation mostly used in right skeweness

# %% [markdown]
# ### Handle age skeweness

# %%

log_age,_ = stats.boxcox(df['age'])
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(log_age, bins=30,color='green', edgecolor='k', alpha=0.7)
plt.title(f'Histogram of age')
plt.xlabel('age')
plt.ylabel('Frequency')


plt.subplot(1, 2, 2)
stats.probplot(log_age, dist="norm", plot=plt )
plt.title(f'Q-Q Plot of age')

plt.tight_layout()
plt.show()




# %% [markdown]
# ### One hot encoding and label encoding
# 

# %%
df['month'] = LabelEncoder().fit_transform(df['month'])
df['day_of_week'] = LabelEncoder().fit_transform(df['day_of_week'])
df['education'] = LabelEncoder().fit_transform(df['education'])
df['job'] = LabelEncoder().fit_transform(df['job'])
df = pd.get_dummies(df, columns=['marital','contact','default','housing','loan','poutcome'], drop_first=True)
df.head(10)

# %% [markdown]
# ### Standardize other columns

# %%


from sklearn.discriminant_analysis import StandardScaler


numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


plt.figure(figsize=(20, 13))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(4, 5, i)
    sns.histplot(df[column], kde=True, color='green', label='Skewness: {:.2f}'.format(df[column].skew()))
    plt.title(f'Histogram of {column}')

plt.tight_layout()
plt.show()


# %%
df.info()

# %% [markdown]
# ## Descretize

# %%
# Discretize features
from sklearn.preprocessing import KBinsDiscretizer


features_to_discretize = ['age','y']
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

df[features_to_discretize] = discretizer.fit_transform(df[features_to_discretize])

# Visualize the effect of discretization
plt.figure(figsize=(14, 8))
for i, column in enumerate(features_to_discretize, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[column], kde=False, bins=5)
    plt.title(f'Discretized {column}')

plt.tight_layout()
plt.show()

# %% [markdown]
# # Correlation

# %%
correlation_matrix = df.corr()
# Plot the correlation matrix using a heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
correlations_with_y = correlation_matrix['y'].sort_values(ascending=False)




# %% [markdown]
# ### Perform PCA
# 

# %%
X = df.drop(columns=['y'])
y = df['y']
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)



# %% [markdown]
# 
# 
# # Optional: DataFrame with principal components

# %%

df_pca = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
df_pca.head()

# %%
# Explained variance
print("Explained variance by each component:", pca.explained_variance_ratio_)


# %% [markdown]
# ### Plotting the principal components
# 

# %%
plt.figure(figsize=(10, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - PC1 vs PC2')
plt.grid(True)
plt.show()

# %% [markdown]
# ### Train-Test Splitting

# %%
X_train, X_test, y_train, y_test = train_test_split(principal_components, y, test_size=0.2, random_state=4)


# %% [markdown]
# ### Train SVM and Logistic Regression Models

# %% [markdown]
# #### SVM

# %%
# print("Training set class distribution:\n", pd.Series(y_train).value_counts())
# print("Test set class distribution:\n", pd.Series(y_test).value_counts())

print(y_train[y_train > 0].count())


# %%

svm_model = SVC(class_weight='balanced',C= 10, gamma=0.1, kernel='linear',verbose=True)
svm_model.fit(X_train, y_train)

# %% [markdown]
# ## SVM Evaluation
# 

# %%
y_pred_svm = svm_model.predict(X_test)
cm_svm = confusion_matrix(y_test, y_pred_svm)
desicion_score_svm = svm_model.decision_function(X_test)
desicion_score_svm


# %%
# from sklearn.model_selection import GridSearchCV


# param_grid = {
#     'C': [0.1, 1],
#     'gamma': [1, 0.1, ],
#     'kernel': [ 'linear']
# }

# grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
# bp=grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)

# %% [markdown]
# #### LR

# %%
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# %% [markdown]
# # Logistic Regression Evaluation

# %%
y_pred_lr = lr_model.predict(X_test)
cm_lr = confusion_matrix(y_test, y_pred_lr)

# %% [markdown]
# # Convert confusion matrix to DataFrame for better visualization

# %%
df_cm_svm = pd.DataFrame(cm_svm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
df_cm_lr = pd.DataFrame(cm_lr, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

# %%
# Plot confusion matrix for SVM
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm_svm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Predicted Negative', 'Predicted Positive'], 
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot confusion matrix for Logistic Regression
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Predicted Negative', 'Predicted Positive'], 
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# %%
print(classification_report(y_test,y_pred_svm))

# %% [markdown]
# ### Model Evaluation Metrics
# 
# #### SVM
# 
# - **Accuracy:** `accuracy_score(y_test, y_pred_svm)`
# - **Classification Report:**
# 

# %% [markdown]
# 
# #### Logistic Regression
# 
# - **Accuracy:** `accuracy_score(y_test, y_pred_lr)`
# - **Classification Report:**
# 

# %% [markdown]
# 
# ### Summary Table
# 
# | Metric                    | SVM                  | Logistic Regression |
# |---------------------------|----------------------|---------------------|
# | **Accuracy**              | `accuracy_score(y_test, y_pred_svm)` | `accuracy_score(y_test, y_pred_lr)` |
# | **Classification Report** | `classification_report(y_test, y_pred_svm)` | `classification_report(y_test, y_pred_lr)` |
# | **Confusion Matrix**      | `confusion_matrix(y_test, y_pred_svm)` | `confusion_matrix(y_test, y_pred_lr)` |
# 


