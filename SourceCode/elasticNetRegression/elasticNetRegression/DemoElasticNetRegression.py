
#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

#Initial dataset observations
df = pd.read_csv('Dataset/winequality-red.csv')
#print(df.head())
#df.info()
df['quality'].sort_values().unique()

def plot_wine_quality_histogram(quality):
    unique_vals = df['quality'].sort_values().unique()
    plt.xlabel("Quality")
    plt.ylabel("Count")
    plt.hist(quality.values, bins=np.append(unique_vals, 9), align='left')
plot_wine_quality_histogram(df['quality'])

def plot_features_correlation(df):
    plt.figure(figsize=(7.5,6))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.set(font_scale=1)
    corr_mat = df.corr()
    ax = sns.heatmap(data=corr_mat, annot=True, fmt='0.1f', vmin=-1.0, vmax=1.0, center=0.0, xticklabels=corr_mat.columns, yticklabels=corr_mat.columns, cmap="Blues")
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
plot_features_correlation(df)

#Data preprocessing
#We first split our data into training (80%) and test (20%) sets.
y = df.quality
X = df.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


#======Initial approaches: Regression

#--Create metrics
def scores_results(y_train, y_test, y_pred_train, y_pred_test):
    #this function will provide us with accuracy and mse scores for training and test sets
    y_pred_train_round = np.round(y_pred_train)
    y_pred_test_round = np.round(y_pred_test)
    accuracy = [accuracy_score(y_train, y_pred_train_round), accuracy_score(y_test, y_pred_test_round)]
    mse_with_rounding = [mean_squared_error(y_train, y_pred_train_round), mean_squared_error(y_test, y_pred_test_round)]
    results = pd.DataFrame(list(zip(accuracy, mse_with_rounding)), columns = ['accuracy score', 'mse (after rounding)'], index = ['train', 'test'])
    return results

#--Linear regression
def linear_reg(X_train_scaled, X_test_scaled, y_train, y_test):
    # basic linear regression
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X_train_scaled, y_train)
    y_pred_train = lm.predict(X_train_scaled)
    y_pred_test = lm.predict(X_test_scaled)
    global metrics_lr #store this for a later comparison between different methods
    metrics_lr = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return scores_results(y_train, y_test, y_pred_train, y_pred_test)

scores =linear_reg(X_train_scaled, X_test_scaled, y_train, y_test)
print("Metric Linear regression: ")
print(scores)

#Adding regularisation
# to avoid overfitting
#3 different popular regularisation methods: Lasso (L1), Ridge (L2) and Elastic Net (L1 and L2).
def lasso_reg(X_train_scaled, X_test_scaled, y_train, y_test):
    from sklearn.linear_model import LassoCV
    n_alphas = 5000
    alpha_vals = np.logspace(-6, 0, n_alphas)
    lr = LassoCV(alphas=alpha_vals, cv=10, random_state=0)
    lr.fit(X_train_scaled, y_train)
    y_pred_train = lr.predict(X_train_scaled)
    y_pred_test = lr.predict(X_test_scaled)
    metrics_lasso = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return metrics_lasso
def elastic_net_reg(X_train_scaled, X_test_scaled, y_train, y_test):
    from sklearn.linear_model import ElasticNetCV
    n_alphas = 300
    l1_ratio = [.1, .3, .5, .7, .9]
    rr = ElasticNetCV(n_alphas=n_alphas, l1_ratio=l1_ratio, cv=10, random_state=0)
    rr.fit(X_train_scaled, y_train)
    y_pred_train = rr.predict(X_train_scaled)
    y_pred_test = rr.predict(X_test_scaled)
    metrics_en = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return metrics_en
def ridge_reg(X_train_scaled, X_test_scaled, y_train, y_test):
    from sklearn.linear_model import RidgeCV
    n_alphas = 100
    alpha_vals = np.logspace(-1, 3, n_alphas)
    rr = RidgeCV(alphas=alpha_vals, cv=10)
    rr.fit(X_train_scaled, y_train)
    y_pred_train = rr.predict(X_train_scaled)
    y_pred_test = rr.predict(X_test_scaled)
    metrics_ridge = [accuracy_score(y_test, np.round(y_pred_test)), mean_squared_error(y_test, y_pred_test), r2_score(y_test, y_pred_test)]
    return metrics_ridge

metrics_lasso = lasso_reg(X_train_scaled, X_test_scaled, y_train, y_test)
metrics_en = elastic_net_reg(X_train_scaled, X_test_scaled, y_train, y_test)
metrics_ridge = ridge_reg(X_train_scaled, X_test_scaled, y_train, y_test)
finalscores = pd.DataFrame(list(zip(metrics_lr, metrics_lasso, metrics_en, metrics_ridge)), columns = ['lr', 'lasso', 'el net', 'ridge'], index = ['acc','mse','r2'])
print("Linear regression (lr) | Lasso (L1) | Ridge (L2) | Elastic Net (L1 and L2)")
print(finalscores)
