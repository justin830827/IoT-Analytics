import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt

def basic_stats(mean, var):
    df = pd.DataFrame({'Mean': mean, 'Variance': var}, columns=["Mean", "Variance"], index=['x1', 'x2', 'x3', 'x4', 'x5'])
    print(df)
    df.to_csv("basic_stats.csv")

def remove_outliers(data):
    # Remove the outliers by the z-score is greater than 3
    z = np.abs(stats.zscore(data))
    data = data[(z < 3).all(axis=1)]
    
    return data

def linear_regression(X,Y):
    X = sm.add_constant(X)
    model = sm.OLS(Y,X).fit()
    print (model.summary())
    y_predict = model.predict(X)
    return y_predict

def compute_error(y,y_predict):
    return np.subtract(y,y_predict)

def chisquare_test(x):
    print ("\nRunning chisquare test.........")
    k2, p = stats.normaltest(x)
    alpha = 0.05
    print('We have p-value is {}'.format(p))
    
    if p > alpha:
        print("Not Significant Result, failed to reject the null hypothesis")
    else:
        print("Significant Result, the null hypothesis rejected")  
    return 0

def plot_hist(x, title):
    plt.hist(x, bins=20)
    plt.title('{} histgram'.format(title))
    plt.savefig('{} histgram.png'.format(title))
    plt.close()

def plot_cor_matrix(data):
    corr = data.corr()
    plt.matshow(corr,10)
    for (i, j), z in np.ndenumerate(corr):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    plt.xticks(range(len(data.columns)), data.columns)
    plt.yticks(range(len(data.columns)), data.columns)
    plt.title('Correlation Matrix')
    plt.savefig("Correlation matrix.png")
    plt.close()

def plot_LR(X, Y, Y_pred, title):
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, 'r+' , color='red')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('y')
    plt.savefig('{}.png'.format(title))
    plt.close()

def plot_qqplot(error, var, title):
    sm.qqplot(error,loc=0,scale=np.sqrt(var),line='q')
    plt.title('{} Q_Q plot'.format(title))
    plt.savefig('{} Q_Q plot.png'.format(title))
    plt.close()

def plot_scatter(error, y_predict, title):
    plt.scatter(error,y_predict)
    plt.xlabel('residuals')
    plt.ylabel('y')
    plt.title('{} scatter plot.png'.format(title))
    plt.savefig('{} scatter plot.png'.format(title))
    plt.close()


    
def main():
    # load Dataset
    data = pd.read_csv('./whu24.csv',names= ['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
    
    # Run Task 1
    print ('\n-------------------------------------------Running Task 1-------------------------------------------')
    # Dataframe slicing
    data1 = data.iloc[:,:5]

    # Calculate histgram, mean, variance
    columns = data1.columns
    mean, variance = [], []
    for column in columns:
        xmean = np.mean(data1[column])
        xvariance = np.var(data1[column])
        mean.append(xmean)
        variance.append(xvariance)
        plot_hist(data1[column], column)
    
    # Show the mean and variance
    basic_stats(mean, variance)

    # Correlation matrix
    plot_cor_matrix(data)

    # Run Task 2
    print ('\n-------------------------------------------Running Task 2-------------------------------------------')
    
    # Simple linear regression
    print("Simple linear regression....")
    # Dataframe extraction
    data2_1 = data[['x1','y']]
    data2_1 = remove_outliers(data2_1)
    x1,y = data2_1['x1'], data2_1['y']
    
    # Task 2.1 & 2.2 Determine a0, a1 and variance, R squared, p-value and F value
    y_predict = linear_regression(x1,y)

    # Task 2.3 Plot the regression line against the data
    plot_LR(x1,y,y_predict,"Simple Linear Regression")

    # Task 2.4 Residuals analysis
    error = compute_error(y,y_predict)
    error_var = np.var(error)
    print ('\nError variance: {}'.format(error_var))
    chisquare_test(error)
    plot_hist(error, "Residuals(simple)")
    plot_qqplot(error, error_var, "Residuals(simple)")
    plot_scatter(error, y_predict, "Residuals(simple)")

    # Task 2.7
    print("\n\nPolynomial Regression....")
    data2_7 = data[['x1','y']]
    data2_7 = remove_outliers(data2_7)
    data2_7['x1^2'] = data2_7['x1'] ** 2
    X,y = data2_7[['x1','x1^2']], data2_7['y']

    # Determine a0, a1 and variance, R squared, p-value and F value
    y_predict = linear_regression(X,y)

    # Plot the regression line against the data
    plot_LR(X['x1'],y,y_predict,"Polynomial Regression")

    # Residuals analysis
    error = compute_error(y,y_predict)
    error_var = np.var(error)
    print ('\nError variance: {}'.format(error_var))
    chisquare_test(error)
    plot_hist(error, "Residuals(poly)")
    plot_qqplot(error, error_var, "Residuals(poly)")
    plot_scatter(error, y_predict, "Residuals(poly)")

    print ('\n-------------------------------------------Running Task 3-------------------------------------------')

    print("First attempt multivariate regression...")
    data3 = remove_outliers(data)
    X,y = data3.iloc[:,:5], data3['y']

    # Task3.1 & 3.2 Determine coefficients and variance, 
    y_predict = linear_regression(X,y)

    # Residuals analysis
    error = compute_error(y,y_predict)
    error_var = np.var(error)
    print ('\nError variance: {}'.format(error_var))
    chisquare_test(error)
    plot_hist(error, "Residuals(multi)")
    plot_qqplot(error, error_var, "Residuals(multi)")
    plot_scatter(error, y_predict, "Residuals(multi)")

    print("\n\nBetter attempt multivariate regression...")
    data3_3 = data[['x1','x5','y']]
    data3_3 = remove_outliers(data3_3)
    X,y = data3_3[['x1','x5']], data3_3['y']

    # Task3.1 & 3.2 Determine coefficients and variance, 
    y_predict = linear_regression(X,y)

    # Residuals analysis
    error = compute_error(y,y_predict)
    error_var = np.var(error)
    print ('\nError variance: {}'.format(error_var))
    chisquare_test(error)
    plot_hist(error, "Residuals(multi_improved)")
    plot_qqplot(error, error_var, "Residuals(multi_improved)")
    plot_scatter(error, y_predict, "Residuals(multi_improved)")


if __name__ == "__main__":
    main()