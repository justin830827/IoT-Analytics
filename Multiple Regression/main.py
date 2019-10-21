import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt

def task1(data):
    # Dataframe slicing
    xdata = data.iloc[:,:5]

    # Calculate histgram, mean, variance
    columns = xdata.columns
    mean, variance = [], []
    for column in columns:
        xmean = np.mean(xdata[column])
        xvariance = np.var(xdata[column])
        mean.append(xmean)
        variance.append(xvariance)
        print (f'Mean of {column}: {xmean}')
        print (f'Variance of {column}: {xvariance}')
        plt.hist(xdata[column], bins=20)
        plt.title(f'{column} histgram')
        plt.savefig(f'{column} histgram.png')
        plt.close()

    # Correlation matrix
    corr = data.corr()
    plt.matshow(corr)
    for (i, j), z in np.ndenumerate(corr):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    plt.savefig("correlation matrix.png")
    plt.close()

    
def main():
    df = pd.read_csv('./whu24.csv',names= ['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
    
    # Run Task 1
    print ('--------------------Running Task 1-----------------------------')
    #task1(df)
    print ('--------------------Running Task 2-----------------------------')
    # Remove X1 outliers
    length = len(df['x1'])
    #print (f'Numbers of before removing outlier {length}')
    df = df[ (np.abs(stats.zscore(df.x1)) <= 3) & (np.abs(stats.zscore(df.x1)) <= 3)]
    length = len(df['x1'])
    #print (f'Numbers of after removing outlier {length}')
    x1 = df['x1'].to_numpy()
    y = df['y'].to_numpy()
    slope, intercept, r_value, p_value, std_err = stats.linregress(x1,y)
    model = LinearRegression().fit(x1.reshape(-1,1), y)
    y_predict = model.predict(x1.reshape(-1,1))
    error_variance = np.var(np.subtract(y,y_predict))
    r_sq = model.score(x1.reshape(-1,1), y)
    print(f'Intercept(a0): {model.intercept_}')
    print(f'Slope(a1): {model.coef_[0]}')
    print (f'Error variance: {error_variance}')
    print(f'R squard: {r_sq}')
    print(f'p value: {p_value}')
    print(sm.OLS(y, x1.reshape(-1,1)).fit().summary())

    
    

    print ('--------------------Running Task 3-----------------------------')



if __name__ == "__main__":
    main()