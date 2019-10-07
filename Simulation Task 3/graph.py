import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mean_df = pd.read_csv('mean_outputs.csv',names=["RT_mean", "RT_mean_lower_CI", "RT_mean_upper_CI", "nonRT_mean", "nonRT_mean_lower_CI", "RT_mean_upper_CI"])
percentile_df = pd.read_csv('95thPercentile_outputs.csv',names=["RT_95th", "RT_95th_lower_CI", "RT_95th_upper_CI", "nonRT_95th", "nonRT_95th_lower_CI", "RT_95th_upper_CI"])

# mean_df.to_csv("mean_outputs.csv")
# percentile_df.to_csv("95thPercentile_outputs.csv")

# width of the bars
barWidth = 0.3
 
# Input means as height of bar
RT_bar = mean_df["RT_mean"].tolist()
nonRT_bar = mean_df["nonRT_mean"].tolist()

# Get errors
RT_yerr = np.subtract(RT_bar, mean_df["RT_mean_lower_CI"].tolist())
nonRT_yerr = np.subtract(nonRT_bar, mean_df["nonRT_mean_lower_CI"].tolist())
 
# The x position of bars
r1 = np.arange(len(RT_bar))
r2 = [x + barWidth for x in r1]
 
# Create blue bars
plt.bar(r1, RT_bar, width = barWidth, color = 'blue', edgecolor = 'black', yerr=RT_yerr, capsize=7, label='RT mean')
plt.bar(r2, nonRT_bar, width = barWidth, color = 'red', edgecolor = 'black', yerr=nonRT_yerr, capsize=7, label='nonRT mean')
 
# Plot mean graph
plt.xticks([r + barWidth for r in range(len(RT_bar))], ['10', '15', '20', '25', '30', '35', '40'])
plt.ylabel('Mean')
plt.xlabel('MIAT of nonRT')
plt.legend()
plt.savefig('mean_plot.png')
plt.close()

# Input means as height of bar
RT_bar = percentile_df["RT_95th"].tolist()
nonRT_bar = percentile_df["nonRT_95th"].tolist()

# Get errors
RT_yerr = np.subtract(RT_bar, percentile_df["RT_95th_lower_CI"].tolist())
nonRT_yerr = np.subtract(nonRT_bar, percentile_df["nonRT_95th_lower_CI"].tolist())
 
# The x position of bars
r1 = np.arange(len(RT_bar))
r2 = [x + barWidth for x in r1]
 
# Create blue bars
plt.bar(r1, RT_bar, width = barWidth, color = 'blue', edgecolor = 'black', yerr=RT_yerr, capsize=7, label='RT 95th percentile')
plt.bar(r2, nonRT_bar, width = barWidth, color = 'red', edgecolor = 'black', yerr=nonRT_yerr, capsize=7, label='nonRT 95th percentile')
 
# Plot 95th percentile graph
plt.xticks([r + barWidth for r in range(len(RT_bar))], ['10', '15', '20', '25', '30', '35', '40'])
plt.ylabel('95th Percentile')
plt.xlabel('MIAT of nonRT')
plt.legend()
plt.savefig('95percentile_plot.png')
plt.close()
