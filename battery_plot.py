import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Read the data from a csv file
data = pd.read_csv('data_output.csv')
# Plot the fourth column on the y axis, and the first on the x axis
# Save the figure
sns_plot = sns.lineplot(data=data,x='Time',y='Power')
sns_plot = sns_plot.get_figure()
sns_plot.savefig('battery_plot.png')
