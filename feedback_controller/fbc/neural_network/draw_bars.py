import matplotlib.pyplot as plt
import numpy as np

# Example data
baseline = [0.02455427675137442, 0.030574818868336257, 0.025107076947724478, 0.024333005872385464, 0.031219207403771508]
our_model =  [0.0431264978983281, 0.03827230745285744, 0.04018639010658122, 0.03994613497793411, 0.03508697204673167]

# Calculating means and standard deviations
means = [np.mean(our_model), np.mean(baseline)]
std_devs = [np.std(our_model), np.std(baseline)]

# Creating the plot
fig, ax = plt.subplots()

# Bar positions 
bar_positions = np.arange(len(means))

# Colors for the bars
colors = ['blue', 'orange']

# Plotting the bars
bars = ax.bar(bar_positions, means, yerr=std_devs, capsize=10, tick_label=['our model', 'baseline'], color=colors)

# Adding labels and title
ax.set_ylabel('Value')
ax.set_title('Mean and Standard Deviation of Offline Loss')

# Adding a legend
ax.legend(bars, ['our model', 'baseline'])

# Showing the plot
plt.show()