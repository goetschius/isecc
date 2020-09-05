import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from matplotlib.offsetbox import AnchoredText
from scipy.stats import norm

my_data = np.loadtxt('temp.txt')
N_points = len(my_data)
N_unique = np.unique(my_data).shape

# For gaussian curve
mu, std = norm.fit(my_data)
percentile_5 = np.percentile(my_data, 5)
percentile_95 = np.percentile(my_data, 95)

n_bins = 75
print(my_data, N_points, N_unique[0])
print(np.percentile(my_data, 5), np.percentile(my_data, 95))
print(np.mean(my_data))

# Pliot the histogram
plt.hist(my_data, bins=n_bins, density=True, alpha=0.6)

# Plot it
xmin, xmax = plt.xlim()
xmin, xmax = (0.95, 1.05)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
#plt.plot(x, p, 'k', linewidth=2 )
plt.plot(x, p, '--', linewidth=2, color='black' )


plt.title('HPV2020 Diameter')
plt.xlabel('O/E')
plt.ylabel('Count')

annotation = " mean: %.3f\n std: %.3f\n bins: %.0f\n datapoints: %.0f\n\n  5th percentile: %.3f\n 95th percentile: %.3f" % (np.mean(my_data), np.std(my_data), n_bins, N_points, percentile_5, percentile_95)

plt.annotate(annotation, xy=(0.65,0.65), xycoords='axes fraction')

plt.show()
