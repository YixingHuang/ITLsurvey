import math
##N=5, all random
# means = [46.50666667,  46.75333333,  46.12666667,  42.47333333,   48.20666667,   46.08666667,   47.84,  47.78 ]
#
# std = [2.261359575,1.411952264,1.875786425,2.256933124,1.825855177,1.990794909,1.980874067,2.020139975]

#N=5, high initialization
means = [47.10666667,46.66666667,46.97333333,45.69333333,49.91333333,48.54666667,50.66666667,48.24]
std=[1.570181921,1.925748101,1.437174751,1.458136111,1.165815782,1.911802423,1.083205121,1.530742439]

# #N=5, low initialization
# means = [47.07333333, 46.26, 46.03333333, 38.52, 47.52, 44.53333333, 44.82, 46.11333333]
# std = [1.734424944, 1.922749468, 1.584479319, 1.057779066, 1.58492902, 1.989166059, 2.189221244, 2.011472841]


# #N = 5, high initialization, four centers for training only
# means = [46.88666667, 46.99333333, 47.52666667, 46.50666667, 47.58, 47.26, 49.25333333,48.11333333]
# std = [1.601450492, 1.451261837, 1.870632077, 1.389351953, 1.280463278, 1.496801187, 1.238389759, 1.660267716]

# #N=3, all random
# means = [47.74666667, 48.14666667, 47.54, 42.85333333, 46.99333333, 45.47333333, 46.43333333, 46.89333333]
# std = [1.995120484, 1.749824293, 1.67673739, 1.841987932, 2.133142951, 2.344462784, 2.210372622, 2.361024519]

# #N=3, high initialization
# means = [48, 48.30666667, 48.11333333, 44.9, 48.49333333, 46.56, 46.6, 45.04]
# std =[1.861404735, 1.894262359, 1.650649472, 1.976935978, 1.649019214, 2.270287844, 3.742763132, 4.384109236]

# #N=3, low initialization
# means = [47.43333333, 47.22666667,46.34666667, 40.98666667, 47.76, 44.02666667, 44.97333333, 46.46666667]
# std = [1.706040654, 1.839590159, 1.460215698, 0.969796756, 1.407761735, 2.111207096, 1.411023921, 1.484943592]

# ## IID data all random
# means = [49.48965517, 49.62068966, 49.12413793, 44.24827586, 47.95862069, 46.35862069, 48.12413793, 47.35172414]
# std = [2.592068235, 2.454933707, 2.421577876, 2.047754988, 2.150335064, 1.637401522, 2.396076432, 2.129825729]

# ## IID data high initialization
# means = [51.44, 51.71333333, 51.19333333, 47.53333333, 50.43333333, 48.72666667, 51.1, 49.07333333]
# std = [1.857992985, 1.729806314, 1.548733261, 1.255425009, 1.120447365, 1.592641123, 0.890950674, 1.219930279]

# # ## IID data low initialization
# means = [49.06666667,		49.1,			48.84, 42.96666667,		49.19333333,45.64666667,46.63333333, 48.14]
# std=[1.940020144, 1.618641406,	1.430119358, 0.879001523,1.417606855,2.134565018,1.530457072,1.516938841]

# # CWT  Non-IID  N=5, high initialization, validation = 5
# means = [50.75333333, 50.95333333, 50.01333333, 48.82666667, 52.5, 52.3, 52.98666667, 52.66666667]
# std = [1.228043954, 0.89701933, 1.186049174, 0.562588435, 0.953758446, 1.001722654, 1.026454673, 0.874675638]

# #CWT N=5, low initialization, validation = 5
# means = [52.11333333, 52.24, 51.80666667, 44.65333333, 52.49333333, 53.45333333, 53.62666667, 53.36]
# std = [1.645628134, 0.987368498, 1.522686676, 1.916846069, 1.744139613, 1.68148812, 1.231464879, 1.245571466]

# # #CWT, low initialization, validation = 10
# means = [53.1, 52.71333333, 52.37333333, 47.75333333, 53.65333333, 50.56875, 54.9, 54.52]
# std = [1.709002532, 1.72421563, 1.522913119, 1.848528461, 1.400426864, 2.81521643, 1.773949422, 1.586668598]

# #CWT, high initialization, validation = 10
# means = [54.76666667, 54.16666667, 53.11333333, 50.24666667, 54.56666667, 51.47333333, 56.02, 55.2]
# std = [1.44707394, 1.38373142, 1.531898379, 0.811441176, 1.161548709, 2.165232063, 1.507086708, 1.337006435]

# #CWT, all random, validation = 5
# means = [57.29655172, 56.18666667, 56.02666667, 55.47333333, 44.82, 56.33333333, 56.61538462, 55.344, 55.232]
# std = [1.540520032, 1.54467569, 1.719449257, 1.222189481, 3.563938155, 1.894120458, 2.06527331, 2.436883255, 2.337862271]

#CWT, all random. validation = 5, N=5
# means = [55.62, 39.12, 50.2, 50.73333333, 49.54, 40.05333333, 54.66666667, 54.9, 54.17333333, 54.42]
# std = [1.54, 2.26, 1.61629632, 1.745404475, 1.767249958, 3.583846261, 2.069371608, 1.788661607, 1.787877442, 1.80447719]

# #CWT all random, validation = 5, N=3
# means = [52.50666667, 52.17142857, 51.14666667, 42.23333333, 53.60666667, 53.68666667, 52.72, 52.67333333 ]
# std = [1.767705182, 1.975939934, 1.767080845, 4.139992781, 2.101220226, 1.846288604, 1.867766507, 1.828120069]

# # #CWT all random, validation = 5, N=4,5
#
# means = [50.51333333, 50.19333333, 49.58, 39.64666667, 52.63333333, 52.78, 51.66666667, 51.25333333]
# std = [1.981767468, 1.82962844, 1.522565892, 3.340321411, 2.132237506, 1.74462722, 3.082020534, 2.828394614]

low = [x - 1.96 * y/math.sqrt(30.0) for x, y in zip(means, std)]
high = [x + 1.96 * y/math.sqrt(30.0) for x, y in zip(means, std)]
low90 = [x - 1.645 * y/math.sqrt(30.0) for x, y in zip(means, std)]
high90 = [x + 1.645 * y/math.sqrt(30.0) for x, y in zip(means, std)]
print(low)
print(high)
print(low90)
print(high90)