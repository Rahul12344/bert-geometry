import numpy as np

def diameter(X):
    norms = np.linalg.norm(X, axis=1)
    max_norm = np.max(norms)
    min_norm = np.min(norms)
    diameter = max_norm - min_norm
    print(diameter)
    return diameter


def estimate(diameters):
    lower_bound = np.zeros((len(diameters), len(diameters)))
    upper_bound = np.zeros((len(diameters), len(diameters)))
    for i in range(len(diameters)):
        for j in range(i, len(diameters)):
            lower_bound[i][j] = 0.5*np.abs(diameters[i]-diameters[j])
            lower_bound[j][i] = lower_bound[i][j]
            upper_bound[i][j] = 0.5 * np.max([diameters[i], diameters[j]])
            upper_bound[j][i] = upper_bound[i][j]
            
    return lower_bound