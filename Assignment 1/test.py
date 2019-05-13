import numpy as np;

w = np.float32([[1,2,3,4]]).T

print(np.transpose(w).dot(w))