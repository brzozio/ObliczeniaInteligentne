import matplotlib.pyplot as plt
import numpy as np

file_path: str = r"2_1.csv"
Data = np.genfromtxt(file_path, delimiter=";")
print(Data)
plt.scatter(Data[:,0],Data[:,1])
plt.show()