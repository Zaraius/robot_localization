from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

def main():
    particles = np.array([[0,30],[20,50],[30,60],[70,0],[80,50],[0,0],[99,0],[0,99],[99,99]])
    weights = np.array([1,0,8,1,4,0,0,0,0])
    x_grid,y_grid = np.mgrid[0:100,0:100]
    interp = griddata(particles,weights,(x_grid,y_grid),method="linear")

    indices = list(np.ndindex(100,100,10))
    values = interp.ravel()
    probabilities = (values / np.sum(values)).tolist()
    print(probabilities)

    plt.imshow(interp)
    plt.show()

if __name__ == "__main__":
    main()