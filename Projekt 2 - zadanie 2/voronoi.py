import numpy as np
import pandas as pd
import torch

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as mpl

import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def voronoi_finite_polygons_2d(vor, radius=None):

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def voronoi(vor, etykiety, radius=None):
    regions, vertices = voronoi_finite_polygons_2d(vor)


    norm = mpl.colors.Normalize(vmin=0, vmax=max(etykiety)+1, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.RdYlGn)
    

    voronoi_plot_2d(vor, show_points=True, show_vertices=False, s=1)
    
    for r in range(len(regions)):
        region = regions[r]
        if not -1 in region:
            polygon = [vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(etykiety[r]))
            
    plt.show()


def plot_decision_boundary(X, func, y_true=None)-> plt.figure:
    fig, ax = plt.subplots()
     # Definiujemy zakres dla osi x i y
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Tworzymy siatkę punktów w celu wygenerowania granicy decyzyjnej
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Obliczamy etykiety dla każdego punktu w siatce
    meshgrid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.double)
    meshgrid_tensor.reshape(-1,28,28)
    Z = func(meshgrid_tensor)
    Z = torch.argmax(Z,dim=1).detach().numpy().reshape(xx.shape)
    print(Z)

    #Predykcja etykiet dla danych testowych podanych w funkcji
    if y_true is None:
        y_tested = torch.argmax(func(X), dim=1).numpy()
    else: y_tested=y_true

    # Rysujemy kontury granicy decyzyjnej
    ax.contourf(xx, yy, Z, alpha=0.4)
    
    # Rysujemy punkty treningowe
    ax.scatter(X[:, 0], X[:, 1], c=y_tested, s=20, edgecolors='k')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()
    return fig

def plot_decision_boundary_ax(X, axes_dec, func, y_true=None)-> None:
     # Definiujemy zakres dla osi x i y
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Tworzymy siatkę punktów w celu wygenerowania granicy decyzyjnej
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Obliczamy etykiety dla każdego punktu w siatce
    Z = func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #print(Z)

    #Predykcja etykiet dla danych testowych podanych w funkcji
    if y_true is None:
        y_tested = func(X)
    else: y_tested=y_true

    # Rysujemy kontury granicy decyzyjnej
    axes_dec.contourf(xx, yy, Z, alpha=0.4)
    
    # Rysujemy punkty treningowe
    axes_dec.scatter(X[:, 0], X[:, 1], c=y_tested, s=20, edgecolors='k')

if __name__ == "__main__":
    pass
