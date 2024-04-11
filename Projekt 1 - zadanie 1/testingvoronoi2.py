import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import Voronoi, voronoi_plot_2d



Data: np.ndarray = np.zeros((6,300,3))
Data[0]= np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\1_1.csv", delimiter=';')
Data[1]= np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\1_2.csv", delimiter=';')
Data[2]= np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\1_3.csv", delimiter=';')
Data[3]= np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\2_1.csv", delimiter=';')
Data[4]= np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\2_2.csv", delimiter=';')
Data[5]= np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\2_3.csv", delimiter=';')

temp_CSV = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\2_1.csv", delimiter=';')
temp_X = temp_CSV[:,0:2]

temp_CSV_1 = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\1_1.csv", delimiter=';')
temp_X_1 = temp_CSV_1[:,0:2]
temp_CSV_2 = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\1_2.csv", delimiter=';')
temp_X_2 = temp_CSV_2[:,0:2]
temp_CSV_3 = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\1_3.csv", delimiter=';')
temp_X_3 = temp_CSV_3[:,0:2]

temp_CSV_21 = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\2_1.csv", delimiter=';')
temp_X_21 = temp_CSV_21[:,0:2]
temp_CSV_22 = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\2_2.csv", delimiter=';')
temp_X_22 = temp_CSV_22[:,0:2]
temp_CSV_23 = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\2_3.csv", delimiter=';')
temp_X_23 = temp_CSV_23[:,0:2]

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

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


    norm = mpl.colors.Normalize(vmin=-1, vmax=max(etykiety)+1, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues_r)

    voronoi_plot_2d(vor, show_points=True, show_vertices=False, s=1)
    #for r in range(len(vor.point_region)):
    #print(f"POINT REGION: {vor.point_region}")
    for r in range(len(regions)):
        region = regions[r]
        if not -1 in region:
            polygon = [vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(etykiety[r]))
            

    plt.show()

def voronoi_fig(vor, etykiety, radius=plt.figure):
    fig, ax = plt.subplots()
    regions, vertices = voronoi_finite_polygons_2d(vor)


    norm = mpl.colors.Normalize(vmin=-1, vmax=max(etykiety)+1, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues_r)

    voronoi_plot_2d(vor, show_points=True, show_vertices=False, s=1)
    #for r in range(len(vor.point_region)):
    #print(f"POINT REGION: {vor.point_region}")
    for r in range(len(regions)):
        region = regions[r]
        if not -1 in region:
            polygon = [vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(etykiety[r]))
    
    return fig



if __name__ == "__main__":
    choice = 3
    if choice is 1:

        kluster_KMeans_1_BEST = KMeans(n_clusters=2)
        kluster_KMeans_1_BEST.fit_predict(temp_X_1)
        vor_1 = Voronoi(temp_X_1)
        etykiety_1_BEST  = kluster_KMeans_1_BEST.labels_
        voronoi(vor_1,etykiety_1_BEST)

        kluster_KMeans_1_WORST = KMeans(n_clusters=4)
        kluster_KMeans_1_WORST.fit_predict(temp_X_1)
        vor_1 = Voronoi(temp_X_1)
        etykiety_1_WORST = kluster_KMeans_1_WORST.labels_
        voronoi(vor_1,etykiety_1_WORST)

        kluster_KMeans_2_BEST = KMeans(n_clusters=2)
        kluster_KMeans_2_BEST.fit_predict(temp_X_2)
        vor_2 = Voronoi(temp_X_2)
        etykiety_2_BEST  = kluster_KMeans_2_BEST.labels_
        voronoi(vor_2,etykiety_2_BEST)
        
        kluster_KMeans_2_WORST = KMeans(n_clusters=4)
        kluster_KMeans_2_WORST.fit_predict(temp_X_2)
        vor_2 = Voronoi(temp_X_2)
        etykiety_2_WORST = kluster_KMeans_2_WORST.labels_
        voronoi(vor_2,etykiety_2_WORST)

        kluster_KMeans_3_BEST = KMeans(n_clusters=3)
        kluster_KMeans_3_BEST.fit_predict(temp_X_3)
        vor_3 = Voronoi(temp_X_3)
        etykiety_3_BEST  = kluster_KMeans_3_BEST.labels_
        voronoi(vor_3,etykiety_3_BEST)
        
        kluster_KMeans_3_WORST = KMeans(n_clusters=6)
        kluster_KMeans_3_WORST.fit_predict(temp_X_3)
        vor_3 = Voronoi(temp_X_3)
        etykiety_3_WORST = kluster_KMeans_3_WORST.labels_
        voronoi(vor_3,etykiety_3_WORST)





        kluster_KMeans_21_BEST = KMeans(n_clusters=2)
        kluster_KMeans_21_BEST.fit_predict(temp_X_21)
        vor_21 = Voronoi(temp_X_21)
        etykiety_21_BEST  = kluster_KMeans_21_BEST.labels_
        voronoi(vor_21,etykiety_21_BEST)

        kluster_KMeans_21_WORST = KMeans(n_clusters=9)
        kluster_KMeans_21_WORST.fit_predict(temp_X_21)
        vor_21 = Voronoi(temp_X_21)
        etykiety_21_WORST = kluster_KMeans_21_WORST.labels_
        voronoi(vor_21,etykiety_21_WORST)

        kluster_KMeans_22_BEST = KMeans(n_clusters=4)
        kluster_KMeans_22_BEST.fit_predict(temp_X_22)
        vor_22 = Voronoi(temp_X_22)
        etykiety_22_BEST  = kluster_KMeans_22_BEST.labels_
        voronoi(vor_22,etykiety_22_BEST)

        kluster_KMeans_22_WORST = KMeans(n_clusters=3)
        kluster_KMeans_22_WORST.fit_predict(temp_X_22)
        vor_22 = Voronoi(temp_X_22)
        etykiety_22_WORST = kluster_KMeans_22_WORST.labels_
        voronoi(vor_22,etykiety_22_WORST)

        kluster_KMeans_23_BEST = KMeans(n_clusters=4)
        kluster_KMeans_23_BEST.fit_predict(temp_X_23)
        vor_23 = Voronoi(temp_X_23)
        etykiety_23_BEST  = kluster_KMeans_23_BEST.labels_
        voronoi(vor_23,etykiety_23_BEST)

        kluster_KMeans_23_WORST = KMeans(n_clusters=9)
        kluster_KMeans_23_WORST.fit_predict(temp_X_23)
        vor_23 = Voronoi(temp_X_23)
        etykiety_23_WORST = kluster_KMeans_23_WORST.labels_
        voronoi(vor_23,etykiety_23_WORST)
    elif choice is 2:
        eps_DBSCAN_1_BEST = DBSCAN(eps=1.3)
        eps_DBSCAN_1_BEST.fit_predict(temp_X_1)
        vor_1 = Voronoi(temp_X_1)
        etykiety_1_BEST  = eps_DBSCAN_1_BEST.labels_
        voronoi(vor_1,etykiety_1_BEST)

        eps_DBSCAN_1_WORST = DBSCAN(eps=0.17)
        eps_DBSCAN_1_WORST.fit_predict(temp_X_1)
        vor_1 = Voronoi(temp_X_1)
        etykiety_1_WORST = eps_DBSCAN_1_WORST.labels_
        voronoi(vor_1,etykiety_1_WORST)

        eps_DBSCAN_2_BEST = DBSCAN(eps=0.5)
        eps_DBSCAN_2_BEST.fit_predict(temp_X_2)
        vor_2 = Voronoi(temp_X_2)
        etykiety_2_BEST  = eps_DBSCAN_2_BEST.labels_
        voronoi(vor_2,etykiety_2_BEST)
        
        eps_DBSCAN_2_WORST = DBSCAN(eps=0.12)
        eps_DBSCAN_2_WORST.fit_predict(temp_X_2)
        vor_2 = Voronoi(temp_X_2)
        etykiety_2_WORST = eps_DBSCAN_2_WORST.labels_
        voronoi(vor_2,etykiety_2_WORST)

        eps_DBSCAN_3_BEST = DBSCAN(eps=0.5)
        eps_DBSCAN_3_BEST.fit_predict(temp_X_3)
        vor_3 = Voronoi(temp_X_3)
        etykiety_3_BEST  = eps_DBSCAN_3_BEST.labels_
        voronoi(vor_3,etykiety_3_BEST)
        
        eps_DBSCAN_3_WORST = DBSCAN(eps=0.1)
        eps_DBSCAN_3_WORST.fit_predict(temp_X_3)
        vor_3 = Voronoi(temp_X_3)
        etykiety_3_WORST = eps_DBSCAN_3_WORST.labels_
        voronoi(vor_3,etykiety_3_WORST)





        eps_DBSCAN_21_BEST = DBSCAN(eps=0.25)
        eps_DBSCAN_21_BEST.fit_predict(temp_X_21)
        vor_21 = Voronoi(temp_X_21)
        etykiety_21_BEST  = eps_DBSCAN_21_BEST.labels_
        voronoi(vor_21,etykiety_21_BEST)

        eps_DBSCAN_21_WORST = DBSCAN(eps=0.1)
        eps_DBSCAN_21_WORST.fit_predict(temp_X_21)
        vor_21 = Voronoi(temp_X_21)
        etykiety_21_WORST = eps_DBSCAN_21_WORST.labels_
        voronoi(vor_21,etykiety_21_WORST)

        eps_DBSCAN_22_BEST = DBSCAN(eps=0.4)
        eps_DBSCAN_22_BEST.fit_predict(temp_X_22)
        vor_22 = Voronoi(temp_X_22)
        etykiety_22_BEST  = eps_DBSCAN_22_BEST.labels_
        voronoi(vor_22,etykiety_22_BEST)

        eps_DBSCAN_22_WORST = DBSCAN(eps=0.1)
        eps_DBSCAN_22_WORST.fit_predict(temp_X_22)
        vor_22 = Voronoi(temp_X_22)
        etykiety_22_WORST = eps_DBSCAN_22_WORST.labels_
        voronoi(vor_22,etykiety_22_WORST)

        eps_DBSCAN_23_BEST = DBSCAN(eps=0.2)
        eps_DBSCAN_23_BEST.fit_predict(temp_X_23)
        vor_23 = Voronoi(temp_X_23)
        etykiety_23_BEST  = eps_DBSCAN_23_BEST.labels_
        voronoi(vor_23,etykiety_23_BEST)

        eps_DBSCAN_23_WORST = DBSCAN(eps=0.1)
        eps_DBSCAN_23_WORST.fit_predict(temp_X_23)
        vor_23 = Voronoi(temp_X_23)
        etykiety_23_WORST = eps_DBSCAN_23_WORST.labels_
        voronoi(vor_23,etykiety_23_WORST)
    else:
        best = [1.3, 0.5, 0.5, 0.25, 0.4, 0.2]
        worst = [0.17, 0.12, 0.1, 0.1, 0.1, 0.1]
        fig, axs = plt.subplots(3, 4)
        for i in range(6):
            temp = DBSCAN(eps=best[i], min_samples=10)
            temp.fit_predict(Data[i, :, 0:2])
            vor = Voronoi(Data[i,:,0:2])
            vor_fig = voronoi_fig(vor,temp.labels_)
            vor_fig.savefig(f'Voronoi_FIG_BEST_{1 if i < 3 else 2}_{(i)%3+1}.png')
            vor_img = plt.imread(f'Voronoi_FIG_BEST_{1 if i < 3 else 2}_{(i)%3+1}.png')
            axs[i%3,int(i/3)*2].imshow(vor_img)
            axs[i%3,int(i/3)*2].set_title("best from set "+str(int(i/3)+1)+"_"+str(i%3+1))

           
    plt.show()