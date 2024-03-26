import numpy as np
import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from warmup import plot_voronoi_diagram
from main import labels, points

"""
Za pomocą technik klasteryzacji można sprawdzić czy obiekty w zbiorze danych są rozłożone w przestrzeni równomiernie, czy też występują w nich rozdzielone skupiska obiektów podobnych. 
Skupiska takie mogą świadczyć o tym, że proces generujący te dane nie jest zupełnie losowy, a idąc dalej można zastanawiać się co takie skupiska reprezentują. 
Oczywiście w przypadku jedno-, dwu- lub trójwymiarowym identyfikację takich skupisk można przeprowadzić wizualnie. W przypadku n>3 ocena wizualna nie jest możliwa. 
Co więcej nie można również wizualnie ocenić, które metody klasteryzacji i które parametry tych metod, najlepiej potrafią zidentyfikować klastry. 
W takiej sytuacji należy skorzystać z miar, które ilościowo oceniają jakość takich klastrów. Jedną z nich jest silhouette score.
"""

"""
--- metoda K-means - W podejściu tym klaster reprezentowany jest przez średnią wartość swoich elementów. Skupiska znajdowane w ten sposób mają zwykle kształt kulisty. 
Liczbę poszukiwanych skupisk ustala się tu odgórnie za pomocą parametru n_clusters. Reszta parametrów może mieć domyślne wartości.

--- metoda DBSCAN - Należy ona do grupy metod gęstościowych. Metody tego typu skupiają się na znajdowaniu zagęszczeń punktów w przestrzeni i łączeniu obszarów o podobnej gęstości. 
Liczba znalezionych w ten sposób skupisk zależy od struktury analizowanego zbioru, a stopień oczekiwanego zagęszczenia należy kontrolować za pomocą parametru eps. 
Należy zwrócić uwagę, że metoda ta umożliwia również wykrywanie wyjątków (ang. outliers), jednak w niniejszym zadaniu nie należy z tego korzystać przyjmując wartość parametru min_samples równą 1. 
Reszta parametrów może mieć domyślne wartości.
"""
def experiment_1_KMeans() -> None:
    for index in range(6):
        for n_clusters in range(1,9):
            #K-Means cluster
            klaster_KMeans: cluster.KMeans = cluster.KMeans(n_clusters=n_clusters)
            klaster_KMeans.fit(points[index])
            y_pred = klaster_KMeans.labels_.astype(int)
            plot_voronoi_diagram(points[index], labels[index], y_pred)

            #Silhouette Score
            sil_score_kmeans : float = silhouette_score(points[index], labels[index])
            plt.plot(n_clusters, sil_score_kmeans, 'ro', label=str(labels[index]))
            plt.title(f'CSV: {index}, {n_clusters}-CLUSTERS')
            plt.xlabel("n-clusters")
            plt.ylabel("Silhouette Score")

def experiment_1_DBSCAN() -> None:
    pass