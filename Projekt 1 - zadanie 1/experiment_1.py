import numpy as np
import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from warmup import plot_voronoi_diagram
from var import labels, points

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
    #----------------------  CZĘŚĆ CSV  ---------------------------
    class best_silhouette_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index:     int     = vindex
            self.best_n_cluster: int     = n_cluster
            self.value:          float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index = vindex
            self.best_n_cluster = n_cluster
            self.value = val

    class worst_silhouette_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index:        int     = vindex
            self.worst_n_cluster:    int     = n_cluster
            self.value:              float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index = vindex
            self.worst_n_cluster = n_cluster
            self.value = val
    
    list_best_silhouette_score  = []
    list_worst_silhouette_score = []
    for index in range(6):
        list_worst_silhouette_score.append(worst_silhouette_score(index,0,1.0))
        list_best_silhouette_score.append(best_silhouette_score(index,0,-1.0))

    fig, axs = plt.subplots(6,3)  
    
    y_pred : list[list[list[int]]] = [[[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[]]]
    
    for index in range(6):
        for n_clusters in range(2,10):
            #K-Means cluster
            klaster_KMeans: cluster.KMeans = cluster.KMeans(n_clusters=n_clusters)
            klaster_KMeans.fit(points[index])
            y_pred[index][n_clusters-2] = klaster_KMeans.labels_.astype(int)
            
            #Silhouette Score
            sil_score_kmeans : float = silhouette_score(points[index], np.ravel(y_pred[index][n_clusters-2]))

            if sil_score_kmeans > list_best_silhouette_score[index].value:
                list_best_silhouette_score[index].setVal(val=sil_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} BEST: INDEX {list_best_silhouette_score[index].best_index} SIL {list_best_silhouette_score[index].value} NCLUST {list_best_silhouette_score[index].best_n_cluster}')
            if sil_score_kmeans < list_worst_silhouette_score[index].value:
                list_worst_silhouette_score[index].setVal(val=sil_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} WORST: INDEX {list_worst_silhouette_score[index].worst_index} SIL {list_worst_silhouette_score[index].value} NCLUST {list_worst_silhouette_score[index].worst_n_cluster}')

            axs[index][0].plot(n_clusters, sil_score_kmeans, 'bo', label=str(y_pred[index][n_clusters-2]))
            axs[index][0].set_title(f'CSV: {1 if index < 3 else 2}_{(index)%3+1}')
            axs[index][0].set_xlabel("n-clusters")
            axs[index][0].set_ylabel("Silhouette Score")

    #Plotowanie najlepszego i najgorszego wyniku Silhouette dla każdego CSV
    for index in range(6):
        vor_ax_best = plot_voronoi_diagram(X=points[list_best_silhouette_score[index].best_index], y_true=None, y_pred=y_pred[index][list_best_silhouette_score[index].best_n_cluster-2])
        vor_ax_best.savefig(f'experiment_1_k_means_vor_ax_best_{1 if index < 3 else 2}_{(index)%3+1}.png')
        vor_image_best = plt.imread(f'experiment_1_k_means_vor_ax_best_{1 if index < 3 else 2}_{(index)%3+1}.png')
        axs[index][1].imshow(vor_image_best)
            
        vor_ax_worst = plot_voronoi_diagram(X=points[list_worst_silhouette_score[index].worst_index], y_true = None, y_pred=y_pred[index][list_worst_silhouette_score[index].worst_n_cluster-2])
        vor_ax_worst.savefig(f'experiment_1_k_means_vor_ax_worst_{1 if index < 3 else 2}_{(index)%3+1}.png')
        vor_image_worst = plt.imread(f'experiment_1_k_means_vor_ax_worst_{1 if index < 3 else 2}_{(index)%3+1}.png')
        axs[index][2].imshow(vor_image_worst)
        print('----------------')
        print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} BEST: INDEX {list_best_silhouette_score[index].best_index} SIL {list_best_silhouette_score[index].value} NCLUST {list_best_silhouette_score[index].best_n_cluster} WORST: INDEX {list_worst_silhouette_score[index].worst_index} SIL {list_worst_silhouette_score[index].value} NCLUST {list_worst_silhouette_score[index].worst_n_cluster}')

        
    #Tytuły
    axs[0][2].set_title(f'WORST SILHOUETTE CASE')
    axs[0][1].set_title(f'BEST SILHOUETTE CASE')
    
    fig.savefig('experiment_1_K_Means_Silhouette_Voronoi.png')
    plt.subplots_adjust(hspace=1.1,wspace=0.5)
    plt.show()

def experiment_1_DBSCAN() -> None:
    #----------------------  CZĘŚĆ CSV  ---------------------------
    class best_silhouette_score(object):
        def __init__(self, vindex: int, eps: float, val: float) -> None:
            self.best_index:   int     = vindex
            self.best_eps:     float   = eps
            self.value:        float   = val

        def setVal(self, vindex: int, eps: float, val: float) -> None:
            self.best_index = vindex
            self.best_eps = eps
            self.value = val

    class worst_silhouette_score(object):
        def __init__(self, vindex: int, eps: float, val: float) -> None:
            self.worst_index:  int     = vindex
            self.worst_eps:    float   = eps
            self.value:        float   = val

        def setVal(self, vindex: int, eps: int, val: float) -> None:
            self.worst_index = vindex
            self.worst_eps = eps
            self.value = val
    
    list_best_silhouette_score  = []
    list_worst_silhouette_score = []
    #list_eps : list[float] = [0.25,0.5,0.75,1.0,1.25,1.5]
    list_eps : list[float] = [0.05,0.1,0.15,0.2,0.25,0.5,0.75,1.0,1.25,1.5]
    #list_eps : list[float] = [0.05,0.1,0.15,0.2,0.25]

    for index in range(6):
        list_worst_silhouette_score.append(worst_silhouette_score(index,0,1.0))
        list_best_silhouette_score.append(best_silhouette_score(index,0,-1.0))

    fig, axs = plt.subplots(6,3)
    
    
    y_pred : list[list[list[int]]] = [[[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[]]]
    
    for index in range(6):
        for iter_eps in range(len(list_eps)):
            #K-Means cluster
            klaster_DBSCAN: cluster.DBSCAN = cluster.DBSCAN(eps=list_eps[iter_eps],min_samples=10)
            klaster_DBSCAN.fit(points[index])
            y_pred[index][iter_eps] = klaster_DBSCAN.labels_.astype(int)
            
            #Silhouette Score
            if len(set(np.ravel(y_pred[index][iter_eps]))) is not 1: #Sprawdzenie czy ilosc labels jest wieksza od 1, set eliminuje duplikaty
                sil_score_kmeans : float = silhouette_score(points[index], np.ravel(y_pred[index][iter_eps]))
                if sil_score_kmeans > list_best_silhouette_score[index].value:
                    list_best_silhouette_score[index].setVal(val=sil_score_kmeans, vindex=index, eps=iter_eps)
                    print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} BEST: INDEX {list_best_silhouette_score[index].best_index} SIL {list_best_silhouette_score[index].value} EPS {list_eps[list_best_silhouette_score[index].best_eps]}')
                if sil_score_kmeans < list_worst_silhouette_score[index].value:
                    list_worst_silhouette_score[index].setVal(val=sil_score_kmeans, vindex=index, eps=iter_eps)
                    print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} WORST: INDEX {list_worst_silhouette_score[index].worst_index} SIL {list_worst_silhouette_score[index].value} EPS {list_eps[list_worst_silhouette_score[index].worst_eps]}')

                axs[index][0].plot(list_eps[iter_eps], sil_score_kmeans, 'bo', label=str(y_pred[index][iter_eps]))
                axs[index][0].set_title(f'CSV: {1 if index < 3 else 2}_{(index)%3+1}')
                axs[index][0].set_xlabel("eps")
                axs[index][0].set_ylabel("Silhouette Score")

    #Plotowanie najlepszego i najgorszego wyniku Silhouette dla każdego CSV
    for index in range(6):
        vor_ax_best = plot_voronoi_diagram(X=points[list_best_silhouette_score[index].best_index], y_true=None, y_pred=y_pred[index][list_best_silhouette_score[index].best_eps])
        vor_ax_best.savefig(f'experiment_1_DBSCAN_vor_ax_best_{1 if index < 3 else 2}_{(index)%3+1}.png')
        vor_image_best = plt.imread(f'experiment_1_DBSCAN_vor_ax_best_{1 if index < 3 else 2}_{(index)%3+1}.png')
        axs[index][1].imshow(vor_image_best)
            
        vor_ax_worst = plot_voronoi_diagram(X=points[list_worst_silhouette_score[index].worst_index], y_true = None, y_pred=y_pred[index][list_worst_silhouette_score[index].worst_eps])
        vor_ax_worst.savefig(f'experiment_1_DBSCAN_vor_ax_worst_{1 if index < 3 else 2}_{(index)%3+1}.png')
        vor_image_worst = plt.imread(f'experiment_1_DBSCAN_vor_ax_worst_{1 if index < 3 else 2}_{(index)%3+1}.png')
        axs[index][2].imshow(vor_image_worst)
        
    #Tytuły
    axs[0][2].set_title(f'WORST SILHOUETTE CASE')
    axs[0][1].set_title(f'BEST SILHOUETTE CASE')
    
    fig.savefig('experiment_1_DBSCAN_Silhouette_Voronoi.png')
    plt.subplots_adjust(hspace=1.1,wspace=0.5)
    plt.show()