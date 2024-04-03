"""
Wyniki drugiego eksperymentu dla sześciu sztucznie wygenerowanych zbiorów danych i metody K-Means. 
Dla każdego zbioru należy pokazać wykres obrazujący zmianę wartości miar adjusted rand score, homogeneity score,  
completeness score oraz V-measure score przy zmieniającym się parametrze n_clusters oraz 
wizualizację klastrów (diagram Woronoja z pokazanymi prawdziwymi etykietami obiektów) 
dla najlepszego i najgorszego przypadku (wskazując, który to był przypadek i dlaczego).

"""
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.metrics import rand_score, homogeneity_score, completeness_score, v_measure_score
from warmup import plot_voronoi_diagram
from var import labels, points

def experiment_2_KMeans() -> None:
    #----------------------  CZĘŚĆ CSV  ---------------------------
    class best_rand_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index:     int     = vindex
            self.best_n_cluster: int     = n_cluster
            self.value:          float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index = vindex
            self.best_n_cluster = n_cluster
            self.value = val

    class worst_rand_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index:        int     = vindex
            self.worst_n_cluster:    int     = n_cluster
            self.value:              float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index = vindex
            self.worst_n_cluster = n_cluster
            self.value = val
    
    class best_homogenity_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index:     int     = vindex
            self.best_n_cluster: int     = n_cluster
            self.value:          float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index = vindex
            self.best_n_cluster = n_cluster
            self.value = val

    class worst_homogenity_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index:        int     = vindex
            self.worst_n_cluster:    int     = n_cluster
            self.value:              float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index = vindex
            self.worst_n_cluster = n_cluster
            self.value = val

    class best_completness_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index:     int     = vindex
            self.best_n_cluster: int     = n_cluster
            self.value:          float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index = vindex
            self.best_n_cluster = n_cluster
            self.value = val

    class worst_completness_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index:        int     = vindex
            self.worst_n_cluster:    int     = n_cluster
            self.value:              float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index = vindex
            self.worst_n_cluster = n_cluster
            self.value = val

    class best_v_measure_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index:     int     = vindex
            self.best_n_cluster: int     = n_cluster
            self.value:          float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index = vindex
            self.best_n_cluster = n_cluster
            self.value = val

    class worst_v_measure_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index:        int     = vindex
            self.worst_n_cluster:    int     = n_cluster
            self.value:              float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index = vindex
            self.worst_n_cluster = n_cluster
            self.value = val



    list_best_rand_score_score  = []
    list_worst_rand_score_score = []
    list_best_homogenity_score  = []
    list_worst_homogenity_score = []
    list_best_completness_score  = []
    list_worst_completness_score = []
    list_best_v_measure_score  = []
    list_worst_v_measure_score = []
    for index in range(6):
        list_best_rand_score_score.append(best_rand_score(index,0,1.0))
        list_worst_rand_score_score.append(worst_rand_score(index,0,1.0))
        list_best_homogenity_score.append(best_homogenity_score(index,0,1.0))
        list_worst_homogenity_score.append(worst_homogenity_score(index,0,1.0))
        list_best_completness_score.append(best_completness_score(index,0,1.0))
        list_worst_completness_score.append(worst_completness_score(index,0,1.0))
        list_best_v_measure_score.append(best_v_measure_score(index,0,1.0))
        list_worst_v_measure_score.append(worst_v_measure_score(index,0,1.0))

    fig, axs = plt.subplots(6,1)  
    fig_vor, ax_vor = plt.subplots(6,2)
    
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
            
            #Rand Score
            rand_score_kmeans : float = rand_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
            if rand_score_kmeans > list_best_rand_score_score[index].value:
                list_best_rand_score_score[index].setVal(val=rand_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} BEST: INDEX {list_best_rand_score_score[index].best_index} SIL {list_best_rand_score_score[index].value} NCLUST {list_best_rand_score_score[index].best_n_cluster}')
            if rand_score_kmeans < list_worst_rand_score_score[index].value:
                list_worst_rand_score_score[index].setVal(val=rand_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} WORST: INDEX {list_worst_rand_score_score[index].worst_index} SIL {list_worst_rand_score_score[index].value} NCLUST {list_worst_rand_score_score[index].worst_n_cluster}')

            axs[index].plot(n_clusters, rand_score_kmeans, 'o', color='yellow', linestyle='solid', linewidth=5, label="Rand Score")
            
            #Homogenity Score
            homogenity_score_kmeans : float = homogeneity_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
            if homogenity_score_kmeans > list_best_rand_score_score[index].value:
                list_best_homogenity_score[index].setVal(val=homogenity_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} BEST: INDEX {list_best_homogenity_score[index].best_index} SIL {list_best_homogenity_score[index].value} NCLUST {list_best_homogenity_score[index].best_n_cluster}')
            if homogenity_score_kmeans < list_worst_homogenity_score[index].value:
                list_worst_homogenity_score[index].setVal(val=homogenity_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} WORST: INDEX {list_worst_homogenity_score[index].worst_index} SIL {list_worst_homogenity_score[index].value} NCLUST {list_worst_homogenity_score[index].worst_n_cluster}')

            axs[index].plot(n_clusters, homogenity_score_kmeans, 'o', color='green', linestyle='solid', linewidth=5, label="Homogeneity Score")
            #Completness Score
            completeness_score_kmeans : float = completeness_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
            if completeness_score_kmeans > list_best_rand_score_score[index].value:
                list_best_completness_score[index].setVal(val=completeness_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} BEST: INDEX {list_best_completness_score[index].best_index} SIL {list_best_completness_score[index].value} NCLUST {list_best_completness_score[index].best_n_cluster}')
            if completeness_score_kmeans < list_worst_completness_score[index].value:
                list_worst_completness_score[index].setVal(val=completeness_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} WORST: INDEX {list_worst_completness_score[index].worst_index} SIL {list_worst_completness_score[index].value} NCLUST {list_worst_completness_score[index].worst_n_cluster}')

            axs[index].plot(n_clusters, completeness_score_kmeans, 'o', color='blue', linestyle='solid', linewidth=5, label="Completness Score")
            #V-Measure Score
            v_mneasure_score_kmeans : float = v_measure_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
            if v_mneasure_score_kmeans > list_best_v_measure_score[index].value:
                list_best_completness_score[index].setVal(val=v_mneasure_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} BEST: INDEX {list_best_v_measure_score[index].best_index} SIL {list_best_v_measure_score[index].value} NCLUST {list_best_v_measure_score[index].best_n_cluster}')
            if v_mneasure_score_kmeans < list_worst_completness_score[index].value:
                list_worst_v_measure_score[index].setVal(val=v_mneasure_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} WORST: INDEX {list_worst_v_measure_score[index].worst_index} SIL {list_worst_v_measure_score[index].value} NCLUST {list_worst_v_measure_score[index].worst_n_cluster}')

            axs[index].plot(n_clusters, v_mneasure_score_kmeans, 'o', color='red', linestyle='solid', linewidth=5, label="V-Measure Score")


            axs[index].set_title(f'CSV: {1 if index < 3 else 2}_{(index)%3+1}')
            axs[index].set_xlabel("n-clusters")
            
            axs[index].set_ylabel("Score")
    """
    #Plotowanie najlepszego i najgorszego wyniku Silhouette dla każdego CSV
    for index in range(6):
        vor_ax_best = plot_voronoi_diagram(X=points[list_best_rand_score_score[index].best_index], y_true=None, y_pred=y_pred[index][list_best_rand_score_score[index].best_n_cluster-2])
        vor_ax_best.savefig(f'experiment_2_k_means_vor_RAND_SCORE_best_{1 if index < 3 else 2}_{(index)%3+1}.png')
        vor_image_best = plt.imread(f'experiment_2_k_means_vor_RAND_SCORE_best_{1 if index < 3 else 2}_{(index)%3+1}.png')
        ax_vor[index][1].imshow(vor_image_best)
            
        vor_ax_worst = plot_voronoi_diagram(X=points[list_worst_rand_score_score[index].worst_index], y_true = None, y_pred=y_pred[index][list_worst_rand_score_score[index].worst_n_cluster-2])
        vor_ax_worst.savefig(f'k_means_vor_ax_worst_{1 if index < 3 else 2}_{(index)%3+1}.png')
        vor_image_worst = plt.imread(f'k_means_vor_ax_worst_{1 if index < 3 else 2}_{(index)%3+1}.png')
        ax_vor[index][2].imshow(vor_image_worst)

        print('----------------')
        print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} BEST: INDEX {list_best_silhouette_score[index].best_index} SIL {list_best_silhouette_score[index].value} NCLUST {list_best_silhouette_score[index].best_n_cluster} WORST: INDEX {list_worst_silhouette_score[index].worst_index} SIL {list_worst_silhouette_score[index].value} NCLUST {list_worst_silhouette_score[index].worst_n_cluster}')
        #Tytuły
        ax_vor[0][2].set_title(f'WORST CASE')
        ax_vor[0][1].set_title(f'BEST CASE')
    """        
    
    
    #fig.savefig('experiment_2_K_Means_Silhouette_Voronoi.png')
    axs[5].legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(hspace=0.6,wspace=0.5)
    plt.show()



def experiment_2_DBSCAN() -> None:
    #----------------------  CZĘŚĆ CSV  ---------------------------
    class best_rand_score(object):
        def __init__(self, vindex: int, eps: int, val: float) -> None:
            self.best_index:     int     = vindex
            self.best_eps:       int     = eps
            self.value:          float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index = vindex
            self.best_eps = n_cluster
            self.value = val

    class worst_rand_score(object):
        def __init__(self, vindex: int, eps: int, val: float) -> None:
            self.worst_index:        int     = vindex
            self.best_eps:           int     = eps
            self.value:              float   = val

        def setVal(self, vindex: int, eps: int, val: float) -> None:
            self.worst_index = vindex
            self.best_eps = eps
            self.value = val
    
    class best_homogenity_score(object):
        def __init__(self, vindex: int, eps: int, val: float) -> None:
            self.best_index:     int     = vindex
            self.best_eps:       int     = eps
            self.value:          float   = val

        def setVal(self, vindex: int, eps: int, val: float) -> None:
            self.best_index = vindex
            self.best_eps = eps
            self.value = val

    class worst_homogenity_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index:        int     = vindex
            self.worst_n_cluster:    int     = n_cluster
            self.value:              float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index = vindex
            self.worst_n_cluster = n_cluster
            self.value = val

    class best_completness_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index:     int     = vindex
            self.best_n_cluster: int     = n_cluster
            self.value:          float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index = vindex
            self.best_n_cluster = n_cluster
            self.value = val

    class worst_completness_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index:        int     = vindex
            self.worst_n_cluster:    int     = n_cluster
            self.value:              float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index = vindex
            self.worst_n_cluster = n_cluster
            self.value = val

    class best_v_measure_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index:     int     = vindex
            self.best_n_cluster: int     = n_cluster
            self.value:          float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index = vindex
            self.best_n_cluster = n_cluster
            self.value = val

    class worst_v_measure_score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index:        int     = vindex
            self.worst_n_cluster:    int     = n_cluster
            self.value:              float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.worst_index = vindex
            self.worst_n_cluster = n_cluster
            self.value = val



    list_best_rand_score_score  = []
    list_worst_rand_score_score = []
    list_best_homogenity_score  = []
    list_worst_homogenity_score = []
    list_best_completness_score  = []
    list_worst_completness_score = []
    list_best_v_measure_score  = []
    list_worst_v_measure_score = []
    for index in range(6):
        list_best_rand_score_score.append(best_rand_score(index,0,1.0))
        list_worst_rand_score_score.append(worst_rand_score(index,0,1.0))
        list_best_homogenity_score.append(best_homogenity_score(index,0,1.0))
        list_worst_homogenity_score.append(worst_homogenity_score(index,0,1.0))
        list_best_completness_score.append(best_completness_score(index,0,1.0))
        list_worst_completness_score.append(worst_completness_score(index,0,1.0))
        list_best_v_measure_score.append(best_v_measure_score(index,0,1.0))
        list_worst_v_measure_score.append(worst_v_measure_score(index,0,1.0))

    fig, axs = plt.subplots(6,1)  
    fig_vor, ax_vor = plt.subplots(6,2)
    
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
            
            #Rand Score
            rand_score_kmeans : float = rand_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
            if rand_score_kmeans > list_best_rand_score_score[index].value:
                list_best_rand_score_score[index].setVal(val=rand_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} BEST: INDEX {list_best_rand_score_score[index].best_index} SIL {list_best_rand_score_score[index].value} NCLUST {list_best_rand_score_score[index].best_n_cluster}')
            if rand_score_kmeans < list_worst_rand_score_score[index].value:
                list_worst_rand_score_score[index].setVal(val=rand_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} WORST: INDEX {list_worst_rand_score_score[index].worst_index} SIL {list_worst_rand_score_score[index].value} NCLUST {list_worst_rand_score_score[index].worst_n_cluster}')

            axs[index].plot(n_clusters, rand_score_kmeans, 'o', color='yellow', linestyle='solid', linewidth=5, label="Rand Score")
            
            #Homogenity Score
            homogenity_score_kmeans : float = homogeneity_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
            if homogenity_score_kmeans > list_best_rand_score_score[index].value:
                list_best_homogenity_score[index].setVal(val=homogenity_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} BEST: INDEX {list_best_homogenity_score[index].best_index} SIL {list_best_homogenity_score[index].value} NCLUST {list_best_homogenity_score[index].best_n_cluster}')
            if homogenity_score_kmeans < list_worst_homogenity_score[index].value:
                list_worst_homogenity_score[index].setVal(val=homogenity_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} WORST: INDEX {list_worst_homogenity_score[index].worst_index} SIL {list_worst_homogenity_score[index].value} NCLUST {list_worst_homogenity_score[index].worst_n_cluster}')

            axs[index].plot(n_clusters, homogenity_score_kmeans, 'o', color='green', linestyle='solid', linewidth=5, label="Homogeneity Score")
            #Completness Score
            completeness_score_kmeans : float = completeness_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
            if completeness_score_kmeans > list_best_rand_score_score[index].value:
                list_best_completness_score[index].setVal(val=completeness_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} BEST: INDEX {list_best_completness_score[index].best_index} SIL {list_best_completness_score[index].value} NCLUST {list_best_completness_score[index].best_n_cluster}')
            if completeness_score_kmeans < list_worst_completness_score[index].value:
                list_worst_completness_score[index].setVal(val=completeness_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} WORST: INDEX {list_worst_completness_score[index].worst_index} SIL {list_worst_completness_score[index].value} NCLUST {list_worst_completness_score[index].worst_n_cluster}')

            axs[index].plot(n_clusters, completeness_score_kmeans, 'o', color='blue', linestyle='solid', linewidth=5, label="Completness Score")
            #V-Measure Score
            v_mneasure_score_kmeans : float = v_measure_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
            if v_mneasure_score_kmeans > list_best_v_measure_score[index].value:
                list_best_completness_score[index].setVal(val=v_mneasure_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} BEST: INDEX {list_best_v_measure_score[index].best_index} SIL {list_best_v_measure_score[index].value} NCLUST {list_best_v_measure_score[index].best_n_cluster}')
            if v_mneasure_score_kmeans < list_worst_completness_score[index].value:
                list_worst_v_measure_score[index].setVal(val=v_mneasure_score_kmeans, vindex=index, n_cluster=n_clusters)
                print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} WORST: INDEX {list_worst_v_measure_score[index].worst_index} SIL {list_worst_v_measure_score[index].value} NCLUST {list_worst_v_measure_score[index].worst_n_cluster}')

            axs[index].plot(n_clusters, v_mneasure_score_kmeans, 'o', color='red', linestyle='solid', linewidth=5, label="V-Measure Score")


            axs[index].set_title(f'CSV: {1 if index < 3 else 2}_{(index)%3+1}')
            axs[index].set_xlabel("n-clusters")
            
            axs[index].set_ylabel("Score")
    """
    #Plotowanie najlepszego i najgorszego wyniku Silhouette dla każdego CSV
    for index in range(6):
        vor_ax_best = plot_voronoi_diagram(X=points[list_best_rand_score_score[index].best_index], y_true=None, y_pred=y_pred[index][list_best_rand_score_score[index].best_n_cluster-2])
        vor_ax_best.savefig(f'experiment_2_k_means_vor_RAND_SCORE_best_{1 if index < 3 else 2}_{(index)%3+1}.png')
        vor_image_best = plt.imread(f'experiment_2_k_means_vor_RAND_SCORE_best_{1 if index < 3 else 2}_{(index)%3+1}.png')
        ax_vor[index][1].imshow(vor_image_best)
            
        vor_ax_worst = plot_voronoi_diagram(X=points[list_worst_rand_score_score[index].worst_index], y_true = None, y_pred=y_pred[index][list_worst_rand_score_score[index].worst_n_cluster-2])
        vor_ax_worst.savefig(f'k_means_vor_ax_worst_{1 if index < 3 else 2}_{(index)%3+1}.png')
        vor_image_worst = plt.imread(f'k_means_vor_ax_worst_{1 if index < 3 else 2}_{(index)%3+1}.png')
        ax_vor[index][2].imshow(vor_image_worst)

        print('----------------')
        print(f'CSV: {1 if index < 3 else 2}_{(index)%3+1} BEST: INDEX {list_best_silhouette_score[index].best_index} SIL {list_best_silhouette_score[index].value} NCLUST {list_best_silhouette_score[index].best_n_cluster} WORST: INDEX {list_worst_silhouette_score[index].worst_index} SIL {list_worst_silhouette_score[index].value} NCLUST {list_worst_silhouette_score[index].worst_n_cluster}')
        #Tytuły
        ax_vor[0][2].set_title(f'WORST CASE')
        ax_vor[0][1].set_title(f'BEST CASE')
    """        
    
    
    #fig.savefig('experiment_2_K_Means_Silhouette_Voronoi.png')
    axs[5].legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(hspace=0.6,wspace=0.5)
    plt.show()