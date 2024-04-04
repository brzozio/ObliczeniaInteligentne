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
from sklearn.metrics import rand_score, homogeneity_score, completeness_score, v_measure_score, silhouette_score
from warmup import plot_voronoi_diagram
from var import labels, points
from sklearn import datasets

def experiment_2_KMeans() -> None:
    #----------------------  CZĘŚĆ CSV  ---------------------------
    class score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index:     int     = vindex
            self.n_cluster: int     = n_cluster
            self.value:          float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index = vindex
            self.n_cluster = n_cluster
            self.value = val



    list_best_rand_score_score : list[score] = []
    list_worst_rand_score_score : list[score] = []
    list_best_homogenity_score : list[score] = []
    list_worst_homogenity_score : list[score] = []
    list_best_completness_score : list[score] = []
    list_worst_completness_score : list[score] = []
    list_best_v_measure_score : list[score] = []
    list_worst_v_measure_score : list[score] = []

    for index in range(6):
        list_best_rand_score_score.append(score(index,0,1.0))
        list_worst_rand_score_score.append(score(index,0,1.0))
        list_best_homogenity_score.append(score(index,0,1.0))
        list_worst_homogenity_score.append(score(index,0,1.0))
        list_best_completness_score.append(score(index,0,1.0))
        list_worst_completness_score.append(score(index,0,1.0))
        list_best_v_measure_score.append(score(index,0,1.0))
        list_worst_v_measure_score.append(score(index,0,1.0))

    fig, axs = plt.subplots(6,1)  
    fig_vor, ax_vor = plt.subplots(6,2)
    
    y_pred : list[list[list[int]]] = [[[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[]]]
    list_n_clusters : list[int] = [2,3,4,5,6,9,13,50]
    list_vision_best_clusters : list[int] = [2,2,9,2,4,4]
    list_vision_worst_clusters : list[int] = [9,9,5,9,9,6]
    
    for index in range(6):
        if index is 0 or index is 1 or index is 3 or index is 4:
            list_k_means_rand : list[float] = []
            list_k_means_homogenity : list[float] = []
            list_k_means_completness : list[float] = []
            list_k_means_v_measure_05 : list[float] = []
            list_k_means_v_measure_1 : list[float] = []
            list_k_means_v_measure_2 : list[float] = []
            for n_clusters in range(2,10):
                #K-Means cluster
                klaster_KMeans: cluster.KMeans = cluster.KMeans(n_clusters=n_clusters)
                klaster_KMeans.fit(points[index])
                y_pred[index][n_clusters-2] = klaster_KMeans.labels_.astype(int)
                
                #Rand Score
                rand_score_kmeans : float = rand_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
                list_k_means_rand.append(rand_score_kmeans)
            
                
                #Homogenity Score
                homogenity_score_kmeans : float = homogeneity_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
                list_k_means_homogenity.append(homogenity_score_kmeans)
            
                #Completness Score
                completeness_score_kmeans : float = completeness_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
                list_k_means_completness.append(completeness_score_kmeans)
            
                #V-Measure Score
                v_mneasure_score_kmeans : float = v_measure_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]), beta=0.5)
                list_k_means_v_measure_05.append(v_mneasure_score_kmeans)

                v_mneasure_score_kmeans : float = v_measure_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]), beta=1.5)
                list_k_means_v_measure_1.append(v_mneasure_score_kmeans)

                v_mneasure_score_kmeans : float = v_measure_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]), beta=2.0)
                list_k_means_v_measure_2.append(v_mneasure_score_kmeans)

            
            axs[index].plot(range(2,10), list_k_means_homogenity, 'o', color='green', linestyle='solid', linewidth=2, label="Homogeneity Score")
            axs[index].plot(range(2,10), list_k_means_v_measure_05, 'o', color='red', linestyle='--', linewidth=2, label="V-Measure Score beta=0.5")
            axs[index].plot(range(2,10), list_k_means_v_measure_1, 'o', color='purple', linestyle='--', linewidth=2, label="V-Measure Score beta=1.0")
            axs[index].plot(range(2,10), list_k_means_v_measure_2, 'o', color='gray', linestyle='--', linewidth=2, label="V-Measure Score beta=2.0")
            axs[index].plot(range(2,10), list_k_means_rand, 'o', color='yellow', linestyle='solid', linewidth=2, label="Rand Score")
            axs[index].plot(range(2,10), list_k_means_completness, 'o', color='blue', linestyle='solid', linewidth=2, label="Completness Score")


            axs[index].set_title(f'CSV: {1 if index < 3 else 2}_{(index)%3+1}')
            axs[index].set_xlabel("n-clusters")
            
            axs[index].set_ylabel("Score") 

        else:
            list_k_means_rand : list[float] = []
            list_k_means_homogenity : list[float] = []
            list_k_means_completness : list[float] = []
            list_k_means_v_measure_05 : list[float] = []
            list_k_means_v_measure_1 : list[float] = []
            list_k_means_v_measure_2 : list[float] = []
            for n_clusters in range(2,10):
                #K-Means cluster
                klaster_KMeans: cluster.KMeans = cluster.KMeans(n_clusters=list_n_clusters[n_clusters-2])
                klaster_KMeans.fit(points[index])
                y_pred[index][n_clusters-2] = klaster_KMeans.labels_.astype(int)
                
                #Rand Score
                rand_score_kmeans : float = rand_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
                list_k_means_rand.append(rand_score_kmeans)
            
                
                #Homogenity Score
                homogenity_score_kmeans : float = homogeneity_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
                list_k_means_homogenity.append(homogenity_score_kmeans)
            
                #Completness Score
                completeness_score_kmeans : float = completeness_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]))
                list_k_means_completness.append(completeness_score_kmeans)
            
                #V-Measure Score
                v_mneasure_score_kmeans : float = v_measure_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]), beta=0.5)
                list_k_means_v_measure_05.append(v_mneasure_score_kmeans)

                v_mneasure_score_kmeans : float = v_measure_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]), beta=1.5)
                list_k_means_v_measure_1.append(v_mneasure_score_kmeans)

                v_mneasure_score_kmeans : float = v_measure_score(np.ravel(labels[index]), np.ravel(y_pred[index][n_clusters-2]), beta=2.0)
                list_k_means_v_measure_2.append(v_mneasure_score_kmeans)
            
            axs[index].plot(list_n_clusters, list_k_means_homogenity, 'o', color='green', linestyle='solid', linewidth=2, label="Homogeneity Score")
            axs[index].plot(list_n_clusters, list_k_means_v_measure_05, 'o', color='red', linestyle='--', linewidth=2, label="V-Measure Score beta=0.5")
            axs[index].plot(list_n_clusters, list_k_means_v_measure_1, 'o', color='purple', linestyle='--', linewidth=2, label="V-Measure Score beta=1.0")
            axs[index].plot(list_n_clusters, list_k_means_v_measure_2, 'o', color='gray', linestyle='--', linewidth=2, label="V-Measure Score beta=2.0")
            axs[index].plot(list_n_clusters, list_k_means_rand, 'o', color='yellow', linestyle='solid', linewidth=2, label="Rand Score")
            axs[index].plot(list_n_clusters, list_k_means_completness, 'o', color='blue', linestyle='solid', linewidth=2, label="Completness Score")


            axs[index].set_title(f'CSV: {1 if index < 3 else 2}_{(index)%3+1}')
            axs[index].set_xlabel("n-clusters")
            
            axs[index].set_ylabel("Score") 
               


    #Plotowanie najlepszego i najgorszego wyniku Silhouette dla każdego CSV
    
        vor_ax_best = plot_voronoi_diagram(X=points[index], y_true=None, y_pred=y_pred[index][list_vision_best_clusters[index]-2])
        vor_ax_best.savefig(f'experiment_2_k_means_vor_best_{1 if index < 3 else 2}_{(index)%3+1}.png')
        vor_image_best = plt.imread(f'experiment_2_k_means_vor_best_{1 if index < 3 else 2}_{(index)%3+1}.png')
        ax_vor[index][0].imshow(vor_image_best)
            
        vor_ax_worst = plot_voronoi_diagram(X=points[index], y_true=None, y_pred=y_pred[index][list_vision_worst_clusters[index]-2])
        vor_ax_worst.savefig(f'experiment_2_k_means_vor_worst_{1 if index < 3 else 2}_{(index)%3+1}.png')
        vor_image_worst = plt.imread(f'experiment_2_k_means_vor_worst_{1 if index < 3 else 2}_{(index)%3+1}.png')
        ax_vor[index][1].imshow(vor_image_worst)

        ax_vor[0][1].set_title(f'WORST CASE')
        ax_vor[0][0].set_title(f'BEST CASE')
           
    
    
    #fig.savefig('experiment_2_K_Means_Silhouette_Voronoi.png')
    axs[5].legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(hspace=0.6,wspace=0.5)
    plt.show()



def experiment_2_DBSCAN() -> None:
    #----------------------  CZĘŚĆ CSV  ---------------------------
    class score(object):
        def __init__(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index:     int     = vindex
            self.n_cluster: int     = n_cluster
            self.value:          float   = val

        def setVal(self, vindex: int, n_cluster: int, val: float) -> None:
            self.best_index = vindex
            self.n_cluster = n_cluster
            self.value = val


    list_eps : list[float] = [0.1,0.15,0.2,0.25,0.3,0.5,0.75,1.0,1.25,1.5]
    list_best_rand_score_score : list[score] = []
    list_worst_rand_score_score : list[score] = []
    list_best_homogenity_score : list[score] = []
    list_worst_homogenity_score : list[score] = []
    list_best_completness_score : list[score] = []
    list_worst_completness_score : list[score] = []
    list_best_v_measure_score : list[score] = []
    list_worst_v_measure_score : list[score] = []

    for index in range(6):
        list_best_rand_score_score.append(score(index,0,1.0))
        list_worst_rand_score_score.append(score(index,0,1.0))
        list_best_homogenity_score.append(score(index,0,1.0))
        list_worst_homogenity_score.append(score(index,0,1.0))
        list_best_completness_score.append(score(index,0,1.0))
        list_worst_completness_score.append(score(index,0,1.0))
        list_best_v_measure_score.append(score(index,0,1.0))
        list_worst_v_measure_score.append(score(index,0,1.0))

    fig, axs = plt.subplots(6,1)  
    fig_vor, ax_vor = plt.subplots(6,2)
    
    y_pred : list[list[list[int]]] = [[[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[]]]
    
    list_vision_best_clusters : list[int] = [2,2,9,2,4,4]
    list_vision_worst_clusters : list[int] = [9,9,5,9,9,6]
    
    for index in range(6):
        if True:
            list_k_means_rand : list[float] = []
            list_k_means_homogenity : list[float] = []
            list_k_means_completness : list[float] = []
            list_k_means_v_measure_05 : list[float] = []
            list_k_means_v_measure_1 : list[float] = []
            list_k_means_v_measure_2 : list[float] = []
            for iter_eps in range(len(list_eps)):
                #K-Means cluster
                klaster_KMeans: cluster.DBSCAN = cluster.DBSCAN(eps=list_eps[iter_eps],min_samples=10)
                klaster_KMeans.fit(points[index])
                y_pred[index][iter_eps] = klaster_KMeans.labels_.astype(int)
                
                #Rand Score
                rand_score_kmeans : float = rand_score(np.ravel(labels[index]), np.ravel(y_pred[index][iter_eps]))
                list_k_means_rand.append(rand_score_kmeans)
            
                
                #Homogenity Score
                homogenity_score_kmeans : float = homogeneity_score(np.ravel(labels[index]), np.ravel(y_pred[index][iter_eps]))
                list_k_means_homogenity.append(homogenity_score_kmeans)
            
                #Completness Score
                completeness_score_kmeans : float = completeness_score(np.ravel(labels[index]), np.ravel(y_pred[index][iter_eps]))
                list_k_means_completness.append(completeness_score_kmeans)
            
                #V-Measure Score
                v_mneasure_score_kmeans : float = v_measure_score(np.ravel(labels[index]), np.ravel(y_pred[index][iter_eps]), beta=0.5)
                list_k_means_v_measure_05.append(v_mneasure_score_kmeans)

                v_mneasure_score_kmeans : float = v_measure_score(np.ravel(labels[index]), np.ravel(y_pred[index][iter_eps]), beta=1.5)
                list_k_means_v_measure_1.append(v_mneasure_score_kmeans)

                v_mneasure_score_kmeans : float = v_measure_score(np.ravel(labels[index]), np.ravel(y_pred[index][iter_eps]), beta=2.0)
                list_k_means_v_measure_2.append(v_mneasure_score_kmeans)

            
            axs[index].plot(list_eps, list_k_means_homogenity, 'o', color='green', linestyle='solid', linewidth=2, label="Homogeneity Score")
            axs[index].plot(list_eps, list_k_means_v_measure_05, 'o', color='red', linestyle='--', linewidth=2, label="V-Measure Score beta=0.5")
            axs[index].plot(list_eps, list_k_means_v_measure_1, 'o', color='purple', linestyle='--', linewidth=2, label="V-Measure Score beta=1.0")
            axs[index].plot(list_eps, list_k_means_v_measure_2, 'o', color='gray', linestyle='--', linewidth=2, label="V-Measure Score beta=2.0")
            axs[index].plot(list_eps, list_k_means_rand, 'o', color='yellow', linestyle='solid', linewidth=2, label="Rand Score")
            axs[index].plot(list_eps, list_k_means_completness, 'o', color='blue', linestyle='solid', linewidth=2, label="Completness Score")


            axs[index].set_title(f'CSV: {1 if index < 3 else 2}_{(index)%3+1}')
            axs[index].set_xlabel("eps")
            axs[index].set_ylabel("Score") 
            axs[index].set_ylim(0.0,1.05)

            #Etykiety ilosci klastrów na wykresie
            for eps_text in range(len(list_eps)):
                axs[index].text(list_eps[eps_text], 0.1, len(set(y_pred[index][eps_text])))

    #Plotowanie najlepszego i najgorszego wyniku Silhouette dla każdego CSV
    
        vor_ax_best = plot_voronoi_diagram(X=points[index], y_true=None, y_pred=y_pred[index][list_vision_best_clusters[index]-2])
        vor_ax_best.savefig(f'experiment_2_DBSCAN_vor_best_{1 if index < 3 else 2}_{(index)%3+1}.png')
        vor_image_best = plt.imread(f'experiment_2_DBSCAN_vor_best_{1 if index < 3 else 2}_{(index)%3+1}.png')
        ax_vor[index][0].imshow(vor_image_best)
            
        vor_ax_worst = plot_voronoi_diagram(X=points[index], y_true=None, y_pred=y_pred[index][list_vision_worst_clusters[index]-2])
        vor_ax_worst.savefig(f'experiment_2_DBSCAN_vor_worst_{1 if index < 3 else 2}_{(index)%3+1}.png')
        vor_image_worst = plt.imread(f'experiment_2_DBSCAN_vor_worst_{1 if index < 3 else 2}_{(index)%3+1}.png')
        ax_vor[index][1].imshow(vor_image_worst)

        ax_vor[0][1].set_title(f'WORST CASE')
        ax_vor[0][0].set_title(f'BEST CASE')
           
    axs[5].legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    

    plt.subplots_adjust(hspace=0.6,wspace=0.5)
    plt.show()



def experiment_2_KMeans_IRIS_AND_OTHERS() -> None:
    fig, axs = plt.subplots(3,1)  
    
    y_pred : list[list[list[int]]] = [[[],[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[],[]]]
    
    list_n_clusters : list[int]          = [2,3,4,5,10,12,13,14,20,29,30,31,50]
    iris          = datasets.load_iris(return_X_y=True)
    breast_cancer = datasets.load_breast_cancer(return_X_y=True)
    wine          = datasets.load_wine(return_X_y=True)
    #list_vision_best_clusters : list[int] = [2,2,9,2,4,4]
    #list_vision_worst_clusters : list[int] = [9,9,5,9,9,6]
    
    #============== IRIS ==============================================================
    list_k_means_rand : list[float] = []
    list_k_means_homogenity : list[float] = []
    list_k_means_completness : list[float] = []
    list_k_means_silhouette : list[float] = []
    for index in range(len(list_n_clusters)):
        #K-Means cluster
        klaster_KMeans: cluster.KMeans = cluster.KMeans(n_clusters=list_n_clusters[index])
        klaster_KMeans.fit(iris[0])
        y_pred[0][index] = klaster_KMeans.labels_.astype(int)

        print(f'IRIS {index} CLUSTERS: {len(set(y_pred[0][index]))}')
        
        #Rand Score
        rand_score_kmeans : float = rand_score(np.ravel(iris[1]), np.ravel(y_pred[0][index]))
        list_k_means_rand.append(rand_score_kmeans)

        
        #Homogenity Score
        homogenity_score_kmeans : float = homogeneity_score(np.ravel(iris[1]), np.ravel(y_pred[0][index]))
        list_k_means_homogenity.append(homogenity_score_kmeans)
    
        #Completness Score
        completeness_score_kmeans : float = completeness_score(np.ravel(iris[1]), np.ravel(y_pred[0][index]))
        list_k_means_completness.append(completeness_score_kmeans)

        #Silhouette Score
        #silhouette_score_kmeans : float = silhouette_score(iris[1], y_pred[0][index])
        #list_k_means_silhouette.append(silhouette_score_kmeans)
        
    
    axs[0].plot(list_n_clusters, list_k_means_homogenity, 'o', color='green', linestyle='solid', linewidth=2, label="Homogeneity Score")
    axs[0].plot(list_n_clusters, list_k_means_rand, 'o', color='yellow', linestyle='solid', linewidth=2, label="Rand Score")
    axs[0].plot(list_n_clusters, list_k_means_completness, 'o', color='blue', linestyle='solid', linewidth=2, label="Completness Score")

    axs[0].set_title('IRIS')
    axs[0].set_xlabel("n-clusters")
    
    axs[0].set_ylabel("Score") 

    #====== WINE =============================
    list_k_means_rand : list[float] = []
    list_k_means_homogenity : list[float] = []
    list_k_means_completness : list[float] = []
    for index in range(len(list_n_clusters)):
        #K-Means cluster
        klaster_KMeans: cluster.KMeans = cluster.KMeans(n_clusters=list_n_clusters[index])
        klaster_KMeans.fit(wine[0])
        y_pred[0][index] = klaster_KMeans.labels_.astype(int)

        print(f'WINE {index} CLUSTERS: {len(set(y_pred[0][index]))}')
        
        #Rand Score
        rand_score_kmeans : float = rand_score(np.ravel(wine[1]), np.ravel(y_pred[0][index]))
        list_k_means_rand.append(rand_score_kmeans)

        
        #Homogenity Score
        homogenity_score_kmeans : float = homogeneity_score(np.ravel(wine[1]), np.ravel(y_pred[0][index]))
        list_k_means_homogenity.append(homogenity_score_kmeans)
    
        #Completness Score
        completeness_score_kmeans : float = completeness_score(np.ravel(wine[1]), np.ravel(y_pred[0][index]))
        list_k_means_completness.append(completeness_score_kmeans)

    
    axs[1].plot(list_n_clusters, list_k_means_homogenity, 'o', color='green', linestyle='solid', linewidth=2, label="Homogeneity Score")
    axs[1].plot(list_n_clusters, list_k_means_rand, 'o', color='yellow', linestyle='solid', linewidth=2, label="Rand Score")
    axs[1].plot(list_n_clusters, list_k_means_completness, 'o', color='blue', linestyle='solid', linewidth=2, label="Completness Score")

    axs[1].set_title('WINE')
    axs[1].set_xlabel("n-clusters")
    
    axs[1].set_ylabel("Score") 
               
    #====== BREAST CANCER ==============
    list_k_means_rand : list[float] = []
    list_k_means_homogenity : list[float] = []
    list_k_means_completness : list[float] = []
    for index in range(len(list_n_clusters)):
        #K-Means cluster
        klaster_KMeans: cluster.KMeans = cluster.KMeans(n_clusters=list_n_clusters[index])
        klaster_KMeans.fit(breast_cancer[0])
        y_pred[0][index] = klaster_KMeans.labels_.astype(int)

        print(f'BREAST {index} CLUSTERS: {len(set(y_pred[0][index]))}')
        
        #Rand Score
        rand_score_kmeans : float = rand_score(np.ravel(breast_cancer[1]), np.ravel(y_pred[0][index]))
        list_k_means_rand.append(rand_score_kmeans)

        
        #Homogenity Score
        homogenity_score_kmeans : float = homogeneity_score(np.ravel(breast_cancer[1]), np.ravel(y_pred[0][index]))
        list_k_means_homogenity.append(homogenity_score_kmeans)
    
        #Completness Score
        completeness_score_kmeans : float = completeness_score(np.ravel(breast_cancer[1]), np.ravel(y_pred[0][index]))
        list_k_means_completness.append(completeness_score_kmeans)

    
    axs[2].plot(list_n_clusters, list_k_means_homogenity, 'o', color='green', linestyle='solid', linewidth=2, label="Homogeneity Score")
    axs[2].plot(list_n_clusters, list_k_means_rand, 'o', color='yellow', linestyle='solid', linewidth=2, label="Rand Score")
    axs[2].plot(list_n_clusters, list_k_means_completness, 'o', color='blue', linestyle='solid', linewidth=2, label="Completness Score")

    axs[2].set_title('BREAST CANCER')
    axs[2].set_xlabel("n-clusters")
    
    axs[2].set_ylabel("Score") 
      
    
    axs[0].legend()
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.savefig(f'experiment_2_KMEANS_IRIS_AND_OTHERS_scores.png')
    plt.subplots_adjust(hspace=0.6,wspace=0.5)
    plt.show()

def experiment_2_DBSCAN_IRIS_AND_OTHERS() -> None:
    fig, axs = plt.subplots(3,1)  
    
    y_pred : list[list[list[int]]] = [[[],[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[],[]],
                                      [[],[],[],[],[],[],[],[],[],[],[],[],[]]]
    
    iris          = datasets.load_iris(return_X_y=True)
    breast_cancer = datasets.load_breast_cancer(return_X_y=True)
    wine          = datasets.load_wine(return_X_y=True)
    #list_vision_best_clusters : list[int] = [2,2,9,2,4,4]
    #list_vision_worst_clusters : list[int] = [9,9,5,9,9,6]
    
    #============== IRIS ==============================================================
    list_k_means_rand : list[float] = []
    list_k_means_homogenity : list[float] = []
    list_k_means_completness : list[float] = []
    list_eps : list[float] = [0.01,0.15,0.2,0.25,0.3,5,85,100,125,150] 
    for index in range(len(list_eps)):
        #K-Means cluster
        klaster_KMeans: cluster.DBSCAN = cluster.DBSCAN(eps=list_eps[index],min_samples=10)
        klaster_KMeans.fit(iris[0])
        y_pred[0][index] = klaster_KMeans.labels_.astype(int)

        print(f'IRIS {index} CLUSTERS: {len(set(y_pred[0][index]))}')
        
        #Rand Score
        rand_score_kmeans : float = rand_score(np.ravel(iris[1]), np.ravel(y_pred[0][index]))
        list_k_means_rand.append(rand_score_kmeans)

        
        #Homogenity Score
        homogenity_score_kmeans : float = homogeneity_score(np.ravel(iris[1]), np.ravel(y_pred[0][index]))
        list_k_means_homogenity.append(homogenity_score_kmeans)
    
        #Completness Score
        completeness_score_kmeans : float = completeness_score(np.ravel(iris[1]), np.ravel(y_pred[0][index]))
        list_k_means_completness.append(completeness_score_kmeans)

        #Silhouette Score
        #silhouette_score_kmeans : float = silhouette_score(iris[1], y_pred[0][index])
        #list_k_means_silhouette.append(silhouette_score_kmeans)
        
    
    axs[0].plot(list_eps, list_k_means_homogenity, 'o', color='green', linestyle='solid', linewidth=2, label="Homogeneity Score")
    axs[0].plot(list_eps, list_k_means_rand, 'o', color='yellow', linestyle='solid', linewidth=2, label="Rand Score")
    axs[0].plot(list_eps, list_k_means_completness, 'o', color='blue', linestyle='solid', linewidth=2, label="Completness Score")

    axs[0].set_title('IRIS')
    axs[0].set_xlabel("eps")
    
    axs[0].set_ylabel("Score") 
    #Etykiety ilosci klastrów na wykresie
    for eps_text in range(len(list_eps)):
        axs[0].text(list_eps[eps_text], 0.1, len(set(y_pred[0][eps_text])))

    #====== WINE =============================
    list_k_means_rand : list[float] = []
    list_k_means_homogenity : list[float] = []
    list_k_means_completness : list[float] = []
    for index in range(len(list_eps)):
        #K-Means cluster
        klaster_KMeans: cluster.DBSCAN = cluster.DBSCAN(eps=list_eps[index],min_samples=10)
        klaster_KMeans.fit(wine[0])
        y_pred[0][index] = klaster_KMeans.labels_.astype(int)

        print(f'WINE {index} CLUSTERS: {len(set(y_pred[0][index]))}')
        
        #Rand Score
        rand_score_kmeans : float = rand_score(np.ravel(wine[1]), np.ravel(y_pred[0][index]))
        list_k_means_rand.append(rand_score_kmeans)

        
        #Homogenity Score
        homogenity_score_kmeans : float = homogeneity_score(np.ravel(wine[1]), np.ravel(y_pred[0][index]))
        list_k_means_homogenity.append(homogenity_score_kmeans)
    
        #Completness Score
        completeness_score_kmeans : float = completeness_score(np.ravel(wine[1]), np.ravel(y_pred[0][index]))
        list_k_means_completness.append(completeness_score_kmeans)

    
    axs[1].plot(list_eps, list_k_means_homogenity, 'o', color='green', linestyle='solid', linewidth=2, label="Homogeneity Score")
    axs[1].plot(list_eps, list_k_means_rand, 'o', color='yellow', linestyle='solid', linewidth=2, label="Rand Score")
    axs[1].plot(list_eps, list_k_means_completness, 'o', color='blue', linestyle='solid', linewidth=2, label="Completness Score")

    axs[1].set_title('WINE')
    axs[1].set_xlabel("eps")
    
    axs[1].set_ylabel("Score")
    #Etykiety ilosci klastrów na wykresie
    for eps_text in range(len(list_eps)):
        axs[1].text(list_eps[eps_text], 0.1, len(set(y_pred[1][eps_text]))) 
               
    #====== BREAST CANCER ==============
    list_k_means_rand : list[float] = []
    list_k_means_homogenity : list[float] = []
    list_k_means_completness : list[float] = []
    for index in range(len(list_eps)):
        #K-Means cluster
        klaster_KMeans: cluster.DBSCAN = cluster.DBSCAN(eps=list_eps[index],min_samples=10)
        klaster_KMeans.fit(breast_cancer[0])
        y_pred[0][index] = klaster_KMeans.labels_.astype(int)

        print(f'BREAST {index} CLUSTERS: {len(set(y_pred[0][index]))}')
        
        #Rand Score
        rand_score_kmeans : float = rand_score(np.ravel(breast_cancer[1]), np.ravel(y_pred[0][index]))
        list_k_means_rand.append(rand_score_kmeans)

        
        #Homogenity Score
        homogenity_score_kmeans : float = homogeneity_score(np.ravel(breast_cancer[1]), np.ravel(y_pred[0][index]))
        list_k_means_homogenity.append(homogenity_score_kmeans)
    
        #Completness Score
        completeness_score_kmeans : float = completeness_score(np.ravel(breast_cancer[1]), np.ravel(y_pred[0][index]))
        list_k_means_completness.append(completeness_score_kmeans)

    
    axs[2].plot(list_eps, list_k_means_homogenity, 'o', color='green', linestyle='solid', linewidth=2, label="Homogeneity Score")
    axs[2].plot(list_eps, list_k_means_rand, 'o', color='yellow', linestyle='solid', linewidth=2, label="Rand Score")
    axs[2].plot(list_eps, list_k_means_completness, 'o', color='blue', linestyle='solid', linewidth=2, label="Completness Score")

    axs[2].set_title('BREAST CANCER')
    axs[2].set_xlabel("eps")
    
    axs[2].set_ylabel("Score") 
    #Etykiety ilosci klastrów na wykresie
    for eps_text in range(len(list_eps)):
        axs[2].text(list_eps[eps_text], 0.1, len(set(y_pred[2][eps_text])))  
    
    axs[0].legend()
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.savefig(f'experiment_2_DBSCAN_IRIS_AND_OTHERS_scores.png')
    plt.subplots_adjust(hspace=0.6,wspace=0.5)
    plt.show()