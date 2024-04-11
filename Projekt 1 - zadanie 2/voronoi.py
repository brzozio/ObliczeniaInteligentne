import numpy as np
import pandas as pd

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def load(path) -> np.array:
    """
    Funkcja powinna wczytywać plik CSV, którego lokalizacja wskazywana jest przez argument
    oraz zwracać dwie tablice NumPy o rozmiarach Nxn oraz N, gdzie N to liczba obiektów,
    a n to liczba wymiarów. Tablice te odpowiadają cechom N obiektów w n-wymiarowej przestrzeni
    (liczby rzeczywiste) oraz ich etyketom (liczby całkowite od 0 do L-1 gdzie L to liczba
    etykiet). Zakładamy, że w pliku CSV jest N linii odpowiadających obiektom, a każda linia
    zaweira n+1 liczb odpowiadających wpierw kolejnym cechom obiektu (n wartości) i jego
    etykiecie (1 wartość). Liczby w każdej linii pliku CSV oddzielone są średnikami.
    """

    X = pd.read_csv(path, sep = ";", usecols=[0,1])
    Labels = pd.read_csv(path, sep = ";", usecols=[2])

    cechy = np.array(X)
    etykiety = np.array(Labels)
     
    return cechy, etykiety


def plot_voronoi_diagram(X, y_true, y_pred) -> plt.Figure:
    """
    Funkcja rysująca diagram Voronoia dla obiektów opisanych tablicą X rozmiaru Nx2 (N to liczba
    obiektów) pogrupowanych za pomocą etykiet y_pred (tablica liczby całkowitych o rozmiarze N).
    Parametr y_true może być równy None, i wtedy nie znamy prawdziwych etykiet, lub być tablicą
    N elementową z prawdziwymi etykietami. Rysując diagram należy zadbać, aby wszystkie obiekty
    były widoczne. Wszystkie rozważane tablice są tablicami NumPy.
    """
   
    # Tworzenie diagramu Voronoi na podstawie punktów X
    vor = Voronoi(X)

    # Tworzenie wykresu
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False)

    # Kolorowanie obszarów Voronoi na podstawie przypisanych etykiet
    if y_true is None:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    else:
        unique_labels = np.unique(y_true)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # Kolorowanie według unikalnych etykiet

    for region, label in zip(vor.regions, y_pred):
        if not -1 in region and label != -1:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=colors[label % len(colors)], alpha=0.4)

    # Dodanie punktów danych do wykresu
    ax.plot(X[:, 0], X[:, 1], 'ko', markersize=3)

    # Dodanie legendy dla etykiet prawdziwych (jeśli dostępne)
    if y_true is not None:
        unique_labels = np.unique(y_true)
        for label, color in zip(unique_labels, colors):
            ax.plot([], [], 'o', label=str(label), markersize=8, color=color)

        ax.legend(title='True Labels')

    # Ustawienie tytułu i etykiet osi
    ax.set_title('Diagram Voronoi')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

    return fig


def plot_decision_boundary(X, func, y_true=None)-> plt.figure:
    fig, ax = plt.subplots()
     # Definiujemy zakres dla osi x i y
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Tworzymy siatkę punktów w celu wygenerowania granicy decyzyjnej
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Obliczamy etykiety dla każdego punktu w siatce
    Z = func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    print(Z)

    #Predykcja etykiet dla danych testowych podanych w funkcji
    if y_true is None:
        y_tested = func(X)
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
    X, y_true = load("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Zadanie rozgrzewkowe\\warmup.csv")
    X = StandardScaler().fit_transform(X)

    algorithm = cluster.KMeans(n_clusters=3)
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(int)
    plot_voronoi_diagram(X, y_true, y_pred)
    plot_voronoi_diagram(X, None, y_pred)

    algorithm = KNeighborsClassifier(n_neighbors=3)
    algorithm.fit(X, y_true)
    plot_decision_boundary(X, y_true, algorithm.predict)
