import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from warmup import load as load_csv

points : list[np.array] = []
labels : list[np.array] = []

def load_lists() -> None:
    points_1_1, labels_1_1 = load_csv("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\1_1.csv")
    points_1_1 = StandardScaler().fit_transform(points_1_1)
    points.append(points_1_1)
    labels.append(labels_1_1)

    points_1_2, labels_1_2 = load_csv("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\1_2.csv")
    points_1_2 = StandardScaler().fit_transform(points_1_2)
    points.append(points_1_2)
    labels.append(labels_1_2)

    points_1_3, labels_1_3 = load_csv("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\1_3.csv")
    points_1_3 = StandardScaler().fit_transform(points_1_3)
    points.append(points_1_3)
    labels.append(labels_1_3)

    points_2_1, labels_2_1 = load_csv("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\2_1.csv")
    points_2_1 = StandardScaler().fit_transform(points_2_1)
    points.append(points_2_1)
    labels.append(labels_2_1)

    points_2_2, labels_2_2 = load_csv("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\2_2.csv")
    points_2_2 = StandardScaler().fit_transform(points_2_2)
    points.append(points_2_2)
    labels.append(labels_2_2)

    points_2_3, labels_2_3 = load_csv("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\2_3.csv")
    points_2_3 = StandardScaler().fit_transform(points_2_3)
    points.append(points_2_3)
    labels.append(labels_2_3)
    

load_lists()