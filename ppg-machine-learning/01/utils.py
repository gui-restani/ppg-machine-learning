import numpy as np

def load_bike_dataset():
    data = np.loadtxt(open("dataset/Bike-Sharing-Dataset/day.csv", "rb"), delimiter=",", skiprows=1, usecols=range(2,14))
    target = np.loadtxt(open("dataset/Bike-Sharing-Dataset/day.csv", "rb"), delimiter=",", skiprows=1, usecols=15)
    return data, target
