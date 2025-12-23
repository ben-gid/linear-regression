# TODO
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

RANDOM_STATE = 55

df = pd.read_csv("/data/heart.csv")

cat_variables = 