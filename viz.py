import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def information(df):
    forme = df.shape
    inform = df.info()
    nombre_de_cuisines = df[['cuisine']].value_counts()
    
    return (forme, nombre_de_cuisines,inform)