from imports import *

class View:

    def information(self, df):
        forme = df.shape
        inform = df.info()
        nombre_de_cuisines = df[['cuisine']].value_counts()
    
        return forme, nombre_de_cuisines,inform