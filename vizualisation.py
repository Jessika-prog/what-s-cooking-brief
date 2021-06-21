from imports import *


class Vizualisation:
    
    def __init__(self, df):
        self.df = pd.read_json(df)
        
    def df_info(self):
        
        len_df = len(self.df)
        all_columns = len(self.df.columns)
        all_nan = self.df.isna().sum().sum()

        print(f"""
        Longueur du dataset : {len_df} enregistrements
        Nombre de colonnes : {all_columns}
        """)

        obs = pd.DataFrame({
            'type': list(self.df.dtypes),
            '% de valeurs nulles': round(self.df.isna().sum() / len_df * 100, 2)
#           'Nbr L dupliquées' : (self.df.duplicated()).sum(),
#           'Nbr V unique' : self.df.nunique()
        })
        
        return obs


    def country_recipes(self):

        cuisine_imp=pd.DataFrame(self.df['cuisine'].value_counts())
        fig, ax = plt.subplots(figsize=(10,10))
        cuisine_imp.plot(y='cuisine', kind='bar', legend=False, title='Most represented countries',grid=True, ax=ax);
        
    def nb_ingredients_by_country(self):
        cuisines = self.df["cuisine"].unique()

        all_cus = dict()
        for cs in cuisines:
            i = list()
            for ing_list in self.df[self.df['cuisine']==cs]['ingredients']:
                for ing in ing_list:
                    i.append(ing)
            all_cus[cs] = i     

        for key in all_cus.keys():
            fig, ax = plt.subplots(figsize=(14,4))
            pd.Series(all_cus[key]).value_counts().head(15).plot.bar(ax=ax, title=key)
            plt.show()
            
    def nb_ingredients_by_recipes(self):
        
        ingredients_number=[]
        for ingredients in self.df['ingredients']:
            ingredients_number.append(len(ingredients))
        
        fig, ax = plt.subplots(figsize=(14,7))
        sns.histplot(x = ingredients_number, ax=ax).set_title("Nombre d'ingrédients par recettes");

        
