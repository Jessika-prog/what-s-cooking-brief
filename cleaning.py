from imports import *

data = Data()
df= data.df_train

class Cleaning:


    def de_liste_a_colonnes(df):
        X=0
        i=0
        for value in df['ingredients']:
            df.at[i,'ingredients_text']=', '.join(value)
            i+=1

        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
        # Use the content column instead of our single text variable
        matrix = vectorizer.fit_transform(df.ingredients_text)
        counts = pd.DataFrame(matrix.toarray(),
                      columns=vectorizer.get_feature_names())
        counts.head()
        X=pd.concat([df,counts],axis=1)
        return X

        de_liste_a_colonnes(df)
    
#     def Removing_special_ingredients(df):
#         """ Removing all ingredients that contains special characters and numbers
#         """
#         new_ingredient_list=[]
#         # new list with all incredients. 
#         for index,list_ingredients in df[['ingredients']].iterrows():
#             new_recepe_ingredient_list =[]
#             for ingredient in list_ingredients.values[0]:
#                 regex=re.compile('[@_!#$%^&*()<>?/\|.}{~:1234567890]')
#                 if(regex.search(ingredient) == None) & (len(ingredient)> 2):
#                     new_recepe_ingredient_list.append(ingredient)
#                 df.at[index,'ingredients']=new_recepe_ingredient_list
#     #        new_ingredient_list.append(new_recepe_ingredient_list)
#     #    df['clean_ingredient_list']=new_ingredient_list
#         return df
#     df=Removing_special_ingredients(df)

# def X_y_selection(df):
#     X=df
#     del (X['cuisine'], X['id'])
#     y=df['cuisine']
#     return X,y