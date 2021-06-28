from imports import *

class Prediction:

   #-------------------------------------------   
    # init object 
    # return object
    def __init__(self, name):

        self.df = pd.read_json(name)      
        self.X = self.df.ingredients
        self.y = self.df.cuisine
        
    #-------------------------------------------   
    # remove special character from ingredients 
    # return array
    def removing_special_ingredients(self, df):
        
        new_ingredient_list=[]
        
        for row in df:
            new_recepe_ingredient_list=[]
            for ingredient in row:
                regex=re.compile('[@_!#$%^&*()<>?/\|.}{~:1234567890]')
                if(regex.search(ingredient) == None) & (len(ingredient)> 2):
                    new_recepe_ingredient_list.append(ingredient)
            
            new_ingredient_list.append(new_recepe_ingredient_list)
        
        return new_ingredient_list
    
    #-------------------------------------------   
    # just keep name, adverb and verb from ingredients 
    # return pd.serie
    def pos_tag_ingredients(self, serie):
        
        serie = serie.map(lambda x: ' '.join(x))
        
        tagged_texts = pos_tag_sents(map(word_tokenize, serie))
        
        filtered_tags = []
        for index, value in enumerate(tagged_texts):
            filtered_tags.append([])
            for j in value:
                #if j[-1].startswith(("N", "R")):
                if j[-1].startswith(("N", "R", "V")):
                    filtered_tags[index].append(j[0])
        
        return filtered_tags        
   
    #-------------------------------------------   
    # execute preprocessing on X_train and X_test
    # return none
    def preprocessing(self, X_train, X_test):
        self.X_train = pd.DataFrame(self.multilab.fit_transform(
            X_train), columns=self.multilab.classes_)
        self.X_test = pd.DataFrame(self.multilab.transform(
            X_test), columns=self.multilab.classes_)
    
    #-------------------------------------------   
    # fit model with kind of model send
    # return object fit model
    def model_fit(self, X_train, y_train, model):
        return model.fit(X_train, y_train)
    
    #-------------------------------------------   
    # predict data with fit model
    # return array of prediction
    def model_predict(self, X_test, model_fit):
        return model_fit.predict(X_test)

    #-------------------------------------------   
    # evaluate accuracy of model
    # return accuracy number
    def score(self, y_test, y_pred):
        self.accuracy = accuracy_score(y_test, y_pred)
        return self.accuracy    
    
    #-------------------------------------------   
    # Predict with on model and save submission on csv file
    # return DataFrame of submission
    def submission(self, x_test, model):
        
        self.multilab = MultiLabelBinarizer()
        
        x_pred = pd.read_json(x_test)
        self.X_test = x_pred.ingredients
              
        #remove special ingredients
        self.X = pd.Series(self.removing_special_ingredients(self.X))
        
        #remove adjectives, verbs, etc.
        #self.X = self.pos_tag_ingredients(self.X)
        
         # preprocessing
        self.preprocessing(self.X, self.X_test)

        # model_fit
        model_fit = self.model_fit(self.X_train, self.y, model)

        # model_predict
        y_pred = self.model_predict(self.X_test, model_fit)
             
        self.submission = pd.DataFrame({'id':x_pred['id'],'cuisine': y_pred})
        
        self.submission.to_csv('submission.csv', index=False)
        
        return self.submission
