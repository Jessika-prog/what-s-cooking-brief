from imports import *

class Cook:
    
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
    # split X in self.X_train, self.X_test, self.y_train, self.y_test
    # return none
    def train_test_split(self):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.33, random_state=42)
   
    #-------------------------------------------   
    # execute preprocessing on X_train and X_test
    # return none
    def preprocessing(self, X_train, X_test):
        self.X_train = pd.DataFrame(self.multilab.fit_transform(
            X_train), columns=self.multilab.classes_)
        self.X_test = pd.DataFrame(self.multilab.transform(
            X_test), columns=self.multilab.classes_)
    
    #-------------------------------------------  
    # execute preprocessing on X
    # return none
    def cross_preprocessing(self, X_cross):
        self.X = pd.DataFrame(self.multilab.fit_transform(
            X_cross), columns=self.multilab.classes_)
    
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
    # Test one model via train_test method and execute all process
    # return none
    def test_model(self, model):
        
        self.multilab = MultiLabelBinarizer()
        
        #remove special ingredients
        self.X = pd.Series(self.removing_special_ingredients(self.X))

        #remove adjectives, verbs, etc.
        #self.X = self.pos_tag_ingredients(self.X)

        #train test split
        self.train_test_split()

        # preprocessing
        self.preprocessing(self.X_train, self.X_test)

        # model_fit
        model_fit = self.model_fit(self.X_train, self.y_train, model)

        # model_predict
        y_pred = self.model_predict(self.X_test, model_fit)

        # model Score
        print(f" Le score pour {str(model)} est : {self.score(self.y_test, y_pred)}")
    
    #-------------------------------------------    
    # Test one model via cross validation method and execute all process
    # return none    
    def cross_model(self, model):
        
        self.multilab = MultiLabelBinarizer()
        
        #remove special ingredients
        self.X = pd.Series(self.removing_special_ingredients(self.X))

        # cross_preprocessing
        self.cross_preprocessing(self.X)

        # cross_val_model_fit
        print(f" Le cross_score pour {str(model)} est : {cross_val_score(model, self.X, self.y, cv=3).mean()}")
        
    #-------------------------------------------    
    # Test a stacking model via train_test method and execute all process
    # return none
    def stacking_training(self, model_1, model_2):
        
        model = StackingClassifier([
            ('model_1', model_1),('model_2', model_2)
        ], final_estimator=LinearSVC())

        self.test_model(model)
        
    #-------------------------------------------   
    # Test a best params for a SGDlassifier model
    # return none    
    def randomize_training(self):
        
        self.randomize_model = SGDClassifier()

        self.multilab = MultiLabelBinarizer()
               
        #remove special ingredients
        self.X = pd.Series(self.removing_special_ingredients(self.X))
        
        #remove adjectives, verbs, etc.
        #self.X = self.pos_tag_ingredients(self.X)
        
        #train test split
        self.train_test_split()

        # preprocessing
        self.preprocessing(self.X_train, self.X_test)

        # model_fit                 
        distributions = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate
                'penalty': ['l2'],
                'n_jobs': [-1]}
        
        rscv = RandomizedSearchCV(self.randomize_model, param_distributions = distributions, cv = 3, random_state=0)
        random_search = rscv.fit(self.X_train,self.y_train)

        print(f"""
        Les meilleurs paramètres sont : {random_search.best_params_}
        """)
        
        self.best_params=random_search.best_params_
               
        # model_predict
        y_pred = random_search.best_estimator_.predict(self.X_test)

        # model Score
        print(f" Le score pour le randomSearch({str(self.randomize_model)}) est : {self.score(self.y_test, y_pred)}")      
                  