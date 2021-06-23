from imports import *

class Prediction:

    def __init__(self, name):

        self.df = pd.read_json(name)
        
        self.X = self.df.ingredients
        self.y = self.df.cuisine
        
        self.linear_params = "C = 0.22685190926977272, penalty = 'l2'"
        
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
 
    def train_test_split(self):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.33, random_state=42)
   
    def preprocessing(self, X_train, X_test):
        self.X_train = pd.DataFrame(self.multilab.fit_transform(
            X_train), columns=self.multilab.classes_)
        self.X_test = pd.DataFrame(self.multilab.transform(
            X_test), columns=self.multilab.classes_)

    def model_fit(self, X_train, y_train):
        self.model_fit = self.model.fit(X_train, y_train)

    def model_predict(self, X_test):
        self.y_pred = self.model_fit.predict(X_test)

    def score(self, y_test, y_pred):
        self.accuracy = accuracy_score(y_test, y_pred)
        print(self.accuracy)

    def mega_process(self, x_test, best_params=None):
        
        self.multilab = MultiLabelBinarizer()
        self.model = LogisticRegression(best_params)
        
        x_pred = pd.read_json(x_test)
        self.X_test = x_pred.ingredients
        
        #remove special ingredients
        self.X = pd.Series(self.removing_special_ingredients(self.X))
        
        #keep adjectives, verbs, etc.
        #self.X = self.pos_tag_ingredients(self.X)

        # preprocessing
        self.preprocessing(self.X, self.X_test)

        # model_fit
        self.model_fit(self.X_train, self.y)

        # model_predict
        self.model_predict(self.X_test)

        
        self.submission = pd.DataFrame({'id':x_pred['id'],'cuisine': self.y_pred})
        
        self.submission.to_csv('submission.csv', index=False)
        
        return self.submission