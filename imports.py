import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
from scipy.stats import uniform

import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.preprocessing import MultiLabelBinarizer
from nltk import word_tokenize, pos_tag, pos_tag_sents

from sklearn import set_config
set_config(display='diagram')

