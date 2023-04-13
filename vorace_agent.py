################################################################################
#
# Coded adapted from https://github.com/aloreggia/vorace
#
################################################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import xgboost as xgb
import keras.backend as K
from keras.layers import Dense
from keras.models import Model
from sklearn import metrics
from sklearn.base import clone
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import math
import random


class Vorace_agent:

    initialState = None
    classifier = None
    history = None
    callbacks_list = None
    epochs = 40
    batch_size = 0

    def __init__(self):
        self.initialState = None
        self.classifier = None
        self.history = None

    def __init__(self, typeP, nClasses, inputLayer=None, batch_size=0, callbacks_list=None, n_classifiers=10):

        if typeP == 6:
            typeP = random.randint(0, 5)
        if typeP == 3:
            typeP = random.randint(0, 2)

        self.batch_size = batch_size
        self.callbacks_list = callbacks_list
        # print(typeP)
        if typeP == 0:
            self.classifier = Vorace_agent.getModel(nClasses, inputLayer)
            self.classifier = Model(inputLayer, self.classifier)
            if nClasses == 2:
                self.classifier.compile(loss='binary_crossentropy', metrics=[
                                        'accuracy'], optimizer='adam')
            else:
                self.classifier.compile(loss='categorical_crossentropy', metrics=[
                                        'accuracy'], optimizer='adam')
            self.initialState = self.classifier.get_weights()

        elif typeP == 1:
            if random.randint(0, 1) == 0:
                self.classifier = DecisionTreeClassifier(criterion="gini", max_depth=random.randint(5, 25), random_state=0)
            else:
                self.classifier = DecisionTreeClassifier(criterion="entropy", max_depth=random.randint(5, 25), random_state=0)
            self.initialState = clone(self.classifier)

        elif typeP == 2:
            A = math.log(pow(2, -5))
            B = math.log(pow(2, 5))

            c_value = math.exp(random.uniform(A, B))

            if random.randint(0, 1) == 0:
                self.classifier = svm.SVC(
                    kernel='rbf', C=c_value, gamma='auto', probability=True)
            else:
                A = 3
                B = 5

                degree = int(round(random.uniform(A, B)))
                #print("C: {}  DEGREE: {}".format(c_value, degree))
                self.classifier = svm.SVC(
                    kernel='poly', degree=degree, C=c_value, gamma='auto', probability=True)
            self.initialState = clone(self.classifier)

        elif typeP == 4:

            value_lists = {'bootstrap': [True, False],
                           'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                           'max_features': ['auto', 'sqrt'],
                           'min_samples_leaf': [1, 2, 4],
                           'min_samples_split': [2, 5, 10],
                           'n_estimators': [10, 20, 50, 100, 200]}
            # 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

            params = {'bootstrap': random.choice(value_lists['bootstrap']),
                      'max_depth': random.choice(value_lists['max_depth']),
                      'max_features': random.choice(value_lists['max_features']),
                      'min_samples_leaf': random.choice(value_lists['min_samples_leaf']),
                      'min_samples_split': random.choice(value_lists['min_samples_split']),
                      'n_estimators': random.choice(value_lists['n_estimators']), }

            self.classifier = RandomForestClassifier(**params)
            self.initialState = clone(self.classifier)
        elif typeP == 5:
            self.classifier = xgb.XGBClassifier(max_depth=random.randint(3, 25), n_estimators=n_classifiers, subsample=random.random(), colsample_bytree=random.random())
            self.initialState = clone(self.classifier)

    def reset(self):
        if isinstance(self.classifier, Model):
            self.classifier.set_weights(self.initialState)
        else:
            self.classifier = clone(self.initialState)

    def fit(self, x, y, y_oneHot=None):
        if isinstance(self.classifier, Model):
            self.history = self.classifier.fit(x, y_oneHot, epochs=self.epochs, batch_size=self.batch_size, shuffle=True, callbacks=self.callbacks_list, verbose=0)
            self.history = self.history.history['accuracy'][-1]
        else:
            self.classifier.fit(x, y)
            y_pred = self.classifier.predict(x)
            self.history = metrics.accuracy_score(y, y_pred)

    def predict(self, x):
        if isinstance(self.classifier, Model):
            y_pred = self.classifier.predict(x, verbose=0)
        else:
            y_pred = self.classifier.predict_proba(x)
        return y_pred

    def getModel(nClass, inputLayer, nHLayers=4):

        n = random.randint(2, nHLayers)
        nInput = K.int_shape(inputLayer)[1]

        # A=math.log(nInput)
        # B=math.log(nInput**2)
        A = math.log(16)
        B = math.log(128)

        # print("A:"+str(A))
        # print("B:"+str(B))
        activation = ('relu', 'tanh')

        nNodes = int(round(math.exp(random.uniform(A, B))))
        act_fun = random.randint(0, len(activation)-1)
        # print("nNodes:"+str(nNodes))
        x = Dense(nNodes, activation=activation[act_fun])(inputLayer)
        # print(K.int_shape(inputLayer)[1])

        for i in range(1, n):
            #nNodes = random.randint(nInput*2,nInput**2)
            nNodes = int(round(math.exp(random.uniform(A, B))))
            # print(nNodes)
            act_fun = random.randint(0, len(activation)-1)
            # print(act_fun)
            x = Dense(nNodes, activation=activation[act_fun])(x)

        if nClass == 2:
            x = Dense(nClass, activation='sigmoid')(x)
        else:
            x = Dense(nClass, activation='softmax')(x)

        return x
