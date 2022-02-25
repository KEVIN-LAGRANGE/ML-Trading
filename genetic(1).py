# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:12:56 2021

@author: 87118
"""
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
def classification_backtest(rate, y_pred_retrun):
    '''如果predict return > 0 就会拿到真的return'''
    y_act = np.where(y_pred_retrun > 0, 1, 0)
    return np.dot(rate, y_act) - sum(rate)

class genetic_model(object):
    def __init__(self, X_train, X_test, y_train, y_rate, model_name, population_size=10,\
                 mutation_rate=0.7, crossover_rate=0.2, elitism=0.1, generations=10, early_stop=3,\
                     population_candidate=[]):
        self.feature_num = len(X_train.columns)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_rate = y_rate
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.model_name = model_name
        self.elitism = elitism
        # self.replication_rate = replication_rate
        if population_candidate:
            self.population = population_candidate
        else:
            self.population = [self.get_random_dic() for i in range(self.population_size)]
        self.score = []
        self.new_population = []
        self.highest_score = 0
        self.best_clf = 0
        self.bset_individual = 0
        self.early_stop = early_stop
        
    def get_random_dic(self):
        ''' 生成random 个体
        feature_num 总共feature 个数
        model_name  模型名字'''
        # n components
        rand_int = random.randint(1, self.feature_num)
        # max depth
        rand_int_depth = random.randint(1, 12)
        # c, float
        c = random.random()
        if self.model_name == "RandomForestClassifier":
            return {"PCA":{"n_components":rand_int}, self.model_name:{"max_depth":rand_int_depth}}
        return {"PCA":{"n_components":rand_int}, self.model_name:{"C":c}}
    
    def fitness(self, params):
        ''' params is individual'''
        if self.model_name == "SVC":
            clf = make_pipeline(
            StandardScaler(),
            #SelectKBest(k='all'),
            PCA(**params["PCA"]), 
            # LogisticRegression(C=c)
            SVC(**params["SVC"]))
            # RandomForestClassifier(max_depth=c))
        elif self.model_name == "LogisticRegression":
            clf = make_pipeline(
                StandardScaler(),
                #SelectKBest(k='all'),
                PCA(**params["PCA"]), 
                LogisticRegression(**params["LogisticRegression"]))
        elif self.model_name == "RandomForestClassifier":
            clf = make_pipeline(
                StandardScaler(),
                #SelectKBest(k='all'),
                PCA(**params["PCA"]), 
                RandomForestClassifier(**params["RandomForestClassifier"]))
        else:
            print("invalid model_name", self.model_name)
        clf.fit(self.X_train, self.y_train)
        prediction = clf.predict(self.X_test)
        # prediction = clf.predict_proba(X_test)

        # print(prediction)
        score = classification_backtest(self.y_rate, prediction)    
        return score, clf
    
    def run(self):
        # print(self.population)
        # for each generation
        best_generation = 0
        for i in range(self.generations): 
            if i - best_generation >= self.early_stop:
                print(f"no update after{self.early_stop}")
                break
            print(f"{i} generation")
            self.new_population = []
            self.score = []

            # # evaluate every individaul
            for individual in self.population:
                score,clf = self.fitness(individual)
                print(score)
                self.score.append(score)
                if score >= self.highest_score:
                    self.highest_score = score
                    self.best_clf = clf
                    self.bset_individual = individual
                    best_generation = i
                print(self.highest_score, "highest")
            # add new population until the generations
            while len(self.new_population) < len(self.population):
                self.action()
                print(len(self.new_population))
            
            #use new population
            self.population = self.new_population    
        
                
    def action(self):
        ''' choose action and choose individual and append new population'''
        # according to score to select individual
        # /  then use probability to chose  operation    
        rand_num = random.random()   
        if rand_num < self.elitism:
            # draw individual based on probability
            individual = random.choices(self.population, weights=self.score)[0]
            self.new_population.append(individual)
                
        elif rand_num < self.elitism + self.crossover_rate:
            # draw two individual based  on probability
            individual_one = random.choices(self.population, weights=self.score)[0]
            individual_two = random.choices(self.population, weights=self.score)[0]
            # cross over swith one of the key
            switch_key = random.choices(list(individual_one))[0]
            switch_info = individual_one[switch_key]
            switch_info_2 = individual_two[switch_key]
            individual_one[switch_key] = switch_info_2
            individual_two[switch_key] = switch_info
            self.new_population.append(individual_one)
            self.new_population.append(individual_two)
            
        else:
            # mutation
            individual = random.choices(self.population, weights=self.score)[0]
            individual = self.mutation(individual)
            self.new_population.append(individual)

    def mutation(self, individual_one):
        switch_key = random.choices(list(individual_one))[0]
        if switch_key == "PCA":
            individual_one[switch_key]["n_components"] += random.randint(-10, 10)
            individual_one[switch_key]["n_components"] = max(individual_one[switch_key]["n_components"], 1)
            individual_one[switch_key]["n_components"] = min(individual_one[switch_key]["n_components"], self.feature_num)

        elif switch_key == "RandomForestClassifier":
            individual_one[switch_key]["max_depth"] += random.randint(-2, 2)
            individual_one[switch_key]["max_depth"] = max(individual_one[switch_key]["max_depth"], 1)

        else:
            individual_one[switch_key]["C"] += random.randint(-2, 2) / 10
            individual_one[switch_key]["C"] = max(individual_one[switch_key]["C"], 0.01)
        return individual_one