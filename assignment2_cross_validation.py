#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 11:06:02 2019

@author: Akhila Ashokan
"""

import pandas as pd
import math
from random import seed
from random import randrange
from statistics import mean

df = pd.read_excel('voting-data.xlsx')

#class that defines decision nodes
class Decision_Node:
    def __init__(self, feature, data):
        self.feature = feature
        self.list_of_feature_possibilities = []
        self.my_data = data #needed to keep data at each node so when pruning can determine
        #correct classifier without retraversing


#this method get a list of features and their possible attributes from the dataset
def get_list_of_features_and_possibilities(df):
    list_ = list(df)
    ret_matrix = []
    for feat in list_:
        column_of_df = df[feat]
        new_list = []
        for element in column_of_df:
            if not(element in new_list):
                new_list.append(element)
        ret_matrix.append(new_list)
    return ret_matrix

#get the proportion of a classifier in a dataset
def get_proportion(data, value, col):
    count = 0
    column = data[col]
    for x in column:
        if (x == value):
            count = count + 1
    return (float(count)/column.size)

#calculates entropy of dataset
def entropy(data):
    pos_proportion = get_proportion(data, 'democrat', 'classifier')
    neg_proportion = get_proportion(data, 'republican', 'classifier')
    if(pos_proportion==0 or neg_proportion==0):
        return 0
    entropy = -1*(pos_proportion*math.log(pos_proportion, 2)) - (neg_proportion*math.log(neg_proportion, 2))
    return entropy

#this method checks to see if all the classifers in the dataset are the same
def check_if_classifiers_all_same(data):
    classifier = data.iloc[0]['classifier']
    for index, row in data.iterrows():
        if classifier != row['classifier']:
            return False
    return classifier

#calculates the info gain of the dataset and feature using the entropy of the feature
def infoGain(data, feat, list_of_attributes_for_feature):
    sum_of_subsets = 0
    for attribute in list_of_attributes_for_feature:
        subset_for_attribute = create_subset(data, feat, attribute)
        if not(subset_for_attribute.empty):
            num = (subset_for_attribute.shape[0] / data.shape[0]) * entropy(subset_for_attribute)
            sum_of_subsets += num
        info_gain = entropy(data) - sum_of_subsets
    return info_gain

#creates the subset of data with a particular feature and attribute
def create_subset(data, feat, attribute):
    return_list = []
    for index, row in data.iterrows():
        if row[feat] == attribute:
            return_list.append(row)
    return pd.DataFrame(return_list)

# creates nested sublists after dividing about feature
def create_subset2(data, feat, list_in_question):
    return_list = []
    for attr in list_in_question:
        return_list.append(create_subset(data, feat, attr))
    return return_list

#returns the majority classifier after a decision node has been created
def get_majority_classifier(data):
    rep_count = 0
    dem_count = 0
    for index, row in data.iterrows():
        if row['classifier'] == 'democrat':
            dem_count = dem_count + 1
        else:
            rep_count = rep_count + 1
    if rep_count < dem_count:
        return 'democrat'
    else:
        return 'republican'

#returns the outcome of decision tree given a set of features
def determine_outcome(root, data_for_unknown_classifier, list_of_features_and_possibilities, list_of_feats):
    if root != None:
        if root.feature == 'democrat' or root.feature == 'republican':
            return root.feature
        ind_of_feat = list_of_feats.index(root.feature)
        if ind_of_feat >= len(list_of_features_and_possibilities):
            print(list_of_features_and_possibilities, root.feature)
        list_of_possibilities = list_of_features_and_possibilities[ind_of_feat]
        index = list_of_possibilities.index(data_for_unknown_classifier[root.feature])
        if index != -1:
            return determine_outcome(root.list_of_feature_possibilities[index], data_for_unknown_classifier,
                                     list_of_features_and_possibilities, list_of_feats)
        else:
            return "N/A"

#returns the error rate when testing a decision tree against the validation set
def test_against_validation_set(root, list_of_features_and_possibilities, list_of_feats,df_val_set):
    count_correct = 0
    count_incorrect = 0
    for index, row in df_val_set.iterrows():
        decision_trees_answer = determine_outcome(root, row,
                                                  list_of_features_and_possibilities,
                                                  list_of_feats)
        if decision_trees_answer == row['classifier']:
            count_correct = count_correct + 1
        else:
            count_incorrect = count_incorrect + 1
    return float(count_incorrect/(count_correct+count_incorrect))

#returns the error rate when testing a decision tree against the training set
def test_against_training_set(root, list_of_features_and_possibilities, list_of_feats, training_set):
    count_correct = 0
    count_incorrect = 0
    for index, row in training_set.iterrows():
        decision_trees_answer = determine_outcome(root, row,
                                                  list_of_features_and_possibilities,
                                                  list_of_feats)
        if decision_trees_answer == row['classifier']:
            count_correct = count_correct + 1
        else:
            count_incorrect = count_incorrect + 1
    return float(count_incorrect/(count_correct+count_incorrect))

# core of algorithm. creates notes and branches depending on infoGain
# 'data' = whole df, 'list_of_features' = list of features left to examine (i.e. outlook)
def id3 (data, list_of_features, list_of_features_and_possibilities):     
    if data.empty:
        return None
    potential_classifier = check_if_classifiers_all_same(data)
    if potential_classifier != False:
        return  Decision_Node(potential_classifier, data)
    if (len(list_of_features) == 0):
        return Decision_Node(get_majority_classifier(data), data)
    max_info_gain = 0
    best_feature = 0
    best_feat_num = 0
    feat_num = 0
    for feat in list_of_features:
        info_gain_of_set = infoGain(data, feat,list_of_features_and_possibilities[feat_num])
        if info_gain_of_set > max_info_gain:
            max_info_gain = info_gain_of_set
            best_feature = feat
            best_feat_num = feat_num
        feat_num = feat_num + 1
    if best_feature == 0:
        return Decision_Node(get_majority_classifier(data), data)
    node = Decision_Node(best_feature, data)
    partitioned_data = create_subset2(data, best_feature, list_of_features_and_possibilities[best_feat_num])
    # copy the list_of_features and list_of_features_and_possibilities into sub-lists
    sub_list_of_features = list(list_of_features)
    sub_list_of_features_and_possibilities = list(list_of_features_and_possibilities)
    del sub_list_of_features_and_possibilities[sub_list_of_features.index(best_feature)]
    sub_list_of_features.remove(best_feature)
    for sub_list in partitioned_data:
        node.list_of_feature_possibilities.append(id3(sub_list, sub_list_of_features, sub_list_of_features_and_possibilities))
    return node

#traverses the decision tree given a root
def traverse(root):
    if root != None:
        print(root.feature)
        for x in root.list_of_feature_possibilities:
            traverse(x)

#checks if the classifiers exists and if not returns false
def check_if_classifiers_exists(list_of_feature_possibilities):
    dem = 0
    rep = 0
    for x in list_of_feature_possibilities:

        if x != None:
            if x.feature == 'democrat':
                dem += 1
            elif x.feature == 'republican':
                rep += 1
            else:
                return False
    return dem == 1 or rep == 1

                
#splits the data set into different folds
def cross_validation(df, k):
    df_list = df.values.tolist()
    df_split = list()
    df_copy = list(df_list)
    fold_size = int(len(df_list) / k)
    for i in range(k):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(df_copy))
            fold.append(df_copy.pop(index))
        df_split.append(fold)
    return df_split

#allows us to reproduce results despite the random generator
seed(1)

def main():

    list_of_features_and_possibilities = [
            ['n', '?', 'y'], ['n', '?', 'y'], ['n', '?', 'y'], ['n', '?', 'y'],
            ['n', '?', 'y'], ['n', '?', 'y'], ['n', '?', 'y'], ['n', '?', 'y'],
            ['n', '?', 'y'], ['n', '?', 'y'], ['n', '?', 'y'], ['n', '?', 'y'],
            ['n', '?', 'y'], ['n', '?', 'y'], ['n', '?', 'y'], ['n', '?', 'y']]
    list_of_feats = list(df)
    del list_of_feats[-1]
    
    accuracy_against_validation = [] #list of accuracy rates against validation set
    #create folds for cross validation given a k value (second parameter)
    folds = cross_validation(df, 2)

    for i in range(len(folds)):

        folds_copy = folds.copy()
        #creates validation data
        validation_list = folds_copy.pop(i)        
        df_val_set = pd.DataFrame(validation_list, columns = list(df.columns))
        df_val_set = df_val_set.reset_index(drop=True)


        #creates training data
        train_df = pd.DataFrame(folds_copy.pop(), columns = list(df.columns))
        while folds_copy:
            temp_df = pd.DataFrame(folds_copy.pop(), columns = list(df.columns))
            train_df = train_df.append(temp_df, ignore_index = True)
        train_df = train_df.reset_index(drop=True)       
    
            
        root = id3(train_df, list_of_feats, list_of_features_and_possibilities)                
                      
        #store the score of best pruned tree for that fold 
        accuracy_against_validation.append(test_against_validation_set(root, list_of_features_and_possibilities, list_of_feats, df_val_set))

    #calculates the mean accuracy against validation for a given k value
    print("K-value:2 ", "Error rate: ", mean(accuracy_against_validation))            
            


if __name__ == "__main__": main()
