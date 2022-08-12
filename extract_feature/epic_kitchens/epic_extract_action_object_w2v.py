# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 2022

@author: alachyankar
"""
import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import pdb
import pandas as pd
import numpy as np
import gensim.downloader as api
import pickle
#%%
print('Loading pretrain w2v model')
model_name = 'word2vec-google-news-300'#best model
model = api.load(model_name)
dim_w2v = 300
print('Done loading model')
#%%
replace_word = [('','')]
#%%


def init_values(anno):
    values_df =  pd.DataFrame({"verb_class": anno["verb_class"], "noun_class": anno["noun_class"]})
    dedup_values_df = values_df.drop_duplicates()
    return dedup_values_df.sort_values(by=["verb_class", "noun_class"]).reset_index(drop=True)


annotations_path = 'data/epic_kitchens/epic-kitchens-100-annotations/EPIC_100_train.csv'
anno = pd.read_csv(annotations_path)
all_interactions =  init_values(anno)
print(all_interactions.head(5))


 
object_path = 'data/epic_kitchens/epic-kitchens-100-annotations/EPIC_100_noun_classes.csv'
df_obj = pd.read_csv(object_path)
objects_unique = df_obj['key'].unique()
objects_unique_ids = df_obj['id'].unique()

actions_path = 'data/epic_kitchens/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv'
df_act = pd.read_csv(actions_path)
actions_unique = df_act['key'].unique()
actions_unique_ids = df_act['id'].unique()

#object_ids = [np.where(object_unique_ids = obj_id)[0][0] for obj_id in all_interactions['noun_class'].values]


Z_o = np.eye(len(objects_unique))[all_interactions['noun_class']]
Z_a = np.eye(len(actions_unique))[all_interactions['verb_class']]

print("Z_o: {}, Z_a: {}".format(Z_o.shape, Z_a.shape))
print("Z_o Head\n{}".format(Z_o[:5]))
print("Z_a Head\n{}".format(Z_a[:5]))
#%%
#%% pre-processing
def preprocessing_actions(words):
    new_words = [' '.join(i.split('-')) for i in words]
    return new_words


def preprocessing_objects(words):
    return [' '.join(i.split(':')[::-1]) for i in words]
    
#%%
print(">>>>actions<<<<")
actions_w2v = []
for s in preprocessing_actions(actions_unique):
    #print(s)
    words = s.split(' ')
    if words[-1] == '':     #remove empty element
        words = words[:-1]
    w2v = np.zeros(dim_w2v)
    for w in words:
        try:
            w2v += model[w]
        except Exception as e:
            print(e)
    actions_w2v.append(w2v[np.newaxis,:])
actions_w2v=np.concatenate(actions_w2v,axis=0)
#%%
print(">>>>objects<<<<")
objects_w2v = []
for s in preprocessing_objects(objects_unique):
    #print(s)
    words = s.split(' ')
    if words[-1] == '':     #remove empty element
        words = words[:-1]
    w2v = np.zeros(dim_w2v)
    for w in words:
        try:
            w2v += model[w]
        except Exception as e:
            print(e)
    objects_w2v.append(w2v[np.newaxis,:])
objects_w2v=np.concatenate(objects_w2v,axis=0)
#%%

content = {'actions_w2v':actions_w2v,'objects_w2v':objects_w2v,'Z_a':Z_a,'Z_o':Z_o}
with open('./w2v/epic_kitchens_act_obj.pkl','wb') as f:
    pickle.dump(content,f)
