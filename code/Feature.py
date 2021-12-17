import numpy as np
import pandas as pd
import networkx as nx

BASE_DIR = './data/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'

# feature extractions

train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

df = pd.concat([train_df, test_df])


g = nx.Graph()
g.add_nodes_from(df.question1)
g.add_nodes_from(df.question2)
edges = list(df[['question1', 'question2']].to_records(index=False))
g.add_edges_from(edges)

def q1_freq(row):
    return(len(list(g.neighbors(row.question1))))
    
def q2_freq(row):
    return(len(list(g.neighbors(row.question2))))

def get_intersection_count(row):
    return(len(set(g.neighbors(row.question1)).intersection(set(g.neighbors(row.question2)))))

train_ic = pd.DataFrame()
test_ic = pd.DataFrame()


train_df['intersection_count'] = train_df.apply(lambda row: get_intersection_count(row), axis=1)
test_df['intersection_count'] = test_df.apply(lambda row: get_intersection_count(row), axis=1)
train_df['q1_count'] = train_df.apply(lambda row: q1_freq(row), axis=1)
test_df['q1_count'] = test_df.apply(lambda row: q1_freq(row), axis=1)
train_df['q2_count'] = train_df.apply(lambda row: q2_freq(row), axis=1)
test_df['q2_count'] = test_df.apply(lambda row: q2_freq(row), axis=1)

train_ic['q1_count'] = train_df['q1_count']
train_ic['q2_count'] = train_df['q2_count']
train_ic['intersection_count'] = train_df['intersection_count']
test_ic['q1_count'] = test_df['q1_count']
test_ic['q2_count'] = test_df['q2_count']
test_ic['intersection_count'] = test_df['intersection_count']

train_ic.to_csv("./data/train_ic.csv", index=False)
test_ic.to_csv("./data/test_ic.csv", index=False)

train_df.to_csv("./data/train_intsec.csv", index=False)
test_df.to_csv("./data/test_intsec.csv", index=False)
