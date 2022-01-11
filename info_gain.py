#Information gain function

def two_group_ent(first, tot):
    return -(first/tot*np.log2(first/tot) +
             (tot-first)/tot*np.log2((tot-first)/tot))

tot_ent = two_group_ent(10, 24)
g17_ent = 15/24 * two_group_ent(11,15) +
           9/24 * two_group_ent(6,9)

answer = tot_ent - g17_ent

"""
    Large depth very often causes overfitting, since a tree that is too deep, can memorize the data.
    Small depth can result in a very simple model, which may cause underfitting.
    Small minimum samples per split may result in a complicated, highly branched tree, which can mean the model has memorized the data, or in other words, overfit.
    Large minimum samples may result in the tree not having enough flexibility to get built, and may result in underfitting.
"""

#Compute antropy for insects assignment
import numpy as np
import pandas as pd

def get_counts(df, query):
    mobugs = df[df.Species == "Mobug"]
    lobugs = df[df.Species == "Lobug"]
    q_ = query or "index == index"
    return len(mobugs.query(q_)), len(lobugs.query(q_))

df = pd.read_csv("ml-bugs.csv")

n_mobugs,        n_lobugs        = get_counts(df, "")
n_brown_mobugs,  n_brown_lobugs  = get_counts(df, "Color == 'Brown'")
n_blue_mobugs,   n_blue_lobugs   = get_counts(df, "Color == 'Blue'")
n_green_mobugs,  n_green_lobugs  = get_counts(df, "Color == 'Green'")
n_less17_mobugs, n_less17_lobugs = get_counts(df, "`Length (mm)` < 17.0")
n_less20_mobugs, n_less20_lobugs = get_counts(df, "`Length (mm)` < 20.0")
