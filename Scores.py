from collections import OrderedDict,Counter
import json,subprocess,math,os
from random import randint
# from multiprocessing import Pool
import pandas as pd
import concurrent.futures
PAMDict={
    'PAM300':{
        'A' : [2, -1, 0, 0, -2, 0, 0, 2, -1, 0, -2, -1, -1, -4, 1, 1, 1, -6, -4, 0, 0] ,
        'R' : [-1, 7, 0, -1, -4, 2, -1, -2, 2, -2, -3, 4, 0, -5, 0, 0, -1, 3, -5, -3, 0] ,
        'N' : [0, 0, 2, 2, -4, 1, 2, 1, 2, -2, -3, 1, -2, -4, 0, 1, 0, -5, -2, -2, 0] ,
        'D' : [0, -1, 2, 4, -6, 2, 4, 1, 1, -2, -4, 0, -3, -6, -1, 0, 0, -7, -5, -2, 0] ,
        'C' : [-2, -4, -4, -6, 15, -6, -6, -4, -4, -3, -7, -6, -6, -5, -3, 0, -2, -9, 1, -2, 0] ,
        'Q' : [0, 2, 1, 2, -6, 4, 3, -1, 3, -2, -2, 1, -1, -5, 0, 0, -1, -5, -4, -2, 0] ,
        'E' : [0, -1, 2, 4, -6, 3, 4, 0, 1, -2, -4, 0, -2, -6, 0, 0, 0, -8, -5, -2, 0] ,
        'G' : [2, -2, 1, 1, -4, -1, 0, 5, -2, -3, -4, -2, -3, -5, 0, 1, 0, -8, -6, -1, 0] ,
        'H' : [-1, 2, 2, 1, -4, 3, 1, -2, 7, -2, -2, 0, -2, -2, 0, -1, -1, -3, 0, -2, 0] ,
        'I' : [0, -2, -2, -2, -3, -2, -2, -3, -2, 5, 3, -2, 3, 1, -2, -1, 0, -6, -1, 4, 0] ,
        'L' : [-2, -3, -3, -4, -7, -2, -4, -4, -2, 3, 7, -3, 4, 3, -3, -3, -2, -2, 0, 2, 0] ,
        'K' : [-1, 4, 1, 0, -6, 1, 0, -2, 0, -2, -3, 5, 0, -6, -1, 0, 0, -4, -5, -2, 0] ,
        'M' : [-1, 0, -2, -3, -6, -1, -2, -3, -2, 3, 4, 0, 6, 1, -2, -2, -1, -5, -2, 2, 0] ,
        'F' : [-4, -5, -4, -6, -5, -5, -6, -5, -2, 1, 3, -6, 1, 11, -5, -4, -3, 1, 9, -1, 0] ,
        'P' : [1, 0, 0, -1, -3, 0, 0, 0, 0, -2, -3, -1, -2, -5, 6, 1, 1, -6, -5, -1, 0] ,
        'S' : [1, 0, 1, 0, 0, 0, 0, 1, -1, -1, -3, 0, -2, -4, 1, 1, 1, -3, -3, -1, 0] ,
        'T' : [1, -1, 0, 0, -2, -1, 0, 0, -1, 0, -2, 0, -1, -3, 1, 1, 2, -6, -3, 0, 0] ,
        'W' : [-6, 3, -5, -7, -9, -5, -8, -8, -3, -6, -2, -4, -5, 1, -6, -3, -6, 22, 0, -7, 0] ,
        'Y' : [-4, -5, -2, -5, 1, -4, -5, -6, 0, -1, 0, -5, -2, 9, -5, -3, -3, 0, 12, -3, 0] ,
        'V' : [0, -3, -2, -2, -2, -2, -2, -1, -2, 4, 2, -2, 2, -1, -1, -1, 0, -7, -3, 5, 0] ,
        '-' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
        },
    'PAM120':{
        'A' : [3, -3, -1, 0, -3, -1, 0, 1, -3, -1, -3, -2, -2, -4, 1, 1, 1, -7, -4, 0, 0] ,
        'R' : [-3, 6, -1, -3, -4, 1, -3, -4, 1, -2, -4, 2, -1, -5, -1, -1, -2, 1, -5, -3, 0] ,
        'N' : [-1, -1, 4, 2, -5, 0, 1, 0, 2, -2, -4, 1, -3, -4, -2, 1, 0, -4, -2, -3, 0] ,
        'D' : [0, -3, 2, 5, -7, 1, 3, 0, 0, -3, -5, -1, -4, -7, -3, 0, -1, -8, -5, -3, 0] ,
        'C' : [-3, -4, -5, -7, 9, -7, -7, -4, -4, -3, -7, -7, -6, -6, -4, 0, -3, -8, -1, -3, 0] ,
        'Q' : [-1, 1, 0, 1, -7, 6, 2, -3, 3, -3, -2, 0, -1, -6, 0, -2, -2, -6, -5, -3, 0] ,
        'E' : [0, -3, 1, 3, -7, 2, 5, -1, -1, -3, -4, -1, -3, -7, -2, -1, -2, -8, -5, -3, 0] ,
        'G' : [1, -4, 0, 0, -4, -3, -1, 5, -4, -4, -5, -3, -4, -5, -2, 1, -1, -8, -6, -2, 0] ,
        'H' : [-3, 1, 2, 0, -4, 3, -1, -4, 7, -4, -3, -2, -4, -3, -1, -2, -3, -3, -1, -3, 0] ,
        'I' : [-1, -2, -2, -3, -3, -3, -3, -4, -4, 6, 1, -3, 1, 0, -3, -2, 0, -6, -2, 3, 0] ,
        'L' : [-3, -4, -4, -5, -7, -2, -4, -5, -3, 1, 5, -4, 3, 0, -3, -4, -3, -3, -2, 1, 0] ,
        'K' : [-2, 2, 1, -1, -7, 0, -1, -3, -2, -3, -4, 5, 0, -7, -2, -1, -1, -5, -5, -4, 0] ,
        'M' : [-2, -1, -3, -4, -6, -1, -3, -4, -4, 1, 3, 0, 8, -1, -3, -2, -1, -6, -4, 1, 0] ,
        'F' : [-4, -5, -4, -7, -6, -6, -7, -5, -3, 0, 0, -7, -1, 8, -5, -3, -4, -1, 4, -3, 0] ,
        'P' : [1, -1, -2, -3, -4, 0, -2, -2, -1, -3, -3, -2, -3, -5, 6, 1, -1, -7, -6, -2, 0] ,
        'S' : [1, -1, 1, 0, 0, -2, -1, 1, -2, -2, -4, -1, -2, -3, 1, 3, 2, -2, -3, -2, 0] ,
        'T' : [1, -2, 0, -1, -3, -2, -2, -1, -3, 0, -3, -1, -1, -4, -1, 2, 4, -6, -3, 0, 0] ,
        'W' : [-7, 1, -4, -8, -8, -6, -8, -8, -3, -6, -3, -5, -6, -1, -7, -2, -6, 12, -2, -8, 0] ,
        'Y' : [-4, -5, -2, -5, -1, -5, -5, -6, -1, -2, -2, -5, -4, 4, -6, -3, -3, -2, 8, -3, 0] ,
        'V' : [0, -3, -3, -3, -3, -3, -3, -2, -3, 3, 1, -4, 1, -3, -2, -2, 0, -8, -3, 5, 0] ,
        '-' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
    },
    'PAM400':{
    'A' : [1, -1, 0, 1, -2, 0, 1, 1, -1, 0, -2, -1, -1, -3, 1, 1, 1, -6, -3, 0, 0] ,
    'R' : [-1, 5, 1, 0, -4, 1, 0, -2, 2, -2, -3, 4, 0, -4, 0, 0, 0, 3, -4, -2, 0] ,
    'N' : [0, 1, 1, 2, -3, 1, 2, 1, 1, -1, -3, 1, -1, -3, 0, 1, 0, -4, -3, -1, 0] ,
    'D' : [1, 0, 2, 3, -5, 2, 3, 1, 1, -2, -3, 1, -2, -5, 0, 1, 0, -7, -4, -2, 0] ,
    'C' : [-2, -4, -3, -5, 17, -5, -5, -3, -3, -2, -6, -5, -5, -4, -2, 0, -2, -8, 1, -2, 0] ,
    'Q' : [0, 1, 1, 2, -5, 3, 2, 0, 3, -2, -2, 1, -1, -4, 0, 0, 0, -5, -4, -1, 0] ,
    'E' : [1, 0, 2, 3, -5, 2, 3, 1, 1, -2, -3, 1, -2, -5, 0, 0, 0, -7, -4, -1, 0] ,
    'G' : [1, -2, 1, 1, -3, 0, 1, 4, -1, -2, -3, -1, -2, -5, 0, 1, 1, -7, -5, -1, 0] ,
    'H' : [-1, 2, 1, 1, -3, 3, 1, -1, 5, -2, -2, 1, -1, -2, 0, 0, -1, -2, 0, -2, 0] ,
    'I' : [0, -2, -1, -2, -2, -2, -2, -2, -2, 4, 3, -2, 2, 2, -1, -1, 0, -5, 0, 3, 0] ,
    'L' : [-2, -3, -3, -3, -6, -2, -3, -3, -2, 3, 7, -2, 4, 3, -2, -2, -1, -2, 0, 3, 0] ,
    'K' : [-1, 4, 1, 1, -5, 1, 1, -1, 1, -2, -2, 4, 0, -5, 0, 0, 0, -3, -4, -2, 0] ,
    'M' : [-1, 0, -1, -2, -5, -1, -2, -2, -1, 2, 4, 0, 5, 1, -1, -1, 0, -4, -1, 2, 0] ,
    'F' : [-3, -4, -3, -5, -4, -4, -5, -5, -2, 2, 3, -5, 1, 11, -4, -3, -3, 2, 10, 0, 0] ,
    'P' : [1, 0, 0, 0, -2, 0, 0, 0, 0, -1, -2, 0, -1, -4, 5, 1, 1, -6, -5, -1, 0] ,
    'S' : [1, 0, 1, 1, 0, 0, 0, 1, 0, -1, -2, 0, -1, -3, 1, 1, 1, -3, -3, -1, 0] ,
    'T' : [1, 0, 0, 0, -2, 0, 0, 1, -1, 0, -1, 0, 0, -3, 1, 1, 1, -5, -3, 0, 0] ,
    'W' : [-6, 3, -4, -7, -8, -5, -7, -7, -2, -5, -2, -3, -4, 2, -6, -3, -5, 26, 1, -6, 0] ,
    'Y' : [-3, -4, -3, -4, 1, -4, -4, -5, 0, 0, 0, -4, -1, 10, -5, -3, -3, 1, 13, -2, 0] ,
    'V' : [0, -2, -1, -2, -2, -1, -1, -1, -2, 3, 3, -2, 2, 0, -1, -1, 0, -6, -2, 4, 0] ,
    '-' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,  
        },
    'PAM500':{
    'A' : [1, -1, 0, 1, -2, 0, 1, 1, 0, 0, -1, 0, -1, -3, 1, 1, 1, -6, -3, 0, 0] ,
    'R' : [-1, 5, 1, 0, -4, 2, 0, -1, 2, -2, -2, 4, 0, -4, 0, 0, 0, 4, -4, -2, 0] ,
    'N' : [0, 1, 1, 2, -3, 1, 1, 1, 1, -1, -2, 1, -1, -4, 0, 1, 0, -5, -3, -1, 0] ,
    'D' : [1, 0, 2, 3, -5, 2, 3, 1, 1, -2, -3, 1, -2, -5, 0, 1, 0, -7, -5, -1, 0] ,
    'C' : [-2, -4, -3, -5, 22, -5, -5, -3, -4, -2, -6, -5, -5, -3, -2, 0, -2, -9, 2, -2, 0] ,
    'Q' : [0, 2, 1, 2, -5, 2, 2, 0, 2, -1, -2, 1, -1, -4, 1, 0, 0, -5, -4, -1, 0] ,
    'E' : [1, 0, 1, 3, -5, 2, 3, 1, 1, -2, -3, 1, -1, -5, 0, 1, 0, -7, -5, -1, 0] ,
    'G' : [1, -1, 1, 1, -3, 0, 1, 4, -1, -2, -3, 0, -2, -5, 1, 1, 1, -8, -5, -1, 0] ,
    'H' : [0, 2, 1, 1, -4, 2, 1, -1, 4, -2, -2, 1, -1, -2, 0, 0, 0, -2, 0, -2, 0] ,
    'I' : [0, -2, -1, -2, -2, -1, -2, -2, -2, 3, 4, -2, 3, 2, -1, -1, 0, -5, 0, 3, 0] ,
    'L' : [-1, -2, -2, -3, -6, -2, -3, -3, -2, 4, 7, -2, 4, 4, -2, -2, -1, -1, 1, 3, 0] ,
    'K' : [0, 4, 1, 1, -5, 1, 1, 0, 1, -2, -2, 4, 0, -5, 0, 0, 0, -3, -5, -2, 0] ,
    'M' : [-1, 0, -1, -2, -5, -1, -1, -2, -1, 3, 4, 0, 4, 1, -1, -1, 0, -4, -1, 2, 0] ,
    'F' : [-3, -4, -4, -5, -3, -4, -5, -5, -2, 2, 4, -5, 1, 13, -4, -3, -3, 3, 13, 0, 0] ,
    'P' : [1, 0, 0, 0, -2, 1, 0, 1, 0, -1, -2, 0, -1, -4, 4, 1, 1, -6, -5, -1, 0] ,
    'S' : [1, 0, 1, 1, 0, 0, 1, 1, 0, -1, -2, 0, -1, -3, 1, 1, 1, -3, -3, -1, 0] ,
    'T' : [1, 0, 0, 0, -2, 0, 0, 1, 0, 0, -1, 0, 0, -3, 1, 1, 1, -6, -3, 0, 0] ,
    'W' : [-6, 4, -5, -7, -9, -5, -7, -8, -2, -5, -1, -3, -4, 3, -6, -3, -6, 34, 2, -6, 0] ,
    'Y' : [-3, -4, -3, -5, 2, -4, -5, -5, 0, 0, 1, -5, -1, 13, -5, -3, -3, 2, 15, -1, 0] ,
    'V' : [0, -2, -1, -1, -2, -1, -1, -1, -2, 3, 3, -2, 2, 0, -1, -1, 0, -6, -1, 3, 0] ,
    '-' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,   
    },
    'PAM30':{
        'A' : [6, -7, -4, -3, -6, -4, -2, -2, -7, -5, -6, -7, -5, -8, -2, 0, -1, -13, -8, -2, 0] ,
        'R' : [-7, 8, -6, -10, -8, -2, -9, -9, -2, -5, -8, 0, -4, -9, -4, -3, -6, -2, -10, -8, 0] ,
        'N' : [-4, -6, 8, 2, -11, -3, -2, -3, 0, -5, -7, -1, -9, -9, -6, 0, -2, -8, -4, -8, 0] ,
        'D' : [-3, -10, 2, 8, -14, -2, 2, -3, -4, -7, -12, -4, -11, -15, -8, -4, -5, -15, -11, -8, 0] ,
        'C' : [-6, -8, -11, -14, 10, -14, -14, -9, -7, -6, -15, -14, -13, -13, -8, -3, -8, -15, -4, -6, 0] ,
        'Q' : [-4, -2, -3, -2, -14, 8, 1, -7, 1, -8, -5, -3, -4, -13, -3, -5, -5, -13, -12, -7, 0] ,
        'E' : [-2, -9, -2, 2, -14, 1, 8, -4, -5, -5, -9, -4, -7, -14, -5, -4, -6, -17, -8, -6, 0] ,
        'G' : [-2, -9, -3, -3, -9, -7, -4, 6, -9, -11, -10, -7, -8, -9, -6, -2, -6, -15, -14, -5, 0] ,
        'H' : [-7, -2, 0, -4, -7, 1, -5, -9, 9, -9, -6, -6, -10, -6, -4, -6, -7, -7, -3, -6, 0] ,
        'I' : [-5, -5, -5, -7, -6, -8, -5, -11, -9, 8, -1, -6, -1, -2, -8, -7, -2, -14, -6, 2, 0] ,
        'L' : [-6, -8, -7, -12, -15, -5, -9, -10, -6, -1, 7, -8, 1, -3, -7, -8, -7, -6, -7, -2, 0] ,
        'K' : [-7, 0, -1, -4, -14, -3, -4, -7, -6, -6, -8, 7, -2, -14, -6, -4, -3, -12, -9, -9, 0] ,
        'M' : [-5, -4, -9, -11, -13, -4, -7, -8, -10, -1, 1, -2, 11, -4, -8, -5, -4, -13, -11, -1, 0] ,
        'F' : [-8, -9, -9, -15, -13, -13, -14, -9, -6, -2, -3, -14, -4, 9, -10, -6, -9, -4, 2, -8, 0] ,
        'P' : [-2, -4, -6, -8, -8, -3, -5, -6, -4, -8, -7, -6, -8, -10, 8, -2, -4, -14, -13, -6, 0] ,
        'S' : [0, -3, 0, -4, -3, -5, -4, -2, -6, -7, -8, -4, -5, -6, -2, 6, 0, -5, -7, -6, 0] ,
        'T' : [-1, -6, -2, -5, -8, -5, -6, -6, -7, -2, -7, -3, -4, -9, -4, 0, 7, -13, -6, -3, 0] ,
        'W' : [-13, -2, -8, -15, -15, -13, -17, -15, -7, -14, -6, -12, -13, -4, -14, -5, -13, 13, -5, -15, 0] ,
        'Y' : [-8, -10, -4, -11, -4, -12, -8, -14, -3, -6, -7, -9, -11, 2, -13, -7, -6, -5, 10, -7, 0] ,
        'V' : [-2, -8, -8, -8, -6, -7, -6, -5, -6, 2, -2, -9, -1, -8, -6, -6, -3, -15, -7, 7, 0] ,
        '-' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,  
        }
    
    }
blosumDict={
    
    'blosum100':{
        'A': [8, -3, -4, -5, -2, -2, -3, -1, -4, -4, -4, -2, -3, -5, -2, 1, -1, -6, -5, -2, 0],
        'R': [-3, 10, -2, -5, -8, 0, -2, -6, -1, -7, -6, 3, -4, -6, -5, -3, -3, -7, -5, -6, 0],
        'N': [-4, -2, 11, 1, -5, -1, -2, -2, 0, -7, -7, -1, -5, -7, -5, 0, -1, -8, -5, -7, 0],
        'D': [-5, -5, 1, 10, -8, -2, 2, -4, -3, -8, -8, -3, -8, -8, -5, -2, -4, -10, -7, -8, 0],
        'C': [-2, -8, -5, -8, 14, -7, -9, -7, -8, -3, -5, -8, -4, -4, -8, -3, -3, -7, -6, -3, 0],
        'Q': [-2, 0, -1, -2, -7, 11, 2, -5, 1, -6, -5, 2, -2, -6, -4, -2, -3, -5, -4, -5, 0],
        'E': [-3, -2, -2, 2, -9, 2, 10, -6, -2, -7, -7, 0, -5, -8, -4, -2, -3, -8, -7, -5, 0],
        'G': [-1, -6, -2, -4, -7, -5, -6, 9, -6, -9, -8, -5, -7, -8, -6, -2, -5, -7, -8, -8, 0],
        'H': [-4, -1, 0, -3, -8, 1, -2, -6, 13, -7, -6, -3, -5, -4, -5, -3, -4, -5, 1, -7, 0],
        'I': [-4, -7, -7, -8, -3, -6, -7, -9, -7, 8, 2, -6, 1, -2, -7, -5, -3, -6, -4, 4, 0],
        'L': [-4, -6, -7, -8, -5, -5, -7, -8, -6, 2, 8, -6, 3, 0, -7, -6, -4, -5, -4, 0, 0],
        'K': [-2, 3, -1, -3, -8, 2, 0, -5, -3, -6, -6, 10, -4, -6, -3, -2, -3, -8, -5, -5, 0],
        'M': [-3, -4, -5, -8, -4, -2, -5, -7, -5, 1, 3, -4, 12, -1, -5, -4, -2, -4, -5, 0, 0],
        'F': [-5, -6, -7, -8, -4, -6, -8, -8, -4, -2, 0, -6, -1, 11, -7, -5, -5, 0, 4, -3, 0],
        'P': [-2, -5, -5, -5, -8, -4, -4, -6, -5, -7, -7, -3, -5, -7, 12, -3, -4, -8, -7, -6, 0],
        'S': [1, -3, 0, -2, -3, -2, -2, -2, -3, -5, -6, -2, -4, -5, -3, 9, 2, -7, -5, -4, 0],
        'T': [-1, -3, -1, -4, -3, -3, -3, -5, -4, -3, -4, -3, -2, -5, -4, 2, 9, -7, -5, -1, 0],
        'W': [-6, -7, -8, -10, -7, -5, -8, -7, -5, -6, -5, -8, -4, 0, -8, -7, -7, 17, 2, -5, 0],
        'Y': [-5, -5, -5, -7, -6, -4, -7, -8, 1, -4, -4, -5, -5, 4, -7, -5, -5, 2, 12, -5, 0],
        'V': [-2, -6, -7, -8, -3, -5, -5, -8, -7, 4, 0, -5, 0, -3, -6, -4, -1, -5, -5, 8, 0],
        '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    'blosum62':{
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],  # A
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],  # R
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],  # N
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],  # D
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],  # C
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0],  # Q
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 0],  # E
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],  # G
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],  # H
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],  # I
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],  # L
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],  # K
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],  # M
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],  # F
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],  # P
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0],  # S
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0],  # T
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],  # W
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, 0],  # Y
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, 0],  # V
    '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    },
    'blosum45':{
    'A': [5, -2, -1, -2, -1, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -2, -2,  0, 0],  # A
    'R': [-2,  7,  0, -1, -3,  1,  0, -2,  0, -3, -2,  3, -1, -2, -2, -1, -1, -2, -1, -2, 0],  # R
    'N': [-1,  0,  6,  2, -2,  0,  0,  0,  1, -2, -3,  0, -2, -2, -2,  1,  0, -4, -2, -3, 0],  # N
    'D': [-2, -1,  2,  7, -3,  0,  2, -1,  0, -4, -3,  0, -3, -4, -1,  0, -1, -4, -2, -3, 0],  # D
    'C': [-1, -3, -2, -3, 12, -3, -3, -3, -3, -3, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1, 0],  # C
    'Q': [-1,  1,  0,  0, -3,  6,  2, -2,  1, -2, -2,  1,  0, -4, -1,  0, -1, -2, -1, -3, 0],  # Q
    'E': [-1,  0,  0,  2, -3,  2,  6, -2,  0, -3, -2,  1, -2, -3,  0,  0, -1, -3, -2, -3, 0],  # E
    'G': [0, -2,  0, -1, -3, -2, -2,  7, -2, -4, -3, -2, -2, -3, -2,  0, -2, -2, -3, -3, 0],  # G
    'H': [-2,  0,  1,  0, -3,  1,  0, -2, 10, -3, -2, -1,  0, -2, -2, -1, -2, -3,  2, -3, 0],  # H
    'I': [-1, -3, -2, -4, -3, -2, -3, -4, -3,  5,  2, -3,  2,  0, -2, -2, -1, -2,  0,  3, 0],  # I
    'L': [-1, -2, -3, -3, -2, -2, -2, -3, -2,  2,  5, -3,  2,  1, -3, -3, -1, -2,  0,  1, 0],  # L
    'K': [-1,  3,  0,  0, -3,  1,  1, -2, -1, -3, -3,  5, -1, -3, -1, -1, -1, -2, -1, -2, 0],  # K
    'M': [-1, -1, -2, -3, -2,  0, -2, -2,  0,  2,  2, -1,  6,  0, -2, -2, -1, -2,  0,  1, 0],  # M
    'F': [-2, -2, -2, -4, -2, -4, -3, -3, -2,  0,  1, -3,  0,  8, -3, -2, -1,  1,  3,  0, 0],  # F
    'P': [-1, -2, -2, -1, -4, -1,  0, -2, -2, -2, -3, -1, -2, -3,  9, -1, -1, -3, -3, -3, 0],  # P
    'S': [1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -3, -1, -2, -2, -1,  4,  2, -4, -2, -1, 0],  # S
    'T': [0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -1, -1,  2,  5, -3, -1,  0, 0],  # T
    'W': [-2, -2, -4, -4, -5, -2, -3, -2, -3, -2, -2, -2, -2,  1, -3, -4, -3, 15,  3, -3, 0],  # W
    'Y': [-2, -1, -2, -2, -3, -1, -2, -3,  2,  0,  0, -1,  0,  3, -3, -2, -1,  3,  8, -1, 0],  # Y
    'V': [0, -2, -3, -3,-1, -3, -3, -3, -3,  3,  1, -2,  1,  0, -3, -1,  0, -3, -1,  5, 0],  # V
    '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    },
    'blosum75':{
    'A' : [4, -2, -2, -2, -1, -1, -1, 0, -2, -2, -2, -1, -1, -3, -1, 1, 0, -3, -2, 0, 0] ,
    'R' : [-2, 6, -1, -2, -4, 1, 0, -3, 0, -3, -3, 2, -2, -3, -2, -1, -1, -3, -2, -3, 0] ,
    'N' : [-2, -1, 6, 1, -3, 0, -1, -1, 0, -4, -4, 0, -3, -4, -3, 0, 0, -4, -3, -3, 0] ,
    'D' : [-2, -2, 1, 6, -4, -1, 1, -2, -1, -4, -4, -1, -4, -4, -2, -1, -1, -5, -4, -4, 0] ,
    'C' : [-1, -4, -3, -4, 9, -3, -5, -3, -4, -1, -2, -4, -2, -2, -4, -1, -1, -3, -3, -1, 0] ,
    'Q' : [-1, 1, 0, -1, -3, 6, 2, -2, 1, -3, -3, 1, 0, -4, -2, 0, -1, -2, -2, -2, 0] ,
    'E' : [-1, 0, -1, 1, -5, 2, 5, -3, 0, -4, -4, 1, -2, -4, -1, 0, -1, -4, -3, -3, 0] ,
    'G' : [0, -3, -1, -2, -3, -2, -3, 6, -2, -5, -4, -2, -3, -4, -3, -1, -2, -3, -4, -4, 0] ,
    'H' : [-2, 0, 0, -1, -4, 1, 0, -2, 8, -4, -3, -1, -2, -2, -2, -1, -2, -2, 2, -4, 0] ,
    'I' : [-2, -3, -4, -4, -1, -3, -4, -5, -4, 4, 1, -3, 1, 0, -3, -3, -1, -3, -2, 3, 0] ,
    'L' : [-2, -3, -4, -4, -2, -3, -4, -4, -3, 1, 4, -3, 2, 0, -3, -3, -2, -2, -1, 1, 0] ,
    'K' : [-1, 2, 0, -1, -4, 1, 1, -2, -1, -3, -3, 5, -2, -4, -1, 0, -1, -4, -2, -3, 0] ,
    'M' : [-1, -2, -3, -4, -2, 0, -2, -3, -2, 1, 2, -2, 6, 0, -3, -2, -1, -2, -2, 1, 0] ,
    'F' : [-3, -3, -4, -4, -2, -4, -4, -4, -2, 0, 0, -4, 0, 6, -4, -3, -2, 1, 3, -1, 0] ,
    'P' : [-1, -2, -3, -2, -4, -2, -1, -3, -2, -3, -3, -1, -3, -4, 8, -1, -1, -5, -4, -3, 0] ,
    'S' : [1, -1, 0, -1, -1, 0, 0, -1, -1, -3, -3, 0, -2, -3, -1, 5, 1, -3, -2, -2, 0] ,
    'T' : [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -2, -1, -1, -2, -1, 1, 5, -3, -2, 0, 0] ,
    'W' : [-3, -3, -4, -5, -3, -2, -4, -3, -2, -3, -2, -4, -2, 1, -5, -3, -3, 11, 2, -3, 0] ,
    'Y' : [-2, -2, -3, -4, -3, -2, -3, -4, 2, -2, -1, -2, -2, 3, -4, -2, -2, 2, 7, -2, 0] ,
    'V' : [0, -3, -3, -4, -1, -2, -3, -4, -4, 3, 1, -3, 1, -1, -3, -2, 0, -3, -2, 4, 0] ,
    '-' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] , 
    },
    'blosum30':{
    'A' : [4, -1, 0, 0, -3, 1, 0, 0, -2, 0, -1, 0, 1, -2, -1, 1, 1, -5, -4, 1, 0] ,
    'R' : [-1, 8, -2, -1, -2, 3, -1, -2, -1, -3, -2, 1, 0, -1, -1, -1, -3, 0, 0, -1, 0] ,
    'N' : [0, -2, 8, 1, -1, -1, -1, 0, -1, 0, -2, 0, 0, -1, -3, 0, 1, -7, -4, -2, 0] ,
    'D' : [0, -1, 1, 9, -3, -1, 1, -1, -2, -4, -1, 0, -3, -5, -1, 0, -1, -4, -1, -2, 0] ,
    'C' : [-3, -2, -1, -3, 17, -2, 1, -4, -5, -2, 0, -3, -2, -3, -3, -2, -2, -2, -6, -2, 0] ,
    'Q' : [1, 3, -1, -1, -2, 8, 2, -2, 0, -2, -2, 0, -1, -3, 0, -1, 0, -1, -1, -3, 0] ,
    'E' : [0, -1, -1, 1, 1, 2, 6, -2, 0, -3, -1, 2, -1, -4, 1, 0, -2, -1, -2, -3, 0] ,
    'G' : [0, -2, 0, -1, -4, -2, -2, 8, -3, -1, -2, -1, -2, -3, -1, 0, -2, 1, -3, -3, 0] ,
    'H' : [-2, -1, -1, -2, -5, 0, 0, -3, 14, -2, -1, -2, 2, -3, 1, -1, -2, -5, 0, -3, 0] ,
    'I' : [0, -3, 0, -4, -2, -2, -3, -1, -2, 6, 2, -2, 1, 0, -3, -1, 0, -3, -1, 4, 0] ,
    'L' : [-1, -2, -2, -1, 0, -2, -1, -2, -1, 2, 4, -2, 2, 2, -3, -2, 0, -2, 3, 1, 0] ,
    'K' : [0, 1, 0, 0, -3, 0, 2, -1, -2, -2, -2, 4, 2, -1, 1, 0, -1, -2, -1, -2, 0] ,
    'M' : [1, 0, 0, -3, -2, -1, -1, -2, 2, 1, 2, 2, 6, -2, -4, -2, 0, -3, -1, 0, 0] ,
    'F' : [-2, -1, -1, -5, -3, -3, -4, -3, -3, 0, 2, -1, -2, 10, -4, -1, -2, 1, 3, 1, 0] ,
    'P' : [-1, -1, -3, -1, -3, 0, 1, -1, 1, -3, -3, 1, -4, -4, 11, -1, 0, -3, -2, -4, 0] ,
    'S' : [1, -1, 0, 0, -2, -1, 0, 0, -1, -1, -2, 0, -2, -1, -1, 4, 2, -3, -2, -1, 0] ,
    'T' : [1, -3, 1, -1, -2, 0, -2, -2, -2, 0, 0, -1, 0, -2, 0, 2, 5, -5, -1, 1, 0] ,
    'W' : [-5, 0, -7, -4, -2, -1, -1, 1, -5, -3, -2, -2, -3, 1, -3, -3, -5, 20, 5, -3, 0] ,
    'Y' : [-4, 0, -4, -1, -6, -1, -2, -3, 0, -1, 3, -1, -1, 3, -2, -2, -1, 5, 9, 1, 0] ,
    'V' : [1, -1, -2, -2, -2, -3, -3, -3, -3, 4, 1, -2, 0, 1, -4, -1, 1, -3, 1, 5, 0] ,
    '-' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
        }
    }

def _ppm_paraProcessing(id,seqs,lengthmatrix,dict_AAPPM):
    _ppm_results=[]
    for eachseq in seqs:
        scoreseq = 0
        for i in range(lengthmatrix):
            if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                scoreseq += 0
            else:
                scoreseq += dict_AAPPM[eachseq[i]][i]
        PPMscore = scoreseq / lengthmatrix
        # dict_predictnameseqPPM[eachname][eachseq] = PPMscore
        _ppm_results.append((eachseq, PPMscore))
    return _ppm_results,id
def PPM(predictpeptide, length, trainpeptide, matrix):
    if matrix == None:
        postrainpep, negtrainpep = [], []
        for key, value in trainpeptide.items():
            if value == '1':
                postrainpep.append(key)
            else:
                negtrainpep.append(key)

        dict_AAPPM = {
            'A': [], 'R': [], 'N': [], 'D': [], 'C': [],
            'Q': [], 'E': [], 'G': [], 'H': [], 'I': [],
            'L': [], 'K': [], 'M': [], 'F': [], 'P': [],
            'S': [], 'T': [], 'W': [], 'Y': [], 'V': [],
            '-': []
        }
        for i in range(int(length)):
            dict_AAfre = {
                'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0,
                'Q': 0, 'E': 0, 'G': 0, 'H': 0, 'I': 0,
                'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0,
                'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0, '-': 0
            }
            for eachseq in trainpeptide:
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                    continue
                else:
                    dict_AAfre[eachseq[i]] += 1
            for eachAA, eachvallist in dict_AAPPM.items():
                if dict_AAfre[eachAA] == 0:
                    eachvallist.append(1 / len(trainpeptide))  # set the minimum value
                else:
                    eachvallist.append(dict_AAfre[eachAA] / len(trainpeptide))

        dict_trainPPM = OrderedDict()
        for eachseq in postrainpep:
            scoreseq = 0
            for i in range(int(length)):
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                    scoreseq += 0
                else:
                    scoreseq += dict_AAPPM[eachseq[i]][i]
            PPMscore = scoreseq / int(length)
            
            dict_trainPPM.setdefault('pos', []).append((eachseq, PPMscore))
        dict_trainPPM['neg'] = OrderedDict()
        for eachseq in negtrainpep:
            scoreseq = 0
            for i in range(int(length)):
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                    scoreseq += 0
                else:
                    scoreseq += dict_AAPPM[eachseq[i]][i]
            PPMscore = scoreseq / int(length)
            
            dict_trainPPM['neg'].setdefault('0', []).append((eachseq, PPMscore))
        PPMmatrix = {}
        PPMmatrix['PPM'] = dict_AAPPM

        return dict_trainPPM, PPMmatrix 

    else:

        for _, matrixs in matrix.items():
            dict_AAPPM = matrixs['PPM']

        lengthmatrix = len(dict_AAPPM['A'])

        dict_predictnameseqPPM = OrderedDict()
        for eachname, seqs in predictpeptide.items():
            dict_predictnameseqPPM[eachname] = OrderedDict()
            # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                 
            #     tasks = [executor.submit(_ppm_paraProcessing, 0,seqs[0:math.ceil(len(seqs)/2)],lengthmatrix,dict_AAPPM),executor.submit(_ppm_paraProcessing,1,seqs[math.ceil(len(seqs)/2):],lengthmatrix,dict_AAPPM)]

           
            #     results = [(task.result()) for task in concurrent.futures.as_completed(tasks)]
            # if results[0][1]==0:
            #     ppmL1=results[0][0]
            #     ppmL2=results[1][0]
            # else:
            #     ppmL1=results[1][0]
            #     ppmL2=results[0][0]
            # # ppmL1=results[0]
            # # ppmL2=results[1]
            # dict_predictnameseqPPM[eachname].setdefault('pre', []).extend(ppmL1)
            # dict_predictnameseqPPM[eachname].setdefault('pre', []).extend(ppmL2)
            for eachseq in seqs:
                scoreseq = 0
                for i in range(lengthmatrix):
                    if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                        scoreseq += 0
                    else:
                        scoreseq += dict_AAPPM[eachseq[i]][i]
                PPMscore = scoreseq / lengthmatrix
                dict_predictnameseqPPM[eachname].setdefault('pre', []).append((eachseq, PPMscore))
                
        return dict_predictnameseqPPM

def _aaf_paraProcessing(id,seqs,lengthmatrix,dict_AAlenfre):
    _aaf_results=[]
    for eachseq in seqs:
        scoreseq = 0
        for i in range(lengthmatrix):
            if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                scoreseq += 0
            else:
                scoreseq += dict_AAlenfre[eachseq[i]][i]
        _aaf_results.append((eachseq, scoreseq))
    return _aaf_results,id
def AAF(predictpeptide, length, trainpeptide, matrix):
    if matrix == None:
        postrainpep, negtrainpep = [], []
        for key, value in trainpeptide.items():
            if value == '1':
                postrainpep.append(key)
            else:
                negtrainpep.append(key)

        dict_AAlenfre = {
            'A': [], 'R': [], 'N': [], 'D': [], 'C': [],
            'Q': [], 'E': [], 'G': [], 'H': [], 'I': [],
            'L': [], 'K': [], 'M': [], 'F': [], 'P': [],
            'S': [], 'T': [], 'W': [], 'Y': [], 'V': [], '-': []
        }
        for i in range(int(length)):
            dict_AAfre = {
                'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0,
                'Q': 0, 'E': 0, 'G': 0, 'H': 0, 'I': 0,
                'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0,
                'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
                '-': 0
            }
            for eachseq in trainpeptide:
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                    continue
                else:
                    dict_AAfre[eachseq[i]] += 1
            for eachAA, eachvallist in dict_AAlenfre.items():
                eachvallist.append((dict_AAfre[eachAA] / len(trainpeptide))  # each AA percentage
                                   /
                                   (max(dict_AAfre.values()) / len(trainpeptide)))  # AA percentage which is largest

        dict_trainAAF = OrderedDict()
        for eachseq in postrainpep:
            scoreseq = 0
            for i in range(int(length)):
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                    scoreseq += 0
                else:
                    scoreseq += dict_AAlenfre[eachseq[i]][i]
            
            dict_trainAAF.setdefault('pos', []).append((eachseq, scoreseq))
        dict_trainAAF['neg'] = OrderedDict()
        for eachseq in negtrainpep:
            scoreseq = 0
            for i in range(int(length)):
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                    scoreseq += 0
                else:
                    scoreseq += dict_AAlenfre[eachseq[i]][i]
            
            dict_trainAAF['neg'].setdefault('0', []).append((eachseq, scoreseq))
        AAFmatrix = {}
        AAFmatrix['AAF'] = dict_AAlenfre

        return dict_trainAAF, AAFmatrix

    else:

        for _, matrixs in matrix.items():
            dict_AAlenfre = matrixs['AAF']

        lengthmatrix = len(dict_AAlenfre['A'])

        dict_predictnameseqAAF = OrderedDict()
        for eachname, seqs in predictpeptide.items():
            dict_predictnameseqAAF[eachname] = OrderedDict()
            
            for eachseq in seqs:
                scoreseq = 0
                for i in range(lengthmatrix):
                    if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                        scoreseq += 0
                    else:
                        scoreseq += dict_AAlenfre[eachseq[i]][i]
                dict_predictnameseqAAF[eachname].setdefault('pre', []).append((eachseq, scoreseq))
                
            # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            #     tasks = [executor.submit(_aaf_paraProcessing, 0,seqs[0:math.ceil(len(seqs)/2)],lengthmatrix,dict_AAlenfre),executor.submit(_aaf_paraProcessing,1,seqs[math.ceil(len(seqs)/2):],lengthmatrix,dict_AAlenfre)]
            #     results = [(task.result()) for task in concurrent.futures.as_completed(tasks)]
            # if results[0][1]==0:
            #     aafL1=results[0][0]
            #     aafL2=results[1][0]
            # else:
            #     aafL1=results[1][0]
            #     aafL2=results[0][0]
            # dict_predictnameseqAAF[eachname].setdefault('pre', []).extend(aafL1)
            # dict_predictnameseqAAF[eachname].setdefault('pre', []).extend(aafL2)
        return dict_predictnameseqAAF

def _ic50_paraProcessing(id,seqs,lengthmatrix,dict_AAPSSM):
    _ic50_results=[]
    for eachseq in seqs:
        PSSMscore = 0
        for i in range(lengthmatrix):
            if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z', '-']:
                continue
            else:
                PSSMscore += dict_AAPSSM[eachseq[i]][i]
        bindingscore = PSSMscore / lengthmatrix
        IC50score = math.pow(50000, (0.8 - bindingscore) / 1.6)
        _ic50_results.append((eachseq, IC50score))
    return _ic50_results,id
def IC50(predictpeptide, length, trainpeptide, matrix):
    currentpath = os.getcwd()
    if matrix == None:
        with open('%s/source/AAUniProt.json' % currentpath, 'r') as f:
            dict_AAUniProtAF = json.load(f)

        postrainpep, negtrainpep = [], []
        for key, value in trainpeptide.items():
            if value == '1':
                postrainpep.append(key)
            else:
                negtrainpep.append(key)

        dict_AAlenfre = {
            'A': [], 'R': [], 'N': [], 'D': [], 'C': [],
            'Q': [], 'E': [], 'G': [], 'H': [], 'I': [],
            'L': [], 'K': [], 'M': [], 'F': [], 'P': [],
            'S': [], 'T': [], 'W': [], 'Y': [], 'V': [],
        }
        for i in range(int(length)):
            dict_AAfre = {
                'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0,
                'Q': 0, 'E': 0, 'G': 0, 'H': 0, 'I': 0,
                'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0,
                'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,

            }
            for eachseq in trainpeptide:
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z', '-']:
                    continue
                else:
                    dict_AAfre[eachseq[i]] += 1
            for eachAA, eachvallist in dict_AAlenfre.items():
                if dict_AAfre[eachAA] == 0:
                    eachvallist.append(1 / len(trainpeptide))  # set the minimum value
                else:
                    eachvallist.append(dict_AAfre[eachAA] / len(trainpeptide))

        dict_AAPSSM = {
            'A': [], 'R': [], 'N': [], 'D': [], 'C': [],
            'Q': [], 'E': [], 'G': [], 'H': [], 'I': [],
            'L': [], 'K': [], 'M': [], 'F': [], 'P': [],
            'S': [], 'T': [], 'W': [], 'Y': [], 'V': [],
        }
        for eachAA, eachAF in dict_AAlenfre.items():
            for eachposAF in eachAF:
                powerval = (eachposAF) / dict_AAUniProtAF[eachAA]
                posPSSM = math.log(powerval, 10)
                dict_AAPSSM[eachAA].append(posPSSM)

        dict_trainIC50 = OrderedDict()
        for eachseq in postrainpep:
            PSSMscore = 0
            for i in range(int(length)):
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z', '-']:
                    continue
                PSSMscore += dict_AAPSSM[eachseq[i]][i]
            bindingscore = PSSMscore / int(length)
            IC50score = math.pow(50000, (0.8 - bindingscore) / 1.6)
            dict_trainIC50.setdefault('pos', []).append((eachseq, IC50score))
        dict_trainIC50['neg'] = OrderedDict()
        for eachseq in negtrainpep:
            PSSMscore = 0
            for i in range(int(length)):
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z', '-']:
                    continue
                PSSMscore += dict_AAPSSM[eachseq[i]][i]
            bindingscore = PSSMscore / int(length)
            IC50score = math.pow(50000, (0.8 - bindingscore) / 1.6)
            dict_trainIC50['neg'].setdefault('0', []).append((eachseq, IC50score))
        IC50matrix = {}
        IC50matrix['IC50'] = dict_AAPSSM

        return dict_trainIC50, IC50matrix

    else:

        for _, matrixs in matrix.items():
            dict_AAPSSM = matrixs['IC50']

        lengthmatrix = len(dict_AAPSSM['A'])

        dict_predictnameseqIC50 = OrderedDict()
        for eachname, seqs in predictpeptide.items():
            dict_predictnameseqIC50[eachname] = OrderedDict()
            
            # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            #     tasks = [executor.submit(_ic50_paraProcessing, 0,seqs[0:math.ceil(len(seqs)/2)],lengthmatrix,dict_AAPSSM),executor.submit(_ic50_paraProcessing,1,seqs[math.ceil(len(seqs)/2):],lengthmatrix,dict_AAPSSM)]
            #     results = [(task.result()) for task in concurrent.futures.as_completed(tasks)]
            # if results[0][1]==0:
            #     ic50L1=results[0][0]
            #     ic50L2=results[1][0]
            # else:
            #     ic50L1=results[1][0]
            #     ic50L2=results[0][0]
            # dict_predictnameseqIC50[eachname].setdefault('pre', []).extend(ic50L1)
            # dict_predictnameseqIC50[eachname].setdefault('pre', []).extend(ic50L2)
            
            for eachseq in seqs:
                PSSMscore = 0
                for i in range(lengthmatrix):
                    if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z', '-']:
                        continue
                    else:
                        PSSMscore += dict_AAPSSM[eachseq[i]][i]
                bindingscore = PSSMscore / lengthmatrix
                IC50score = math.pow(50000, (0.8 - bindingscore) / 1.6)
                dict_predictnameseqIC50[eachname].setdefault('pre', []).append((eachseq, IC50score))
            
        return dict_predictnameseqIC50

def _pwm_paraProcessing(id,seqs,lengthmatrix,dict_AAPWM):
    _pwm_results=[]
    for eachseq in seqs:
        scoreseq = 0
        for i in range(lengthmatrix):
            if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                scoreseq += 0
            else:
                scoreseq += dict_AAPWM[eachseq[i]][i]
        PWMscore = scoreseq / lengthmatrix
        _pwm_results.append((eachseq, PWMscore))
    return _pwm_results,id
def PWM(predictpeptide, length, trainpeptide, matrix):
    if matrix == None:
        postrainpep, negtrainpep = [], []
        for key, value in trainpeptide.items():
            if value == '1':
                postrainpep.append(key)
            else:
                negtrainpep.append(key)

        dict_AAlenfre = {
            'A': [], 'R': [], 'N': [], 'D': [], 'C': [],
            'Q': [], 'E': [], 'G': [], 'H': [], 'I': [],
            'L': [], 'K': [], 'M': [], 'F': [], 'P': [],
            'S': [], 'T': [], 'W': [], 'Y': [], 'V': [], '-': [],
        }
        for i in range(int(length)):
            dict_AAfre = {
                'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0,
                'Q': 0, 'E': 0, 'G': 0, 'H': 0, 'I': 0,
                'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0,
                'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0, '-': 0,
            }
            for eachseq in trainpeptide:
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                    continue
                else:
                    dict_AAfre[eachseq[i]] += 1
            for eachAA, eachvallist in dict_AAlenfre.items():
                if dict_AAfre[eachAA] == 0:
                    eachvallist.append(1 / len(trainpeptide))  # set the minimum value
                else:
                    eachvallist.append(dict_AAfre[eachAA] / len(trainpeptide))

        dict_AAPWM = {
            'A': [], 'R': [], 'N': [], 'D': [], 'C': [],
            'Q': [], 'E': [], 'G': [], 'H': [], 'I': [],
            'L': [], 'K': [], 'M': [], 'F': [], 'P': [],
            'S': [], 'T': [], 'W': [], 'Y': [], 'V': [], '-': []
        }
        for eachAA, eachAF in dict_AAlenfre.items():
            for eachposAF in eachAF:
                powerval = (eachposAF) / 0.05
                posPWM = math.log(powerval, 2)
                dict_AAPWM[eachAA].append(posPWM)

        dict_trainPWM = OrderedDict()
        for eachseq in postrainpep:
            scoreseq = 0
            for i in range(int(length)):
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                    scoreseq += 0
                else:
                    scoreseq += dict_AAPWM[eachseq[i]][i]
            PWMscore = scoreseq / int(length)
            dict_trainPWM.setdefault('pos', []).append((eachseq, PWMscore))
        dict_trainPWM['neg'] = OrderedDict()
        for eachseq in negtrainpep:
            scoreseq = 0
            for i in range(int(length)):
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                    scoreseq += 0
                else:
                    scoreseq += dict_AAPWM[eachseq[i]][i]
            PWMscore = scoreseq / int(length)
            dict_trainPWM['neg'].setdefault('0', []).append((eachseq, PWMscore))

        PWMmatrix = {}
        PWMmatrix['PWM'] = dict_AAPWM

        return dict_trainPWM, PWMmatrix

    else:

        for _, matrixs in matrix.items():
            dict_AAPWM = matrixs['PWM']

        lengthmatrix = len(dict_AAPWM['A'])

        dict_predictnameseqPWM = OrderedDict()
        for eachname, seqs in predictpeptide.items():
            dict_predictnameseqPWM[eachname] = OrderedDict()
            
            for eachseq in seqs:
                scoreseq = 0
                for i in range(lengthmatrix):
                    if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                        scoreseq += 0
                    else:
                        scoreseq += dict_AAPWM[eachseq[i]][i]
                PWMscore = scoreseq / lengthmatrix
                dict_predictnameseqPWM[eachname].setdefault('pre', []).append((eachseq, PWMscore))
                
            # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            #     tasks = [executor.submit(_pwm_paraProcessing, 0,seqs[0:math.ceil(len(seqs)/2)],lengthmatrix,dict_AAPWM),executor.submit(_pwm_paraProcessing,1,seqs[math.ceil(len(seqs)/2):],lengthmatrix,dict_AAPWM)]
            #     results = [(task.result()) for task in concurrent.futures.as_completed(tasks)]
            # if results[0][1]==0:
            #     pwmL1=results[0][0]
            #     pwmL2=results[1][0]
            # else:
            #     pwmL1=results[1][0]
            #     pwmL2=results[0][0]

            # dict_predictnameseqPWM[eachname].setdefault('pre', []).extend(pwmL1)
            # dict_predictnameseqPWM[eachname].setdefault('pre', []).extend(pwmL2)
            
        return dict_predictnameseqPWM

def _bsi_paraProcessing(id,seqs,AA,dict_trainindexscore,blosum62):
    _bsi_results=[]
    for eachseq in seqs:
        Btestknownlist = []
        indexstr = ''
        for eachAA in eachseq:
            indexstr += (str(AA.index(eachAA)) + ' ')
        indexstr = indexstr.strip()
        for AAtrainindex in dict_trainindexscore.keys():
            if indexstr != AAtrainindex:
                AAtrainindexlist = AAtrainindex.split(' ')
                sumpredSb = 0
                for resipred in enumerate(eachseq):
                    sumpredSb += blosum62[resipred[1]][int(AAtrainindexlist[resipred[0]])]
                Btestknownlist.append(sumpredSb / int(dict_trainindexscore[AAtrainindex]))
            else:
                continue
        BSI62score = max(Btestknownlist)
        _bsi_results.append((eachseq, BSI62score))
    return _bsi_results,id
def BSI(predictpeptide, trainpeptide, blosum62, matrix, blosumName):
    AA = 'ARNDCQEGHILKMFPSTWYV-'

    if matrix == None:
        postrainpep, negtrainpep = [], []
        for key, value in trainpeptide.items():
            if value == '1':
                postrainpep.append(key)
            else:
                negtrainpep.append(key)

        dict_trainindexscore = OrderedDict()
        for eachpep in trainpeptide:
            indexlist = ''
            sumtrainSb = 0
            for eachAA in eachpep:
                indexlist += (str(AA.index(eachAA)) + ' ')
            indexlist = indexlist.strip()
            for resitrain in enumerate(eachpep):
                sumtrainSb += blosum62[resitrain[1]][AA.index(resitrain[1])]
            if sumtrainSb == 0:
                sumtrainSb = 1
            dict_trainindexscore[indexlist] = sumtrainSb

        dict_trainBSI62 = OrderedDict()
        for eachseq in postrainpep:
            Btestknownlist = []
            indexstr = ''
            for eachAA in eachseq:
                indexstr += (str(AA.index(eachAA)) + ' ')
            indexstr = indexstr.strip()
            for AAtrainindex in dict_trainindexscore.keys():
                if indexstr != AAtrainindex:
                    AAtrainindexlist = AAtrainindex.split(' ')
                    sumpredSb = 0
                    for resipred in enumerate(eachseq):
                        sumpredSb += blosum62[resipred[1]][int(AAtrainindexlist[resipred[0]])]
                    Btestknownlist.append(sumpredSb / int(dict_trainindexscore[AAtrainindex]))
                else:
                    continue
            BSI62score = max(Btestknownlist)
            dict_trainBSI62.setdefault('pos', []).append((eachseq, BSI62score))

        dict_trainBSI62['neg'] = OrderedDict()
        for eachseq in negtrainpep:
            Btestknownlist = []
            indexstr = ''
            for eachAA in eachseq:
                indexstr += (str(AA.index(eachAA)) + ' ')
            indexstr = indexstr.strip()
            for AAtrainindex in dict_trainindexscore.keys():
                if indexstr != AAtrainindex:
                    AAtrainindexlist = AAtrainindex.split(' ')
                    sumpredSb = 0
                    for resipred in enumerate(eachseq):
                        sumpredSb += blosum62[resipred[1]][int(AAtrainindexlist[resipred[0]])]
                    Btestknownlist.append(sumpredSb / int(dict_trainindexscore[AAtrainindex]))
                else:
                    continue
            BSI62score = max(Btestknownlist)
            dict_trainBSI62['neg'].setdefault('0', []).append((eachseq, BSI62score))
        BSI62matrix = {}
        BSI62matrix[blosumName] = dict_trainindexscore

        
        return dict_trainBSI62, BSI62matrix

    else:

        for _, matrixs in matrix.items():
            dict_trainindexscore = matrixs[blosumName]

        dict_predictnameseqBSI62 = OrderedDict()
        for eachname, seqs in predictpeptide.items():
            dict_predictnameseqBSI62[eachname] = OrderedDict()
            
            for eachseq in seqs:
                Btestknownlist = []
                indexstr = ''
                for eachAA in eachseq:
                    indexstr += (str(AA.index(eachAA)) + ' ')
                indexstr = indexstr.strip()
                for AAtrainindex in dict_trainindexscore.keys():
                    if indexstr != AAtrainindex:
                        AAtrainindexlist = AAtrainindex.split(' ')
                        sumpredSb = 0
                        for resipred in enumerate(eachseq):
                            sumpredSb += blosum62[resipred[1]][int(AAtrainindexlist[resipred[0]])]
                        Btestknownlist.append(sumpredSb / int(dict_trainindexscore[AAtrainindex]))
                    else:
                        continue
                BSI62score = max(Btestknownlist)
                dict_predictnameseqBSI62[eachname].setdefault('pre', []).append((eachseq, BSI62score))
                
            # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            #     tasks = [executor.submit(_bsi_paraProcessing, 0,seqs[0:math.ceil(len(seqs)/2)],AA,dict_trainindexscore,blosum62),executor.submit(_bsi_paraProcessing,1,seqs[math.ceil(len(seqs)/2):],AA,dict_trainindexscore,blosum62)]
            #     results = [(task.result()) for task in concurrent.futures.as_completed(tasks)]
            # if results[0][1]==0:
            #     bsiL1=results[0][0]
            #     bsiL2=results[1][0]
            # else:
            #     bsiL1=results[1][0]
            #     bsiL2=results[0][0]
            # dict_predictnameseqBSI62[eachname].setdefault('pre', []).extend(bsiL1)
            # dict_predictnameseqBSI62[eachname].setdefault('pre', []).extend(bsiL2)
            
        return dict_predictnameseqBSI62

def _pam_paraProcessing(id,seqs,AA,dict_trainindexscore,pam):
    _pam_results=[]
    for eachseq in seqs:
        Btestknownlist = []
        indexstr = ''
        for eachAA in eachseq:
            indexstr += (str(AA.index(eachAA)) + ' ')
        indexstr = indexstr.strip()
        for AAtrainindex in dict_trainindexscore.keys():
            if indexstr != AAtrainindex:
                AAtrainindexlist = AAtrainindex.split(' ')
                sumpredSb = 0
                for resipred in enumerate(eachseq):
                    sumpredSb += pam[resipred[1]][int(AAtrainindexlist[resipred[0]])]
                Btestknownlist.append(sumpredSb / int(dict_trainindexscore[AAtrainindex]))
            else:
                continue
        pamscore = max(Btestknownlist)
        _pam_results.append((eachseq, pamscore))
    return _pam_results,id
def PAM(predictpeptide, trainpeptide, pam, matrix, pamName):
    AA = 'ARNDCQEGHILKMFPSTWYV-'

    if matrix == None:
        postrainpep, negtrainpep = [], []
        for key, value in trainpeptide.items():
            if value == '1':
                postrainpep.append(key)
            else:
                negtrainpep.append(key)

        dict_trainindexscore = OrderedDict()
        for eachpep in trainpeptide:
            indexlist = ''
            sumtrainSb = 0
            for eachAA in eachpep:
                indexlist += (str(AA.index(eachAA)) + ' ')
            indexlist = indexlist.strip()
            for resitrain in enumerate(eachpep):
                sumtrainSb += pam[resitrain[1]][AA.index(resitrain[1])]
            if sumtrainSb == 0:
                sumtrainSb = 1
            dict_trainindexscore[indexlist] = sumtrainSb

        dict_trainPAM = OrderedDict()
        for eachseq in postrainpep:
            Btestknownlist = []
            indexstr = ''
            for eachAA in eachseq:
                indexstr += (str(AA.index(eachAA)) + ' ')
            indexstr = indexstr.strip()
            for AAtrainindex in dict_trainindexscore.keys():
                if indexstr != AAtrainindex:
                    AAtrainindexlist = AAtrainindex.split(' ')
                    sumpredSb = 0
                    for resipred in enumerate(eachseq):
                        sumpredSb += pam[resipred[1]][int(AAtrainindexlist[resipred[0]])]
                    Btestknownlist.append(sumpredSb / int(dict_trainindexscore[AAtrainindex]))
                else:
                    continue
            pamscore = max(Btestknownlist)
            dict_trainPAM.setdefault('pos', []).append((eachseq, pamscore))

        dict_trainPAM['neg'] = OrderedDict()
        for eachseq in negtrainpep:
            Btestknownlist = []
            indexstr = ''
            for eachAA in eachseq:
                indexstr += (str(AA.index(eachAA)) + ' ')
            indexstr = indexstr.strip()
            for AAtrainindex in dict_trainindexscore.keys():
                if indexstr != AAtrainindex:
                    AAtrainindexlist = AAtrainindex.split(' ')
                    sumpredSb = 0
                    for resipred in enumerate(eachseq):
                        sumpredSb += pam[resipred[1]][int(AAtrainindexlist[resipred[0]])]
                    Btestknownlist.append(sumpredSb / int(dict_trainindexscore[AAtrainindex]))
                else:
                    continue
            pamscore = max(Btestknownlist)
            dict_trainPAM['neg'].setdefault('0', []).append((eachseq, pamscore))
        PAMmatrix = {}
        PAMmatrix[pamName] = dict_trainindexscore

        return dict_trainPAM, PAMmatrix

    else:

        for _, matrixs in matrix.items():
            dict_trainindexscore = matrixs[pamName]

        dict_predictnameseqPAM = OrderedDict()
        for eachname, seqs in predictpeptide.items():
            dict_predictnameseqPAM[eachname] = OrderedDict()
            
            # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            #     tasks = [executor.submit(_pam_paraProcessing, 0,seqs[0:math.ceil(len(seqs)/2)],AA,dict_trainindexscore,pam),executor.submit(_pam_paraProcessing,1,seqs[math.ceil(len(seqs)/2):],AA,dict_trainindexscore,pam)]
            #     results = [(task.result()) for task in concurrent.futures.as_completed(tasks)]
            # if results[0][1]==0:
            #     pamL1=results[0][0]
            #     pamL2=results[1][0]
            # else:
            #     pamL1=results[1][0]
            #     pamL2=results[0][0]    
            # dict_predictnameseqPAM[eachname].setdefault('pre', []).extend(pamL1)
            # dict_predictnameseqPAM[eachname].setdefault('pre', []).extend(pamL2)
            
            for eachseq in seqs:
                Btestknownlist = []
                indexstr = ''
                for eachAA in eachseq:
                    indexstr += (str(AA.index(eachAA)) + ' ')
                indexstr = indexstr.strip()
                for AAtrainindex in dict_trainindexscore.keys():
                    if indexstr != AAtrainindex:
                        AAtrainindexlist = AAtrainindex.split(' ')
                        sumpredSb = 0
                        for resipred in enumerate(eachseq):
                            sumpredSb += pam[resipred[1]][int(AAtrainindexlist[resipred[0]])]
                        Btestknownlist.append(sumpredSb / int(dict_trainindexscore[AAtrainindex]))
                    else:
                        continue
                pamscore = max(Btestknownlist)
                dict_predictnameseqPAM[eachname].setdefault('pre', []).append((eachseq, pamscore))
            
        return dict_predictnameseqPAM

def _nns_paraProcessing(id,seqs,AA,list_trainindex,blosum62):
    nns_result=[]
    for eachpeptide in seqs:
        liAApred = []
        indexstr = ''
        for eachAA in eachpeptide:
            indexstr += (str(AA.index(eachAA)) + ' ')
        indexstr = indexstr.strip()
        for trainindexstr in list_trainindex:
            if indexstr != trainindexstr:
                trainindexstr = trainindexstr.split(' ')
                similarityseq = []
                for resi in enumerate(eachpeptide):
                    similarityAB = blosum62[resi[1]][int(trainindexstr[resi[0]])]
                    similarityseq.append(similarityAB)
                liAApred.append(similarityseq)
        dfAApred = pd.DataFrame(liAApred)
        seAApredsum = dfAApred.apply(sum)
        seAApredavg = seAApredsum.div(len(list_trainindex))
        seAApredscore = seAApredavg.sum()
        nns_result.append((eachpeptide, seAApredscore))
    return nns_result,id
def NNS(predictpeptide, trainpeptide, matrix):
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],  # A
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],  # R
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],  # N
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],  # D
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],  # C
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0],  # Q
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 0],  # E
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],  # G
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],  # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],  # L
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],  # K
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],  # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],  # P
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0],  # S
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],  # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, 0],  # Y
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, 0],  # V
        '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    }
    AA = 'ARNDCQEGHILKMFPSTWYV-'

    if matrix == None:
        postrainpep, negtrainpep = [], []
        for key, value in trainpeptide.items():
            if value == '1':
                postrainpep.append(key)
            else:
                negtrainpep.append(key)

        list_trainindex = []
        for eachpep in postrainpep:
            indexstr = ''
            for eachAA in eachpep:
                indexstr += (str(AA.index(eachAA)) + ' ')
            indexstr = indexstr.strip()
            list_trainindex.append(indexstr)

        dict_trainNNS = OrderedDict()
        for eachpeptide in postrainpep:
            liAApred = []
            indexstr = ''
            for eachAA in eachpeptide:
                indexstr += (str(AA.index(eachAA)) + ' ')
            indexstr = indexstr.strip()
            for trainindexstr in list_trainindex:
                if indexstr != trainindexstr:
                    trainindexstr = trainindexstr.split(' ')
                    similarityseq = []
                    for resi in enumerate(eachpeptide):
                        similarityAB = blosum62[resi[1]][int(trainindexstr[resi[0]])]
                        similarityseq.append(similarityAB)
                    liAApred.append(similarityseq)
            dfAApred = pd.DataFrame(liAApred)
            seAApredsum = dfAApred.apply(sum)
            seAApredavg = seAApredsum.div(len(list_trainindex))
            seAApredscore = seAApredavg.sum()
            dict_trainNNS.setdefault('pos', []).append((eachpeptide, seAApredscore))
        dict_trainNNS['neg'] = OrderedDict()
        for eachpeptide in negtrainpep:
            liAApred = []
            indexstr = ''
            for eachAA in eachpeptide:
                indexstr += (str(AA.index(eachAA)) + ' ')
            indexstr = indexstr.strip()
            for trainindexstr in list_trainindex:
                if indexstr != trainindexstr:
                    trainindexstr = trainindexstr.split(' ')
                    similarityseq = []
                    for resi in enumerate(eachpeptide):
                        similarityAB = blosum62[resi[1]][int(trainindexstr[resi[0]])]
                        similarityseq.append(similarityAB)
                    liAApred.append(similarityseq)
            dfAApred = pd.DataFrame(liAApred)
            seAApredsum = dfAApred.apply(sum)
            seAApredavg = seAApredsum.div(len(list_trainindex))
            seAApredscore = seAApredavg.sum()
            dict_trainNNS['neg'].setdefault('0', []).append((eachpeptide, seAApredscore))
        NNSmatrix = {}
        NNSmatrix['NNS'] = list_trainindex

        return dict_trainNNS, NNSmatrix

    else:

        for _, matrixs in matrix.items():
            list_trainindex = matrixs['NNS']

        dict_predictnameseqNNS = OrderedDict()
        for eachname, seqs in predictpeptide.items():
            dict_predictnameseqNNS[eachname] = OrderedDict()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                tasks = [executor.submit(_nns_paraProcessing, 0,seqs[0:math.ceil(len(seqs)/2)],AA,list_trainindex,blosum62),executor.submit(_nns_paraProcessing,1,seqs[math.ceil(len(seqs)/2):],AA,list_trainindex,blosum62)]
                results = [(task.result()) for task in concurrent.futures.as_completed(tasks)]
            if results[0][1]==0:
                nnsL1=results[0][0]
                nnsL2=results[1][0]
            else:
                nnsL1=results[1][0]
                nnsL2=results[0][0]
            dict_predictnameseqNNS[eachname].setdefault('pre', []).extend(nnsL1)
            dict_predictnameseqNNS[eachname].setdefault('pre', []).extend(nnsL2)
            
            # for eachpeptide in seqs:
            #     liAApred = []
            #     indexstr = ''
            #     for eachAA in eachpeptide:
            #         indexstr += (str(AA.index(eachAA)) + ' ')
            #     indexstr = indexstr.strip()
            #     for trainindexstr in list_trainindex:
            #         if indexstr != trainindexstr:
            #             trainindexstr = trainindexstr.split(' ')
            #             similarityseq = []
            #             for resi in enumerate(eachpeptide):
            #                 similarityAB = blosum62[resi[1]][int(trainindexstr[resi[0]])]
            #                 similarityseq.append(similarityAB)
            #             liAApred.append(similarityseq)
            #     dfAApred = pd.DataFrame(liAApred)
            #     seAApredsum = dfAApred.apply(sum)
            #     seAApredavg = seAApredsum.div(len(list_trainindex))
            #     seAApredscore = seAApredavg.sum()
            #     # dict_predictnameseqNNS[eachname][eachpeptide] = seAApredscore
            #     dict_predictnameseqNNS[eachname].setdefault('pre', []).append((eachpeptide, seAApredscore))
            
        return dict_predictnameseqNNS

def _knn_paraProcessing(id,seqs,AA,dict_ktrainindexlabel,blosum62,NUM):
    dict_pepposkKNN = OrderedDict()
    num=NUM
    for eachpeptide in seqs:
        dickey = str(eachpeptide) + str(num)  
        num += 1
        dict_pepposkKNN[dickey] = OrderedDict()
        eachpeptidedistance = []
        indexstr = ''
        for eachAA in eachpeptide:
            indexstr += (str(AA.index(eachAA)) + ' ')
        indexstr = indexstr.strip()
        for k, trainindex in dict_ktrainindexlabel.items():
            for trainindexstr, label in trainindex.items():
                if indexstr != trainindexstr:
                    lengthmatrix = len(trainindexstr)
                    trainindexstr = trainindexstr.split(' ')
                    similarityseq = 0
                    for resi in enumerate(eachpeptide):
                        similarityseq += (blosum62[resi[1]][int(trainindexstr[resi[0]])] + 4) / 15
                    distanceseq = 1 - similarityseq / lengthmatrix
                    eachpeptidedistance.append((distanceseq, label))
        sorted_by_score = sorted(eachpeptidedistance, key=lambda t: t[0])
        k_value = float((list(dict_ktrainindexlabel.keys()))[0])
        select_by_k = sorted_by_score[0:int(len(sorted_by_score) * k_value)]
        numposlable = 0
        for eachselect in select_by_k:
            if eachselect[1] == '1':
                numposlable += 1
        try:
            perpos_by_k = numposlable / len(select_by_k)
        except:
            perpos_by_k = 0

        dict_pepposkKNN[dickey] = (k_value, perpos_by_k)
    return dict_pepposkKNN,id
def KNN(predictpeptide, length, trainpeptide, matrix):
    kvalues = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15,0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]

    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],  # A
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],  # R
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],  # N
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],  # D
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],  # C
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0],  # Q
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 0],  # E
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],  # G
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],  # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],  # L
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],  # K
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],  # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],  # P
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0],  # S
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],  # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, 0],  # Y
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, 0],  # V
        '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    }
    AA = 'ARNDCQEGHILKMFPSTWYV-'

    if matrix == None:
        postrainpep, negtrainpep = [], []
        for key, value in trainpeptide.items():
            if value == '1':
                postrainpep.append(key)
            else:
                negtrainpep.append(key)

        dict_trainindexlabel = OrderedDict()
        for eachpep, label in trainpeptide.items():
            indexstr = ''
            for eachAA in eachpep:
                indexstr += (str(AA.index(eachAA)) + ' ')
            indexstr = indexstr.strip()
            dict_trainindexlabel[indexstr] = label

        dict_trainKNN = OrderedDict()
        dict_pepposkKNN, dict_kposscore = OrderedDict(), OrderedDict()
        dict_pepnegkKNN, dict_knegscore = OrderedDict(), OrderedDict()
        k_diff = []
        for eachpeptide in postrainpep:
            dict_pepposkKNN[eachpeptide] = OrderedDict()
            eachpeptidedistance = []
            indexstr = ''
            for eachAA in eachpeptide:
                indexstr += (str(AA.index(eachAA)) + ' ')
            indexstr = indexstr.strip()
            for trainindexstr, label in dict_trainindexlabel.items():
                if indexstr != trainindexstr:
                    trainindexstr = trainindexstr.split(' ')
                    similarityseq = 0
                    for resi in enumerate(eachpeptide):
                        similarityseq += (blosum62[resi[1]][int(trainindexstr[resi[0]])] + 4) / 15
                    distanceseq = 1 - similarityseq / int(length)
                    eachpeptidedistance.append((distanceseq, label))
            sorted_by_score = sorted(eachpeptidedistance, key=lambda t: t[0])
            for eachk in kvalues:
                select_by_k = sorted_by_score[0:int(len(sorted_by_score) * eachk)]
                numposlable = 0
                for eachselect in select_by_k:
                    if eachselect[1] == '1':
                        numposlable += 1
                try:
                    perpos_by_k = numposlable / len(select_by_k)
                except:
                    perpos_by_k = 0
                dict_kposscore.setdefault(eachk, []).append(perpos_by_k)
                dict_pepposkKNN[eachpeptide][eachk] = perpos_by_k

        for eachpeptide in negtrainpep:
            dict_pepnegkKNN[eachpeptide] = OrderedDict()
            eachpeptidedistance = []
            indexstr = ''
            for eachAA in eachpeptide:
                indexstr += (str(AA.index(eachAA)) + ' ')
            indexstr = indexstr.strip()
            for trainindexstr, label in dict_trainindexlabel.items():
                if indexstr != trainindexstr:
                    trainindexstr = trainindexstr.split(' ')
                    similarityseq = 0
                    for resi in enumerate(eachpeptide):
                        similarityseq += (blosum62[resi[1]][int(trainindexstr[resi[0]])] + 4) / 15
                    distanceseq = 1 - similarityseq / int(length)
                    eachpeptidedistance.append((distanceseq, label))
            sorted_by_score = sorted(eachpeptidedistance, key=lambda t: t[0])
            for eachk in kvalues:
                select_by_k = sorted_by_score[0:int(len(sorted_by_score) * eachk)]
                numposlable = 0
                for eachselect in select_by_k:
                    if eachselect[1] == '1':
                        numposlable += 1
                try:
                    perpos_by_k = numposlable / len(select_by_k)
                except:
                    perpos_by_k = 0
                dict_knegscore.setdefault(eachk, []).append(perpos_by_k)
                dict_pepnegkKNN[eachpeptide][eachk] = perpos_by_k
                
        for eachkvalue in kvalues:
            numnegmore50, numposmore50 = 0, 0
            for eachkscore in dict_knegscore[eachkvalue]:
                if eachkscore > 0.5:
                    numnegmore50 += 1
            for eachkscore in dict_kposscore[eachkvalue]:
                if eachkscore > 0.5:
                    numposmore50 += 1
            k_diff.append((eachkvalue, (numposmore50 - numnegmore50)))
        k_diff_sort_by_numdiff = sorted(k_diff, key=lambda t: t[1], reverse=True)
        k_to_choose = k_diff_sort_by_numdiff[0][0]
        dict_trainKNN['neg'] = OrderedDict()
        for eachpep, kscore in dict_pepnegkKNN.items():
            dict_trainKNN['neg'].setdefault('0', []).append((eachpep, kscore[k_to_choose]))
        for eachpep, kscore in dict_pepposkKNN.items():
            dict_trainKNN.setdefault('pos', []).append((eachpep, kscore[k_to_choose]))
        KNNmatrix = {}
        KNNmatrix['KNN'] = {}
        KNNmatrix['KNN'][k_to_choose] = dict_trainindexlabel


        return dict_trainKNN, KNNmatrix

    else:

        for _, matrixs in matrix.items():
            dict_ktrainindexlabel = matrixs['KNN']

        dict_predictnameseqKNN = OrderedDict()
        for eachname, seqs in predictpeptide.items():
            dict_predictnameseqKNN[eachname] = OrderedDict()
            
            # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            #     tasks = [executor.submit(_knn_paraProcessing, 0,seqs[0:math.ceil(len(seqs)/2)],AA,dict_ktrainindexlabel,blosum62,0),executor.submit(_knn_paraProcessing,1,seqs[math.ceil(len(seqs)/2):],AA,dict_ktrainindexlabel,blosum62,math.ceil(len(seqs)/2))]
            #     results = [(task.result()) for task in concurrent.futures.as_completed(tasks)]
            # if results[0][1]==0:
            #     dict_pepposkKNN1=results[0][0]
            #     dict_pepposkKNN2=results[1][0]
            # else:
            #     dict_pepposkKNN1=results[1][0]
            #     dict_pepposkKNN2=results[0][0]
            # # dict_pepposkKNN1=results[0]
            # # dict_pepposkKNN2=results[1]
            # for eachpep, kscore in dict_pepposkKNN1.items():
            #     # print(eachpep, kscore)
            #     # dict_predictnameseqKNN[eachname][eachpep] = kscore[1]
            #     dict_predictnameseqKNN[eachname].setdefault('pre', []).append((eachpep, kscore[1]))
            # for eachpep1, kscore1 in dict_pepposkKNN2.items():
            #     dict_predictnameseqKNN[eachname].setdefault('pre', []).append((eachpep1, kscore1[1]))

            dict_pepposkKNN = OrderedDict()
            num = 0
            for eachpeptide in seqs:
                dickey = str(eachpeptide) + str(num)  
                num += 1
                dict_pepposkKNN[dickey] = OrderedDict()
                eachpeptidedistance = []
                indexstr = ''
                for eachAA in eachpeptide:
                    indexstr += (str(AA.index(eachAA)) + ' ')
                    
                indexstr = indexstr.strip()
                for k, trainindex in dict_ktrainindexlabel.items():
                    for trainindexstr, label in trainindex.items():
                        if indexstr != trainindexstr:
                            lengthmatrix = int(length)
                            trainindexstr = trainindexstr.split(' ')
                            similarityseq = 0
                            for resi in enumerate(eachpeptide):
                                similarityseq += (blosum62[resi[1]][int(trainindexstr[resi[0]])] + 4) / 15
                            distanceseq = 1 - similarityseq / lengthmatrix
                            eachpeptidedistance.append((distanceseq, label))
                            
                sorted_by_score = sorted(eachpeptidedistance, key=lambda t: t[0])
                k_value = float((list(dict_ktrainindexlabel.keys()))[0])
                
                select_by_k = sorted_by_score[0:int(len(sorted_by_score) * k_value)]
                numposlable = 0
                for eachselect in select_by_k:
                    if eachselect[1] == '1':
                        numposlable += 1
                try:
                    perpos_by_k = numposlable / len(select_by_k)
                except:
                    perpos_by_k = 0
                dict_pepposkKNN[dickey] = (k_value, perpos_by_k)

            for eachpep, kscore in dict_pepposkKNN.items():
                dict_predictnameseqKNN[eachname].setdefault('pre', []).append((eachpep, kscore[1]))
                
        return dict_predictnameseqKNN

def _wls_paraProcessing(id,seqs,lengthmatrix,dict_WLS):
    _wls_results=[]
    for eachseq in seqs:
        scoreseq = 0
        for i in range(lengthmatrix):
            if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                scoreseq += 0
            else:
                scoreseq += float(dict_WLS[eachseq[i]][i])
        _wls_results.append((eachseq, scoreseq))
    return _wls_results,id
def WLS(predictpeptide, length, trainpeptide, matrix):
    currentpath = os.getcwd()
    if matrix == None:
        while True:
            templatefolder1 = str(randint(0, 10000))
            if os.path.exists('%s/%s' % (currentpath, templatefolder1)):
                continue
            else:
                break
        os.system('mkdir %s/%s' % (currentpath, templatefolder1))
        postrainpep, negtrainpep = [], []
        for key, value in trainpeptide.items():
            if value == '1':
                postrainpep.append(key)
            else:
                negtrainpep.append(key)

        with open('%s/%s/trainpeptide.fasta' % (currentpath, templatefolder1), 'w') as f:
            rownum = 1
            for eachtrainpeptide in trainpeptide:
                f.write('>' + str(rownum) + '\n')
                f.write(eachtrainpeptide + '\n')
                rownum += 1

        # subprocess.call(['%s/source/weblogo/seqlogo' % currentpath, ('-f %s/%s/trainpeptide.fasta' % (currentpath, templatefolder)), ('-o %s/%s/trainpeptide' % (currentpath, templatefolder))])
        subprocess.run('weblogo -f %s/%s/trainpeptide.fasta -o %s/%s/trainpeptide.eps' % (
        currentpath, templatefolder1, currentpath, templatefolder1), shell=True)  # ,shell=True
        # subprocess.run('/data/software/miniconda3/envs/prosperousplus/bin/weblogo -f %s/%s/trainpeptide.fasta -o %s/%s/trainpeptide.eps' % (
            # currentpath, templatefolder1, currentpath, templatefolder1), shell=True)
        #         subprocess.run('E:\Anaconda3\envs\Monash\Scripts\weblogo.exe -f %s/%s/trainpeptide.fasta -o %s/%s/trainpeptide.eps' % (currentpath, templatefolder1,currentpath, templatefolder1),shell=True)#,shell=True
        # subprocess.run('weblogo -f %s/trainpeptide.fasta -o %s/trainpeptide.eps' % (templatefolder,templatefolder),shell=True)
        dict_WLS = {
            'A': [], 'R': [], 'N': [], 'D': [], 'C': [],
            'Q': [], 'E': [], 'G': [], 'H': [], 'I': [],
            'L': [], 'K': [], 'M': [], 'F': [], 'P': [],
            'S': [], 'T': [], 'W': [], 'Y': [], 'V': [], '-': [],
        }
        with open('%s/%s/trainpeptide.eps' % (currentpath, templatefolder1), 'r') as f:
            valuelist = []
            startpoint = 0
            addempty = 1
            for eachline in f:
                # if ') startstack' in eachline:
                #     startpoint += 1
                # elif startpoint == 1 and 'endstack' not in eachline:
                #     valuelist.append(eachline.split(' ')[1:3])
                # elif startpoint == 1 and 'endstack' in eachline:
                #     for eachvalue in valuelist:
                #         dict_WLS[eachvalue[1][1]].append(eachvalue[0])
                #     for eachAAtemp, eachwistemp in dict_WLS.items():
                #         if len(eachwistemp) != addempty:
                #             eachwistemp.append('0')
                #     startpoint -= 1
                #     valuelist = []
                #     addempty += 1
                # else:
                #     continue
                if ') StartStack' in eachline and '%' not in eachline:
                    startpoint += 1
                elif startpoint == 1 and 'EndStack' not in eachline:
                    valuelist.append(eachline.split(' ')[1:])
                elif startpoint == 1 and 'EndStack' in eachline:
                    for eachvalue in valuelist:
                        if eachvalue[-1] == 'ShowSymbol\n':
                            dict_WLS[eachvalue[7][1]].append(eachvalue[1])
                    for eachAAtemp, eachwistemp in dict_WLS.items():
                        if len(eachwistemp) != addempty:
                            eachwistemp.append('0')
                    startpoint -= 1
                    valuelist = []
                    addempty += 1
                else:
                    continue

        dict_trainWLS = OrderedDict()
        for eachseq in postrainpep:
            scoreseq = 0
            for i in range(int(length)):
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                    scoreseq += 0
                else:
                    scoreseq += float(dict_WLS[eachseq[i]][i])
            dict_trainWLS.setdefault('pos', []).append((eachseq, scoreseq))
        dict_trainWLS['neg'] = OrderedDict()
        for eachseq in negtrainpep:
            scoreseq = 0
            for i in range(int(length)):
                if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
                    scoreseq += 0
                else:
                    scoreseq += float(dict_WLS[eachseq[i]][i])
            # dict_trainWLS[eachseq] = (scoreseq, -1)
            dict_trainWLS['neg'].setdefault('0', []).append((eachseq, scoreseq))
        WLSmatrix = {}
        WLSmatrix['WLS'] = dict_WLS
        os.system('rm -r %s/%s' % (currentpath, templatefolder1))

        return dict_trainWLS, WLSmatrix

    else:
        
        for _, matrixs in matrix.items():
            dict_WLS = matrixs['WLS']

        lengthmatrix = len(dict_WLS['A'])

        dict_predictnameseqWLS = OrderedDict()
        for eachname, seqs in predictpeptide.items():
            dict_predictnameseqWLS[eachname] = OrderedDict()

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

                tasks = [executor.submit(_wls_paraProcessing, 0,seqs[0:math.ceil(len(seqs)/2)],lengthmatrix,dict_WLS),executor.submit(_wls_paraProcessing,1,seqs[math.ceil(len(seqs)/2):],lengthmatrix,dict_WLS)]
                results = [(task.result()) for task in concurrent.futures.as_completed(tasks)]
            if results[0][1]==0:
                wlsL1=results[0][0]
                wlsL2=results[1][0]
                wlsL1.extend(wlsL2)
            else:
                wlsL1=results[1][0]
                wlsL2=results[0][0]
                wlsL1.extend(wlsL2)
            
            dict_predictnameseqWLS[eachname].setdefault('pre', []).extend(wlsL1)

            # for eachseq in seqs:
            #     scoreseq = 0
            #     for i in range(lengthmatrix):
            #         if eachseq[i] in ['B', 'J', 'O', 'U', 'X', 'Z']:
            #             scoreseq += 0
            #         else:
            #             scoreseq += float(dict_WLS[eachseq[i]][i])
            #     # dict_predictnameseqWLS[eachname][eachseq] = scoreseq
            #     dict_predictnameseqWLS[eachname].setdefault('pre', []).append((eachseq, scoreseq))
            
        return dict_predictnameseqWLS
