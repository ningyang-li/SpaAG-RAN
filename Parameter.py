# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 10:12:35 2022

@author: fes_map
"""

import argparse


parser = argparse.ArgumentParser(description="Networks")

parser.add_argument("--env", type=int, default=0, help="type of operation system (0: windows, 1: ubuntu)")
parser.add_argument("--ds_name", type=str, default="Indian_Pines", help="name of data set")

parser.add_argument("--train_ratio", type=dict, default={"Indian_Pines": 0.15, "PaviaU": 0.05, "Botswana": 0.15}, help="training sample proportion")
parser.add_argument("--val_ratio", type=dict, default={"Indian_Pines": 0.05, "PaviaU": 0.05, "Botswana": 0.05}, help="training sample proportion")
parser.add_argument("--width", type=dict, default={"Indian_Pines": 11, "PaviaU": 5, "Botswana": 7}, help="width of sample")
parser.add_argument("--n_category", type=dict, default={"Indian_Pines": 16, "PaviaU": 9, "Botswana": 14}, help="number of category")
parser.add_argument("--band", type=dict, default={"Indian_Pines": 200, "PaviaU": 103, "Botswana": 145}, help="number of band")

parser.add_argument("--bs", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=200, help="number of iteration")
parser.add_argument("--exp", type=int, default=10, help="number of experiments")
parser.add_argument("--stack", type=int, default=3, help="number of residual blocks")

parser.add_argument("filters", type=dict, default={"Indian_Pines": [4, 8, 16, 32], "PaviaU": [8, 16, 32, 64], "Botswana": [4, 8, 16, 32]}, help="filters of SSFEM")
parser.add_argument("kernel_size", type=dict, default={"Indian_Pines": [3, 3, 3], "PaviaU": [3, 3, 3], "Botswana": [3, 3, 3]}, help="kernel_sizes of SSFEM")
parser.add_argument("alpha", type=dict, default={"Indian_Pines": 20., "PaviaU": 10., "Botswana": 20.}, help="alpha of ISS sigmoid")
parser.add_argument("t", type=dict, default={"Indian_Pines": 0.3, "PaviaU": 0.4, "Botswana": 0.5}, help="t of ISS sigmoid")
parser.add_argument("sc_weight", type=dict, default={"Indian_Pines": 0.1, "PaviaU": 0.1, "Botswana": 0.01}, help="weight of spatial consistency loss")


args = parser.parse_args()
args.ds_name = "Indian_Pines"





