# Process the camera internal and external parameters output from Photoscan.

import numpy as np
import os
import os.path as ops
import sys
import pickle

# Read in the output from photoscan.
# The format of the data in the input file:
# Data needs to be arranged in camera order.
# Camera Intrinsics: width heigh f cx cy
# Camera Extrinsics: ID x y z Omega Phi Kappa r11 r12 r13 r21 r22 r23 r31 r32 r33

# 获取当前文件夹路径
filename1, filename2 = "Int.txt", "Ext.txt"
file_path = ops.abspath ( ops.join ( ops.dirname ( __file__ ), '.'  ) )
# 打开内参文件并处理得到内参矩阵
with open(file_path + '/' + filename1) as fn1:
    int_file = fn1.read()

int_list = int_file.split('\t')
width, heigh, f, cx, cy = float(int_list[0]), float(int_list[1]), float(int_list[2]), float(int_list[3]), float(int_list[4])
K =  np.mat(np.zeros((3,3)))
K[0,0] = K[1,1] = f
K[0,2] = width/2+cx
K[1,2] = heigh/2+cy
K[2,2] = 1
# 打开外参文件并处理得到外参矩阵
with open(file_path + '/' + filename2) as fn2:
    ext_file = fn2.read()

ext_list = ext_file.rstrip().split('\n')
ext_list = ext_list[2:]  # 去掉文件中前两行无用信息

ext_dict = {}
# 将相机ID和对应的参数存入字典
for data in ext_list:
    data_list = data.split('\t')
    ext_dict[data_list[0]] = [ float(x) for x in data_list[1:] ]
# photoscan获得的参数的坐标系需经过trans矩阵转换
trans = np.mat([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
# 求得变换矩阵，需要乘以trans变换，再求逆，再删去齐次行
for key,v in ext_dict.items():
    mid_mat = np.mat([[v[6],v[9],v[12],v[0]],[v[7],v[10],v[13],v[1]],[v[8],v[11],v[14],v[2]],[0,0,0,1]])
    ext_dict[key] = np.delete(((mid_mat * trans).I),-1,axis = 0)

P_mat = {}
# p等于内外参矩阵相乘
for key, v in ext_dict.items():
    P_mat[key] = K * v
# mat数据转换为array数据
K = np.array(K)
for key in ext_dict.keys():
    ext_dict[key] = np.array(ext_dict[key])
    P_mat[key] = np.array(P_mat[key])

n = len(ext_dict)
K_list = [K] * n
ext_final_list = []
P_list = []
for v in ext_dict.values():
    ext_final_list.append(v)
for v in P_mat.values():
    P_list.append(v)
# 存为pickle文件
K1 = np.stack ( K_list , axis=0 ).astype ( np.float32 )
P = np.stack ( P_list , axis=0 ).astype ( np.float32 )
RT = np.stack ( ext_final_list , axis=0 ).astype ( np.float32 )
parameter_dict = {'K': K1, 'P': P, 'RT': RT}

with open (file_path + '/' + 'camera_parameter.pickle' , 'wb' ) as f:
    pickle.dump ( parameter_dict, f )
