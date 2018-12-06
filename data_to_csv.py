#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File : data_to_csv.py
@Time : 2018/12/06 23:09:10
@Author : JiahuiRen 
'''
# here put the import lib

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
def user_data():
	with open ('ml-100k/u.occupation','r') as f:
		workitem = {}
		i = 0 
		for line in f:
			line = line.strip('\n')
			workitem[str(line)] = i
			i +=1
	with open("ml-100k/u.user",'r') as f:
		user_info = []
		for line in f:
			user_info_tmp =[]
			data = line.split('|')
			user_id = int(data[0])
			user_age = int(data[1])
			if data[2] == 'M':
				user_sex = 0
			else:
				user_sex = 1
			user_work = int(workitem[str(data[3])])
			user_info_tmp.append(user_id)
			user_info_tmp.append(user_age)
			user_info_tmp.append(user_sex)
			user_info_tmp.append(user_work)
			user_info.append(user_info_tmp)
	user_data = np.array(user_info)
	onehot = OneHotEncoder()
	onehot_work = onehot.fit_transform(user_data[:,3:4]).toarray()
	user_data = np.append(user_data[:,0:3],onehot_work,axis=1)
	return user_data