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


def user_info():
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

def rating_to_csv():

	with open('dataset/ml-100k/u1.test','r') as f:
		data = []
		for line in f:
			data_tmp = line.split() 
			data_tmp = [int(item) for item in data_tmp]
			data.append(data_tmp)
	df = pd.DataFrame(data,columns=['usdr_id','movie_id','rating','timestamp'])
	df.to_csv('dataset/ml-100k/test_rating.csv',index=False)

def rating_matrix():
	#将数据转化为有空数据的矩阵
    rating_data = pd.read_csv('dataset/ml-100k/test_rating.csv')
    user_id = rating_data['usdr_id'].unique()

    movie_id = rating_data['movie_id'].unique()
    rating_matrix = np.zeros([len(user_id),len(movie_id)])
    rating_matrix = pd.DataFrame(rating_matrix, index=user_id, columns=movie_id)
    count = 0
    user_num= len(user_id)
    for uid in user_id:
        user_rating = rating_data[rating_data['usdr_id'] == uid].drop(['usdr_id', 'timestamp'], axis=1)
        user_rated_num = len(user_rating)

        for row in range(0, user_rated_num):
            movieId = user_rating['movie_id'].iloc[row]
            rating_matrix.loc[uid, movieId] = user_rating['rating'].iloc[row]

        count += 1
        if count % 100 == 0:
            completed_percentage = round(float(count) / user_num * 100)
            print("Completed %s" % completed_percentage + "%")
    rating_matrix.to_csv('dataset/ml-100k/user-rating_test.csv')


    
if __name__ == "__main__":
    rating_to_csv()
    rating_matrix()