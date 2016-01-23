import pickle
from dataset_loader import DataSetLoader
import numpy as np
from numpy import dtype


def print_result_range():
    for ml_name in ['knn', 'svm']:
        print '*********** ',ml_name
        for data_set in ['letter', 'sat']:
            all_result = []
            for i in range(1, 6):
                result = pickle.load(open('result/all_feature/range/{}_{}_{}.obj'.format(ml_name, data_set, i), 'rb'))
                result = result[data_set]
                all_result.extend(result)
            result_np = np.array(all_result)
            print data_set, ',', np.mean(np.array(result_np[:, 0], dtype='float')), ',', np.mean(np.array(result_np[:, 1], dtype='float')), ',', np.mean(np.array(result_np[:, 5], dtype='float')), ',', np.mean(np.array(result_np[:, 6], dtype='float')), ',', np.mean(np.array(result_np[:, 10], dtype='float')), ',', np.mean(np.array(result_np[:, 11], dtype='float'))      
            
def print_all():
    ml_name_lst = ['bagging', 'boosted', 'decsiontree', 'randomforest', 'nb', 'svm', 'knn']
    for ml_name in ml_name_lst:
        print '********** ', ml_name
        dataset_name_lst = ['heart', 'letter', 'austra', 'german', 'sat', 'segment', 'vehicle']
        if ml_name in ['svm', 'knn']:
            dataset_name_lst = ['heart', 'austra', 'german', 'segment', 'vehicle']
        for dataset_name in dataset_name_lst:
            result_obj = pickle.load(open('result/all_feature/{}_{}.obj'.format(ml_name, dataset_name), 'rb'))
            result_lst = result_obj[dataset_name]
            result_np = np.array(result_lst)
            print dataset_name, ',', np.mean(np.array(result_np[:, 0], dtype='float')), ',', np.mean(np.array(result_np[:, 1], dtype='float')), ',', np.mean(np.array(result_np[:, 5], dtype='float')), ',', np.mean(np.array(result_np[:, 6], dtype='float')), ',', np.mean(np.array(result_np[:, 10], dtype='float')), ',', np.mean(np.array(result_np[:, 11], dtype='float'))      
   
def print_result():
#     print_all()
    print_result_range()
           
if __name__ == '__main__':
    print_result()
