import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from cmp import data_size

def get_result(ml_name, dataset_name, data_size, result_type):
    fsc_per_ml = []
    if ml_name in ['svm', 'knn'] and dataset_name in ['letter', 'sat']:
        result_obj = []
        for i in range(1, 6):
            result = pickle.load(open('result/{}/range/{}_{}_{}.obj'.format(result_type, ml_name, dataset_name, i), 'rb'))
            result = result[dataset_name]
            result_obj.extend(result)
    else:
        
        result_obj = pickle.load(open('result/{}/{}_{}.obj'.format(result_type, ml_name, dataset_name), 'rb'))
    if ml_name == 'svm' and (dataset_name == 'letter' or dataset_name == 'sat'):
        result_lst = result_obj
    else:
        result_lst = result_obj[dataset_name]
    result_np = np.array(result_lst)
    if data_size == 75:
        fsc_per_ml.extend(np.array(result_np[:, 1], dtype='float'))
    elif data_size == 50:
        fsc_per_ml.extend(np.array(result_np[:, 6], dtype='float'))
    elif data_size == 25:
        fsc_per_ml.extend(np.array(result_np[:, 11], dtype='float'))
        
    return fsc_per_ml 

def process_result():
    for data_size in [75, 50, 25]:
        for ml_name in ['bagging', 'boosted', 'decsiontree', 'randomforest', 'nb', 'svm']:
            a_all_feature = []
            a_missing = []
            for dataset_name in ['heart', 'letter', 'austra', 'german', 'sat', 'segment', 'vehicle']:
                missing = get_result(ml_name, dataset_name, data_size, 'missing')
                all_feature = get_result(ml_name, dataset_name, data_size, 'all_feature')
                
                a_missing.extend(missing)
                a_all_feature.extend(all_feature)
                
            p_value = stats.ttest_ind(a_all_feature, a_missing)[1]
#             if p_value > 0.05:
            print '{} - {} - {}'.format(ml_name, data_size, p_value)
                

if __name__ == '__main__':
    process_result()