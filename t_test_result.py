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
    elif ml_name == 'knn' and (dataset_name == 'letter' or dataset_name == 'sat'):
        result_lst = result_obj        
    else:
        result_lst = result_obj[dataset_name]
    result_np = np.array(result_lst)
    index_access = [1, 6, 11]
#     index_access = [0, 5, 10]
    
    if data_size == 75:
        fsc_per_ml.extend(np.array(result_np[:, index_access[0]], dtype='float'))
    elif data_size == 50:
        fsc_per_ml.extend(np.array(result_np[:, index_access[1]], dtype='float'))
    elif data_size == 25:
        fsc_per_ml.extend(np.array(result_np[:, index_access[2]], dtype='float'))
        
    return fsc_per_ml 

def process_result():
    all_diff = []
    ml_name_lst = ['bagging', 'boosted', 'decsiontree', 'randomforest', 'nb', 'svm', 'knn']
    for ml_name in ml_name_lst:
        diff = []
        for dataset_name in ['heart', 'letter', 'austra', 'german', 'sat', 'segment', 'vehicle']:
            for data_size in [75, 50, 25]:
                missing = get_result(ml_name, dataset_name, data_size, 'missing')
                all_feature = get_result(ml_name, dataset_name, data_size, 'all_feature')
                diff_result = (np.array(all_feature) - np.array(missing)) * 100.0 / np.array(all_feature)
                diff.extend(diff_result)
        all_diff.append(diff)
    
#     plt.subplot(2, 1, 1)
#     plt.plot(all_diff[0])
     
#     plt.subplot(2, 1, 2)
    plt.boxplot(all_diff, meanline=True, showmeans=True)
    means = [np.mean(x) for x in all_diff]
    print means
    print len(all_diff[1]), np.average(all_diff[1]), np.sum(all_diff[1])
    print len(all_diff[6]), np.average(all_diff[6]), all_diff[6]
    plt.scatter(range(1, 8), means)
    plt.xticks(range(1, 8), ml_name_lst)
    plt.show()

                
if __name__ == '__main__':
    process_result()
