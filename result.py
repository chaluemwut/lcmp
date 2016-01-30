import pickle
from dataset_loader import DataSetLoader
import numpy as np
from numpy import dtype
import matplotlib.pyplot as plt
from scipy import stats

def m_lst(data_size, machine_learning_name):
    ml_name_lst = ['bagging', 'boosted', 'decsiontree', 'randomforest', 'nb', 'svm', 'knn']
    fsc_acc = []
    for ml_name in machine_learning_name:
        print '********** ', ml_name
        fsc_per_ml = []
        dataset_name_lst = ['heart', 'letter', 'austra', 'german', 'sat', 'segment', 'vehicle']
        if ml_name in ['svm', 'knn']:
            dataset_name_lst = ['heart', 'austra', 'german', 'segment', 'vehicle']
        for dataset_name in dataset_name_lst:
            if ml_name in ['svm', 'knn'] and dataset_name in ['letter', 'sat']:
                result_obj = []
                for i in range(1, 6):
                    result = pickle.load(open('result/all_feature/range/{}_{}_{}.obj'.format(ml_name, dataset_name, i), 'rb'))
                    result = result[dataset_name]
                    result_obj.extend(result)                
            else:
                result_obj = pickle.load(open('result/all_feature/{}_{}.obj'.format(ml_name, dataset_name), 'rb'))
            result_lst = result_obj[dataset_name]
            result_np = np.array(result_lst)
            if data_size == 75:
                fsc_per_ml.extend(np.array(result_np[:, 1], dtype='float'))
            elif data_size == 50:
                fsc_per_ml.extend(np.array(result_np[:, 6], dtype='float'))
            elif data_size == 25:
                fsc_per_ml.extend(np.array(result_np[:, 11], dtype='float'))
            
        fsc_acc.append(fsc_per_ml)
    return fsc_acc
#     plt.boxplot(fsc_acc)
#     plt.show()

def t_test():
#     random_forest = m_lst(75, ['randomforest'])
#     print np.mean(random_forest)
    data_size = 25
    for m in ['bagging', 'svm', 'boosted', 'decsiontree', 'knn', 'nb']:
        random_forest = m_lst(data_size, ['randomforest'])
        other = m_lst(data_size, [m])
        print len(random_forest[0])
#         print stats.ttest_ind(random_forest[0], other[0])
#     for d_size in [75, 50, 25]:
#         ml_result = m_lst(d_size, ['bagging'])
#         print np.mean(ml_result)
    

def print_result_range():
    for ml_name in ['knn', 'svm']:
        print '*********** ',ml_name
        for data_set in ['letter', 'sat']:
            all_result = []
            for i in range(1, 6):
                result = pickle.load(open('result/missing/range/{}_{}_{}.obj'.format(ml_name, data_set, i), 'rb'))
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
            result_obj = pickle.load(open('result/missing/{}_{}.obj'.format(ml_name, dataset_name), 'rb'))
            result_lst = result_obj[dataset_name]
            result_np = np.array(result_lst)
            print dataset_name, ',', np.mean(np.array(result_np[:, 0], dtype='float')), ',', np.mean(np.array(result_np[:, 1], dtype='float')), ',', np.mean(np.array(result_np[:, 5], dtype='float')), ',', np.mean(np.array(result_np[:, 6], dtype='float')), ',', np.mean(np.array(result_np[:, 10], dtype='float')), ',', np.mean(np.array(result_np[:, 11], dtype='float'))      
   
def print_result():
    print_all()
    print_result_range()
           
if __name__ == '__main__':
    print_result()
#     t_test()
