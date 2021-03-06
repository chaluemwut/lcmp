import logging, sys, random, copy, pickle, time
from dataset_loader import DataSetLoader
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from config import Config
from svm import LibSVMWrapper
from sklearn import cross_validation
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# issuse
# 1. fix test size at 25%
# 2. after pass crossvalidation use train new model
# 3. keep parameter with use
# 4. kepp all data in machine learning process

log = logging.getLogger('data')
data_size = [0.25, 0.50, 0.75]

class CmpMl(object):
    
    def __init__(self, ml_name, dataset_name):
        self.ml_name = ml_name
        self.dataset_name = dataset_name
    
    def load_dataset(self):
        loader = DataSetLoader()
        lst = loader.loadData()
        return lst

    def gen_ml_lst(self, d_size, dataset_name):
        random_lst = []
        boosted_lst = []
        bagging_lst = []
        for i in Config.base_estimation:
            random_lst.append(RandomForestClassifier(n_estimators=i))
            boosted_lst.append(AdaBoostClassifier(n_estimators=i))
            bagging_lst.append(BaggingClassifier(DecisionTreeClassifier(), n_estimators=i))

        knn_lst = self.gen_knn(d_size, dataset_name)
        
        svm_lst = [LibSVMWrapper(kernel=0),
           LibSVMWrapper(kernel=1, degree=2),
           LibSVMWrapper(kernel=1, degree=3),
           LibSVMWrapper(kernel=2),
           LibSVMWrapper(kernel=3)
           ]
        
        return {
                'bagging':bagging_lst,
                'boosted':boosted_lst,
                'randomforest':random_lst,
                'nb':[GaussianNB()],
                'knn':knn_lst,
                'decsiontree':[DecisionTreeClassifier()],
                'svm':svm_lst
        }
 
    def gen_knn(self, d_size, dataset_name):
        knn_lst = []
        rng = None
        if dataset_name == 'heart':
            if d_size == 0.75:
                # rng = range(2, 12)
                rng = range(22, 51)
            elif d_size == 0.5:
                # rng = range(2, 25)
                rng = range(2, 104)
            elif d_size == 0.25:
                # rng = range(2, 38)
                rng = range(2, 156)
        elif dataset_name == 'letter':
            if d_size == 0.75:
                # rng = range(2, 100)
                rng = range(2, 399)
            elif d_size == 0.5:
                # rng = range(2, 200)
                rng = range(2, 799)
            elif d_size == 0.25:
                # rng = range(2, 300)
                rng = range(2, 1199)
        elif dataset_name == 'austra':
            if d_size == 0.75:
                # rng = range(2, 171)
                rng = range(2, 137)
            elif d_size == 0.5:
                # rng = range(2, 344)
                rng = range(2, 275)
            elif d_size == 0.25:
                # rng = range(2, 102)
                rng = range(2, 413)
        elif dataset_name == 'german':
            if d_size == 0.75:
                # rng = range(2, 49)
                rng = range(2, 199)
            elif d_size == 0.5:
                # rng = range(2, 99)
                rng = range(2, 399)
            elif d_size == 0.25:
                # rng = range(2, 149)
                rng = range(2, 599)
        elif dataset_name == 'sat':
            if d_size == 0.75:
                # rng = range(2, 320)
                rng = range(2, 1286)
            elif d_size == 0.5:
                # rng = range(2, 642)
                rng = range(2, 2573)
            elif d_size == 0.25:
                # rng = range(2, 964)
                rng = range(2, 3860)
        elif dataset_name == 'segment':
            if d_size == 0.75:
                # rng = range(2, 114)
                rng = range(2, 461)
            elif d_size == 0.5:
                # rng = range(2, 230)
                rng = range(2, 923)
            elif d_size == 0.25:
                # rng = range(2, 345)
                rng = range(2, 1385)
        elif dataset_name == 'vehicle':
            if d_size == 0.75:
                # rng = range(2, 36)
                rng = range(2, 149)
            elif d_size == 0.5:
                # rng = range(2, 74)
                rng = range(2, 299)
            elif d_size == 0.25:
                # rng = range(2, 111)
                rng = range(2, 450)

        for i in rng:
            knn_lst.append(KNeighborsClassifier(n_neighbors=i))
            
        return knn_lst

    def cross_validation(self, ml_lst, x, y):
        score_lst = []    
        for ml in ml_lst:
            log.info('start cross val')
            try:
                scores = cross_validation.cross_val_score(ml, x, y, cv=5)
                log.info('end cross val')
                score_lst.append(scores.mean())
            except Exception as e:
                log.info(str(e))       
        np_score = np.array(score_lst)
        max_idx = np_score.argmax()
        return ml_lst[max_idx]
    
    def copy_model(self, ml_org):
        if self.ml_name == 'bagging':
            return RandomForestClassifier(n_estimators=ml_org.n_estimators)
        elif self.ml_name == 'boosted':
            return AdaBoostClassifier(n_estimators=ml_org.n_estimators)
        elif self.ml_name == 'randomforest':
            return RandomForestClassifier(n_estimators=ml_org.n_estimators)
        elif self.ml_name == 'nb':
            return GaussianNB()
        elif self.ml_name == 'knn':
            return KNeighborsClassifier(n_neighbors=ml_org.n_neighbors)
        elif self.ml_name == 'decsiontree':
            return DecisionTreeClassifier()
        elif self.ml_name == 'svm':
            params = ml_org.get_params()
            return LibSVMWrapper(kernel=params['kernel'], degree=params['degree'])
        
    def get_model_parameter(self, ml_org):
        if self.ml_name == 'bagging':
            return ml_org.n_estimators
        elif self.ml_name == 'boosted':
            return ml_org.n_estimators
        elif self.ml_name == 'randomforest':
            return ml_org.n_estimators
        elif self.ml_name == 'knn':
            return ml_org.n_neighbors
        elif self.ml_name == 'svm':
            params = ml_org.get_params()
            return 'kernel = {} degree'.format(params['kernel'], params['degree'])

    def remove_by_chi2_process(self, x, y):
        from sklearn.feature_selection import  SelectKBest, f_classif
        chi2 = SelectKBest(f_classif, k=3)
        x_train = chi2.fit_transform(x, y)
        result_idx = []
        feature_len = len(x[0])
        for i in range(0, feature_len):
            column_data = x[:, i]
            if  np.array_equal(column_data, x_train[:, 0]):
                result_idx.append(i)
            if  np.array_equal(column_data, x_train[:, 1]):
                result_idx.append(i)
            if  np.array_equal(column_data, x_train[:, 2]):
                result_idx.append(i)
        x = np.delete(x, result_idx, axis=1)
        return x, y
    
    def index_remove_by_anova(self, x, y):
        from sklearn.feature_selection import  SelectKBest, f_classif
        chi2 = SelectKBest(f_classif, k=3)
        x_train = chi2.fit_transform(x, y)
        result_idx = []
        feature_len = len(x[0])
        for i in range(0, feature_len):
            column_data = x[:, i]
            if  np.array_equal(column_data, x_train[:, 0]):
                result_idx.append(i)
            if  np.array_equal(column_data, x_train[:, 1]):
                result_idx.append(i)
            if  np.array_equal(column_data, x_train[:, 2]):
                result_idx.append(i)
        return result_idx
    
    def remove_by_index(self, x, result_idx):
        x_new = np.delete(x, result_idx, axis=1)
        return x_new
                            
    def process_gen_training_data(self, range_index=None):
        result = {}
        log.info('***** start ' + self.dataset_name)
        data_range = None
        if range_index == '1':
            data_range = range(0, 10)
        elif range_index == '2':
            data_range = range(10, 20)
        elif range_index == '3':
            data_range = range(20, 30)
        elif range_index == '4':
            data_range = range(30, 40)
        elif range_index == '5':
            data_range = range(40, 50)
        elif range_index == None:
            data_range = range(0, Config.reperating_loop)  
            
        all_data_rec = []
        for i in data_range:
            log.info('*********** loop : {}'.format(i))
            data_rec = []
            for d_size in data_size:
                folder_name = Config.training_data_path
                x_train = pickle.load(open('{}x_train_{}_{}_{}'.format(folder_name, self.dataset_name, d_size, i),'rb'))
                y_train = pickle.load(open('{}y_train_{}_{}_{}'.format(folder_name, self.dataset_name, d_size, i), 'rb'))
                x_test_org = pickle.load(open('{}x_test_{}_{}_{}'.format(folder_name, self.dataset_name, d_size, i), 'rb'))
                y_test_org = pickle.load(open('{}y_test_{}_{}_{}'.format(folder_name, self.dataset_name, d_size, i), 'rb'))
                
                index_remove = self.index_remove_by_anova(x_train, y_train)
                x_train = self.remove_by_index(x_train, index_remove)
                x_test_org = self.remove_by_index(x_test_org, index_remove)
                               
                ml_lst = self.gen_ml_lst(d_size, self.dataset_name)[self.ml_name]
                ml_cross = self.cross_validation(ml_lst, x_train, y_train)
                ml_new_train = self.copy_model(ml_cross)
                ml_c = copy.deepcopy(ml_new_train)
                ml_c.fit(x_train, y_train)
                start = time.time()
                y_pred = ml_c.predict(x_test_org)
                total_time = time.time() - start
                acc = accuracy_score(y_test_org, y_pred)
                fsc = f1_score(y_test_org, y_pred)
                data_rec.append(acc)
                data_rec.append(fsc)
                data_rec.append(total_time)
                data_rec.append(len(y_pred))
                data_rec.append(self.get_model_parameter(ml_c))
                predict_file = 'predict/{}_{}_{}_{}'.format(self.ml_name, self.dataset_name, d_size, i)
                pickle.dump(y_pred, open(predict_file, 'wb'))
                
            all_data_rec.append(data_rec)
        result[self.dataset_name] = all_data_rec
        if range_index == None:
            pickle.dump(result, open('result/{}_{}.obj'.format(self.ml_name, self.dataset_name), 'wb'))
        else:
            pickle.dump(result, open('result/{}_{}_{}.obj'.format(self.ml_name, self.dataset_name, range_index), 'wb'))
        log.info('************** end ')

def initlog():
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(format)
    log.addHandler(ch) 
    fh = logging.FileHandler('log/result.log')
    fh.setFormatter(format)
    log.addHandler(fh)
      
def maincmp(ml_name, dataset_name):
    initlog()
    log.info('start')
    cmpml = CmpMl(ml_name, dataset_name)
    cmpml.process_gen_training_data()
    log.info('end')

if __name__ == '__main__':
    ml_name = sys.argv[1]
    dataset_name = sys.argv[2]
    maincmp(ml_name, dataset_name)
