import logging, sys, random, copy, pickle, time
from cmp import CmpMl

log = logging.getLogger('data')
data_size = [0.25, 0.50, 0.75]

def initlog():
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(format)
    log.addHandler(ch) 
    fh = logging.FileHandler('log/result.log')
    fh.setFormatter(format)
    log.addHandler(fh)
    
def maincmp(ml_name, dataset_name, range_index):
    initlog()
    log.info('start')
    cmpml = CmpMl(ml_name, dataset_name)
    cmpml.process_gen_training_data(range_index)
    log.info('end')

if __name__ == '__main__':
    ml_name = sys.argv[1]
    dataset_name = sys.argv[2]
    range_index = sys.argv[3]
    maincmp(ml_name, dataset_name, range_index)