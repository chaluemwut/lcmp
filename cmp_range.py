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
    data_range = None
    if range_index == 1:
        data_range = range(0, 10)
    elif range_index == 2:
        data_range = range(10, 20)
    elif range_index == 3:
        data_range = range(20, 30)
    elif range_index == 4:
        data_range = range(30, 40)
    elif range_index == 5:
        data_range = range(40, 50)
    cmpml.process_byrange(data_range)
    log.info('end')

if __name__ == '__main__':
    ml_name = sys.argv[1]
    dataset_name = sys.argv[2]
    range_index = sys.argv[3]
    maincmp(ml_name, dataset_name, range_index)