# @ FileName: config.py
# @ Author: Alexis
# @ Time: 20-11-28 下午9:17
from os.path import join, dirname, abspath
from src import tool

# Basic project info
AUTHOR = "Youngy"
PROGRAM = "Ultrasonic"
DESCRIPTION = "Biliary atresia diagnose. " \
              "If you find any bug, please new issue. "

# Main CMDs. This decides what kind of cmd you will use.
cmd_list = ['temp', 'train', 'test']

log_name = 'Ultrasonic'

# add parsers to this procedure
globals().update(vars(tool.gen_parser()))


def init_path_config(main_file):
    # global_variables
    gv = globals()
    project_dir = abspath(join(dirname(main_file), '..'))  # 定位到src的上一层
    gv['project_dir'] = project_dir
    gv['data_dir'] = data_dir = join(project_dir, 'data')
    gv['log_dir'] = join(data_dir, 'log')
    gv['loss_dir'] = join(data_dir, 'loss')
    gv['model_dir'] = join(data_dir, 'model')
    gv['best_model_dir'] = join(data_dir, 'best_model')
    gv['step_model_dir'] = join(data_dir, 'step_temp')
    # tensorboard dir
    gv['tb_dir'] = join(data_dir, 'tb')


    gv['ImageNet100_dir'] = '/data/DataSets/MyImagenet'
    gv['CIFAR10_dir'] = '/home/cccc/Desktop/share/deeplearning_project/pnasnet/data/DataSets/cifar10'

    # local
    gv['Gallbladder_train_dir'] = '../../dataset/final_train_cl/siggraph17'
    gv['Gallbladder_test_dir'] = '../../dataset/final_train_cl/siggraph17'

    # lab15
    # gv['Gallbladder_train_dir'] = '/data1/yangyang/project/pnasnet/data/DataSets/final_train'
    # gv['Gallbladder_test_dir'] = '/data1/yangyang/project/pnasnet/data/DataSets/final_train'

    # lab16
    # gv['Gallbladder_train_dir'] = '/data0/yangyang/project/pnasnet/data/DataSets/final_train'
    # gv['Gallbladder_test_dir'] = '/data0/yangyang/project/pnasnet/data/DataSets/final_train'

    # gv['train_csv'] = 'label_contain_empty.csv'
    # gv['test_csv'] = 'label_except_empty.csv'

    # lab97
    # gv['Gallbladder_train_dir'] = '/data/yangyang/project/pnasnet/data/DataSets/final_train'
    # gv['Gallbladder_test_dir'] = '/data/yangyang/project/pnasnet/data/DataSets/final_train'
