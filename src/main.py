# @ FileName: main.py
# @ Author: Alexis
# @ Time: 20-11-28 下午9:17
from src import config, agent, logger

if __name__ == '__main__':
    config.init_path_config(__file__)  # __file__当前文件的绝对路径
    pblog = logger.get_pblog(total=config.epoch)
    try:
        pblog.debug('###start###')
        if config.cmd == 'temp':
            pass
        elif config.cmd == 'train':
            args = {'desc': config.desc,
                    'cuda': config.cuda,
                    'num_workers': config.num_workers,
                    'dataset': config.dataset,
                    'img_size': config.img_size,
                    'ImageNet1000_dir': config.ImageNet100_dir,
                    'Gallbladder_train_dir': config.Gallbladder_train_dir,
                    'Gallbladder_test_dir': config.Gallbladder_test_dir,
                    'train_csv': config.train_csv,
                    'test_csv': config.test_csv,
                    'model': config.model,
                    'optimizer': config.optimizer,
                    'action': config.action,
                    'model_dir': config.model_dir,
                    'best_model_dir': config.best_model_dir,
                    'step_model_dir': config.step_model_dir,
                    'tb_dir': config.tb_dir,
                    'pre_train': config.pre_train,
                    'epoch': config.epoch,
                    'batch_size': config.batch_size,
                    'weight': config.weight,
                    'lr': config.lr,
                    'lr_epoch': config.lr_epoch,
                    'no_eval': config.no_eval,
                    'personal_eval': config.personal_eval,
                    'save_cm': config.save_cm}
            agent.Trainer(args).train()
        else:
            raise ValueError('No cmd: {}'.format(config.cmd))
    except:
        pblog.exception('Exception Logged')
        exit(1)
    else:
        pblog.debug('###ok###')
