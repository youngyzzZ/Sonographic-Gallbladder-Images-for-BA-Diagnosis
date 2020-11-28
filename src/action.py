# @ FileName: action.py
# @ Author: Alexis
# @ Time: 20-11-28 下午9:17
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from src import tool


class BaseAction:
    loss_legend = ['| loss: {:0<10.8f}']
    eval_on_train = True
    eval_legend = ['| acc: {:0<5.3f}%']
    eval_personal_legend = ['| sensitivity: {:0<5.3f}%', '| specitivity: {:0<5.3f}%']

    @staticmethod
    def cal_logits(x, net):
        return net(x)

    @staticmethod
    def cal_loss(y, y_hat, weight):
        weigth = torch.tensor([weight]).cuda()
        loss = F.cross_entropy(y_hat, y, weight=weigth)
        return loss,

    @staticmethod
    def cal_eval(y, y_hat):
        count_right = np.empty(1, np.float32)
        count_sum = np.empty(1, np.float32)
        y_hat = y_hat.argmax(1)
        count_right[0] = (y_hat == y).sum().item()
        count_sum[0] = y.size(0)
        return 100 * count_right, count_sum

    @staticmethod
    def update_opt(epoch, net, opt_type, lr=1e-2, lr_epoch=35):
        decay = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        if epoch % lr_epoch == 0:
            times = int(epoch / lr_epoch)
            times = len(decay) - 1 if times >= len(decay) else times
            if opt_type == 'sgd':
                return torch.optim.SGD(net.parameters(), lr=lr * decay[times],
                                       momentum=0.9,
                                       weight_decay=5e-4)
            elif opt_type == 'adam':
                return torch.optim.Adam(net.parameters(), lr=lr * decay[times])
        else:
            return None

    @staticmethod
    def save_model(ism, model, path, *args):
        ''''
        ism, model, path, msg, pblog, acc, epoch
        '''
        acc, epoch = args
        if ism:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        state = {
            'net': state_dict,
            'acc': acc,
            'epoch': epoch}
        torch.save(state, path)

    @staticmethod
    def save_graph(model, img_size, tblog, pblog):
        dummyInput = torch.randn([1, 3, img_size, img_size]).cuda()
        tblog.add_graph(model, dummyInput)
        pblog.debug('Graph saved')

    @staticmethod
    def cal_scalars(metric, metric_legend, msg, pblog):
        scalars = dict()
        for n, s in zip(metric, metric_legend):
            msg += s.format(n)
            scalars[s.split(':')[0][2:]] = n
        pblog.info(msg)
        return scalars

    @staticmethod
    def log_confusion_matrix(labels, predictions, class_names):
        cm = confusion_matrix(labels, predictions)
        # normalise confusion matrix for diff sized groups
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_image = tool.plot_confusion_matrix(cm_norm, class_names)
        return cm_image


def get_action(args):
    action = args['action']
    if action == 'base':
        return BaseAction()
    else:
        raise ValueError('No action: {}'.format(action))
