# @ FileName: agent.py
# @ Author: Alexis
# @ Time: 20-11-28 下午9:17
import os
from os.path import join
import numpy as np
import torch
from tensorboardX import SummaryWriter
from src import tool, logger
from src import dataset as DataSet
from src import net as Model
from src import action as Action


class Trainer:

    def __init__(self, args):
        # cuda setting
        os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']

        # dir setting
        self.model_dir = args['model_dir']
        self.best_model_dir = args['best_model_dir']
        self.step_model_dir = args['step_model_dir']
        tool.check_mkdir(self.model_dir)
        tool.check_mkdir(self.best_model_dir)

        # dataset setting
        self.dataloader = DataSet.get_dataloader(args)
        self.no_eval = args['no_eval']
        self.personal_eval = args['personal_eval']
        self.img_size = args['img_size']
        args['mean'] = self.dataloader.mean
        args['std'] = self.dataloader.std
        args['num_classes'] = self.dataloader.num_classes

        # basic setting
        self.opt_type = args['optimizer']
        self.lr = args['lr']
        self.lr_epoch = args['lr_epoch']
        self.epoch = args['epoch']
        self.weight = args['weight']
        self.eval_best = 0
        self.eval_best_epoch = 0
        self.save_cm = args['save_cm']  # save confusion matrix

        # model name config
        self.model_desc = '{}_{}_{}_{}'. \
            format(args['dataset'], args['model'], args['action'], args['desc'])
        self.model_pkl = self.model_desc + '.ckpt'

        # logger setup
        self.pblog = logger.get_pblog()
        self.pblog.total = self.epoch
        self.tblog = SummaryWriter(join(args['tb_dir'], self.model_desc))

        # model setup
        self.action = Action.get_action(args)
        self.model = Model.get_net(args)

        if args['pre_train']:
            state_dir = join(self.model_dir, self.model_desc)
            state = torch.load(state_dir, map_location='cpu')
            self.model.load_state_dict(state['net'])
        self.model.cuda()
        # self.action.save_graph(self.model, self.img_size, self.tblog,
        #                        self.pblog)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            # ism: IS using Multiple gpus
            self.ism = True
        else:
            self.ism = False

    def __del__(self):
        if hasattr(self, 'tb_log'):
            self.tblog.close()

    def train(self):
        self.pblog.info(self.model_desc)
        optimizer = None
        for epoch in range(self.epoch):
            # get optimizer
            temp = self.action.update_opt(epoch, self.model, self.opt_type,
                                          self.lr, self.lr_epoch)
            if temp is not None:
                optimizer = temp

            self.model.train()
            loss_l = []
            loss_n = []
            dl_len = len(self.dataloader.train)
            ll = len(self.action.eval_legend)
            c_right = np.zeros(ll, np.float32)
            c_sum = np.zeros(ll, np.float32)
            main_loss = 0
            for idx, item in enumerate(self.dataloader.train):
                tx, ty = item['image'], item['label']
                tx, ty = tx.cuda(non_blocking=True), ty.cuda(non_blocking=True)
                # get network output logits
                logits = self.action.cal_logits(tx, self.model)
                # cal loss
                loss = self.action.cal_loss(ty, logits, self.weight)
                # cal acc
                right_e, sum_e = self.action.cal_eval(ty, logits)
                # backward
                optimizer.zero_grad()
                loss[0].backward()
                optimizer.step()

                c_right += right_e
                c_sum += sum_e
                loss_l.append([ii.item() for ii in loss])
                loss_n.append(ty.size(0))
                main_loss += loss[0].item()
                self.pblog.pb(idx, dl_len, 'Loss: %.5f | Acc: %.3f%%' % (
                    main_loss / (idx + 1), c_right / c_sum))
            loss_l = np.array(loss_l).T
            loss_n = np.array(loss_n)
            loss = (loss_l * loss_n).sum(axis=1) / loss_n.sum()
            c_res = c_right / c_sum

            msg = 'Epoch: {:>3}'.format(epoch)
            loss_scalars = self.action.cal_scalars(loss,
                                                   self.action.loss_legend, msg,
                                                   self.pblog)
            self.tblog.add_scalars('loss', loss_scalars, epoch)

            msg = 'train->   '
            acc_scalars = self.action.cal_scalars(c_res,
                                                  self.action.eval_legend, msg,
                                                  self.pblog)
            self.tblog.add_scalars('eval/train', acc_scalars, epoch)

            if not self.no_eval:
                if not self.personal_eval:
                    with torch.no_grad():
                        self.eval(epoch)
                else:
                    with torch.no_grad():
                        self.eval_personal(epoch)

        path = os.path.join(self.model_dir, self.model_desc)
        self.action.save_model(self.ism, self.model, path, self.eval_best,
                               self.eval_best_epoch)
        self.pblog.debug('Training completed, save the last epoch model')
        temp = 'Result, Best: {:.2f}%, Epoch: {}'.format(self.eval_best,
                                                         self.eval_best_epoch)
        self.tblog.add_text('best', temp, self.epoch)
        self.pblog.info(temp)

    def eval(self, epoch):
        self.model.eval()
        ll = len(self.action.eval_legend)
        c_right = np.zeros(ll, np.float32)
        c_sum = np.zeros(ll, np.float32)
        dl_len = len(self.dataloader.eval)
        labels = []
        predictions = []
        for idx, item in enumerate(self.dataloader.eval):
            x, y = item['image'], item['label']
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = self.action.cal_logits(x, self.model)
            right_e, sum_e = self.action.cal_eval(y, logits)
            c_right += right_e
            c_sum += sum_e
            labels.extend(y.cpu().data)
            predictions.extend(logits.argmax(1).cpu().data)
            self.pblog.pb(idx, dl_len, 'Acc: %.3f %%' % (c_right / c_sum))
        msg = 'eval->    '
        c_res = c_right / c_sum
        acc_scalars = self.action.cal_scalars(c_res, self.action.eval_legend,
                                              msg, self.pblog)
        self.tblog.add_scalars('eval/eval', acc_scalars, epoch)

        if self.save_cm:
            cm_figure = self.action.log_confusion_matrix(labels, predictions,
                                                         self.dataloader.class_names)
            self.tblog.add_figure('Confusion Matrix', cm_figure, epoch)

        if c_res[0] > self.eval_best and epoch > 30:
            self.eval_best_epoch = epoch
            self.eval_best = c_res[0]
            path = os.path.join(self.best_model_dir, 'Best_' + self.model_desc)
            self.action.save_model(self.ism, self.model, path, self.eval_best,
                                   self.eval_best_epoch)
            self.pblog.debug('Update the best model')

    def eval_personal(self, epoch):
        self.model.eval()
        ll = len(self.action.eval_legend)
        c_right = np.zeros(ll, np.float32)
        c_sum = np.zeros(ll, np.float32)
        dl_len = len(self.dataloader.eval)

        labels = []
        predictions = []
        preindex = None
        prelabel = None
        personal_vote = [0. for i in range(2)]
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        for idx, item in enumerate(self.dataloader.eval):
            x, y, img_names = item['image'], item['label'], item['img_name']
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = self.action.cal_logits(x, self.model)
            _, prediction = torch.max(logits.data, 1)
            for i, name in enumerate(img_names):
                index, *_ = name.split('_')
                # init pre
                if preindex is None:
                    preindex = index
                if prelabel is None:
                    prelabel = y[0]

                if index != preindex:
                    if personal_vote[0] >= personal_vote[1]:
                        predictions.append(0)
                        if prelabel == 0:
                            class_correct[0] += 1
                    else:
                        predictions.append(1)
                        if prelabel == 1:
                            class_correct[1] += 1
                    labels.append(prelabel.item())
                    class_total[prelabel] += 1
                    personal_vote = [0. for i in range(2)]
                    preindex = index
                    prelabel = y[i]
                    personal_vote[prediction[i]] += 1

                else:
                    personal_vote[prediction[i]] += 1

            self.pblog.pb(idx, dl_len, 'Sen: %.3f %%  | Spe: %.3f %%' % (
                100 * class_correct[0] / (class_total[0] + 1e-6), 100 * class_correct[1] / (class_total[1] + 1e-6)))

        # deal the last patient
        if personal_vote[0] >= personal_vote[1]:
            predictions.append(0)
            if prelabel == 0:
                class_correct[0] += 1
        else:
            predictions.append(1)
            if prelabel == 1:
                class_correct[1] += 1

        labels.append(prelabel.item())
        class_total[prelabel] += 1

        msg = 'eval->    '
        c_res = [100 * class_correct[0] / class_total[0], 100 * class_correct[1] / class_total[1]]
        acc_scalars = self.action.cal_scalars(c_res, self.action.eval_personal_legend,
                                              msg, self.pblog)
        self.tblog.add_scalars('eval/eval', acc_scalars, epoch)

        if self.save_cm:
            cm_figure = self.action.log_confusion_matrix(labels, predictions,
                                                         self.dataloader.class_names)
            self.tblog.add_figure('Confusion Matrix', cm_figure, epoch)

        if c_res[0] > self.eval_best and epoch > 30 and class_correct[1] / class_total[1] >= 0.85:
            self.eval_best_epoch = epoch
            self.eval_best = c_res[0]
            path = os.path.join(self.best_model_dir, 'Best_' + self.model_desc)
            self.action.save_model(self.ism, self.model, path, self.eval_best,
                                   self.eval_best_epoch)
            self.pblog.debug('Update the best model')

        # if epoch % 10 == 0:
        #     path = os.path.join(self.step_model_dir, 'step_{}_'.format(str(epoch / 10)) + self.model_desc)
        #     self.action.save_model(self.ism, self.model, path, self.eval_best, self.eval_best_epoch)
        #     self.pblog.debug('Update the step model')
