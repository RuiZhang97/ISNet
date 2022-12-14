"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import re
import os
import shutil
import torch
from datetime import datetime
import logging
from subprocess import call
import shlex
from tensorboardX import SummaryWriter
import numpy as np
from utils.image_page import ImagePage
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from PIL import Image

# Create unique output dir name based on non-default command line args
def make_exp_name(args, parser):
    exp_name = '{}-{}'.format(args.dataset[:3], args.arch[:])
    dict_args = vars(args)

    # sort so that we get a consistent directory name
    argnames = sorted(dict_args)

    # build experiment name with non-default args
    for argname in argnames:
        if dict_args[argname] != parser.get_default(argname):
            if argname == 'exp' or argname == 'arch' or argname == 'prev_best_filepath':
                continue
            if argname == 'snapshot':
                arg_str = '-PT'
            elif argname == 'nosave':
                arg_str = ''
                argname=''
            elif argname == 'freeze_trunk':
                argname = ''
                arg_str = '-fr'
            elif argname == 'syncbn':
                argname = ''
                arg_str = '-sbn'
            elif argname == 'relaxedloss':
                argname = ''
                arg_str = 're-loss'
            elif isinstance(dict_args[argname], bool):
                arg_str = 'T' if dict_args[argname] else 'F'
            else:
                arg_str = str(dict_args[argname])[:6]
            exp_name += '-{}_{}'.format(str(argname), arg_str)
    # clean special chars out
    exp_name = re.sub(r'[^A-Za-z0-9_\-]+', '', exp_name)
    exp_name = 'testing'
    return exp_name


def save_log(prefix, output_dir, date_str):
    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    filename = os.path.join(output_dir, prefix + '_' + date_str + '.log')
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=date_fmt,
                        filename=filename, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def save_code(exp_path, date_str):
    code_root = '.'  # FIXME!
    zip_outfile = os.path.join(exp_path, 'code_{}.tgz'.format(date_str))
    print('Saving code to {}'.format(zip_outfile))
    cmd = 'tar -czvf {zip_outfile} --exclude=\'*.pyc\' --exclude=\'*.png\' ' +\
        '--exclude=\'*tfevents*\' {root}/train.py ' + \
        ' {root}/utils {root}/datasets {root}/models'
    cmd = cmd.format(zip_outfile=zip_outfile, root=code_root)
    call(shlex.split(cmd), stdout=open(os.devnull, 'wb'))


def prep_experiment(args, parser):
    '''
    Make output directories, setup logging, Tensorboard, snapshot code.
    '''
    ckpt_path = args.ckpt
    tb_path = args.tb_path
    exp_name = make_exp_name(args, parser)
    args.exp_path = os.path.join(ckpt_path, args.exp, exp_name)
    args.tb_exp_path = os.path.join(tb_path, args.exp, exp_name)
    args.ngpu = torch.cuda.device_count()
    args.date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                        'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    args.last_record = {}
    os.makedirs(args.exp_path, exist_ok=True)
    os.makedirs(args.tb_exp_path, exist_ok=True)
    save_log('log', args.exp_path, args.date_str)
    #save_code(args.exp_path, args.date_str)
    open(os.path.join(args.exp_path, args.date_str + '.txt'), 'w').write(
        str(args) + '\n\n')

    writer = SummaryWriter(logdir=args.tb_exp_path, comment=args.tb_tag)
    return writer

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def evaluate_eval(args, net, optimizer, val_loss, mf_score, hist, dump_images, heatmap_images, writer, epoch=0, dataset=None, ):
    '''
    Modified IOU mechanism for on-the-fly IOU calculations ( prevents memory overflow for
    large dataset) Only applies to eval/eval.py
    '''
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    print_evaluate_results(hist, iu, writer, epoch, dataset)
    freq = hist.sum(axis=1) / hist.sum()
    mean_iu = np.nanmean(iu)
    logging.info('mean {}'.format(mean_iu))
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    #return acc, acc_cls, mean_iu, fwavacc

    # update latest snapshot
    if 'mean_iu' in args.last_record:
        last_snapshot = 'last_epoch_{}_mean-iu_{:.5f}.pth'.format(
            args.last_record['epoch'], args.last_record['mean_iu'])
        last_snapshot = os.path.join(args.exp_path, last_snapshot)
        try:
            os.remove(last_snapshot)
        except OSError:
            pass
    last_snapshot = 'last_epoch_{}_mean-iu_{:.5f}.pth'.format(epoch, mean_iu)
    last_snapshot = os.path.join(args.exp_path, last_snapshot)
    args.last_record['mean_iu'] = mean_iu
    args.last_record['epoch'] = epoch
    
    torch.cuda.synchronize()
    
    torch.save({
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'mean_iu': mean_iu,
        'command': ' '.join(sys.argv[1:])
    }, last_snapshot)

    # update best snapshot
    if mean_iu > args.best_record['mean_iu'] :
        # remove old best snapshot
        if args.best_record['epoch'] != -1:
            best_snapshot = 'best_epoch_{}_mean-iu_{:.5f}.pth'.format(
                args.best_record['epoch'], args.best_record['mean_iu'])
            best_snapshot = os.path.join(args.exp_path, best_snapshot)
            assert os.path.exists(best_snapshot), \
                'cant find old snapshot {}'.format(best_snapshot)
            os.remove(best_snapshot)

        
        # save new best
        args.best_record['val_loss'] = val_loss.avg
        args.best_record['mask_f1_score'] = mf_score.avg
        args.best_record['epoch'] = epoch
        args.best_record['acc'] = acc
        args.best_record['acc_cls'] = acc_cls
        args.best_record['mean_iu'] = mean_iu
        args.best_record['fwavacc'] = fwavacc

        best_snapshot = 'best_epoch_{}_mean-iu_{:.5f}.pth'.format(
            args.best_record['epoch'], args.best_record['mean_iu'])
        best_snapshot = os.path.join(args.exp_path, best_snapshot)
        shutil.copyfile(last_snapshot, best_snapshot)
        
    
        to_save_dir = os.path.join(args.exp_path, 'best_images')
        os.makedirs(to_save_dir, exist_ok=True)
        ip = ImagePage(epoch, '{}/index.html'.format(to_save_dir))

        val_visual = []
        
        idx = 0
        
        visualize = standard_transforms.Compose([
            standard_transforms.Scale(384),
            standard_transforms.ToTensor()
        ])
        for bs_idx, bs_data in enumerate(dump_images):
            for local_idx, data in enumerate(zip(bs_data[0], bs_data[1],bs_data[2])):
                gt_pil = args.dataset_cls.colorize_mask(data[0].cpu().numpy())
                pred = data[1].cpu().numpy()
                predictions_pil = args.dataset_cls.colorize_mask(pred)
                img_name = data[2]
                
                prediction_fn = '{}_prediction.png'.format(img_name)
                predictions_pil.save(os.path.join(to_save_dir, prediction_fn))
                gt_fn = '{}_gt.png'.format(img_name)
                gt_pil.save(os.path.join(to_save_dir, gt_fn))
                ip.add_table([(gt_fn, 'gt'), (prediction_fn, 'prediction')])
                val_visual.extend([visualize(gt_pil.convert('RGB')),
                                   visualize(predictions_pil.convert('RGB'))])
                idx = idx+1
                if idx >= 9:
                    ip.write_page()
                    break
        for bs_idx, bs_data in enumerate(heatmap_images):
            for local_idx, data in enumerate(zip(bs_data[0], bs_data[1],bs_data[2])):
            
                gt_pil = args.dataset_cls.colorize_mask(data[0].cpu().numpy())
                
                predictions_pil = data[1].cpu().numpy()
                predictions_pil = (predictions_pil / predictions_pil.max()) * 255
                predictions_pil = Image.fromarray(predictions_pil.astype(np.uint8))
                img_name = data[2]
                
                prediction_fn = '{}_prediction.png'.format(img_name)
                predictions_pil.save(os.path.join(to_save_dir, prediction_fn))
                gt_fn = '{}_gt.png'.format(img_name)
                gt_pil.save(os.path.join(to_save_dir, gt_fn))
                ip.add_table([(gt_fn, 'gt'), (prediction_fn, 'prediction')])
                val_visual.extend([visualize(gt_pil.convert('RGB')),
                                   visualize(predictions_pil.convert('RGB'))])
                idx = idx+1
                if idx >= 9:
                    ip.write_page()
                    break

        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=10, padding=5)
        writer.add_image(last_snapshot, val_visual)

    logging.info('-' * 107)
    fmt_str = '[epoch %d], [val loss %.5f], [mask f1 %.5f], [acc %.5f], [acc_cls %.5f], ' +\
              '[mean_iu %.5f], [fwavacc %.5f]'
    logging.info(fmt_str % (epoch, val_loss.avg, mf_score.avg, acc, acc_cls, mean_iu, fwavacc))
    fmt_str = 'best record: [val loss %.5f], [mask f1 %.5f], [acc %.5f], [acc_cls %.5f], ' +\
              '[mean_iu %.5f], [fwavacc %.5f], [epoch %d], '
    logging.info(fmt_str % (args.best_record['val_loss'], args.best_record['mask_f1_score'],
                            args.best_record['acc'],
                            args.best_record['acc_cls'], args.best_record['mean_iu'],
                            args.best_record['fwavacc'], args.best_record['epoch']))
    logging.info('-' * 107)

    # tensorboard logging of validation phase metrics

    writer.add_scalar('training/acc', acc, epoch)
    writer.add_scalar('training/acc_cls', acc_cls, epoch)
    writer.add_scalar('training/mean_iu', mean_iu, epoch)
    writer.add_scalar('training/val_loss', val_loss.avg, epoch)
    writer.add_scalar('training/mask_f1_score', mf_score.avg, epoch)


def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist



def print_evaluate_results(hist, iu, writer=None, epoch=0, dataset=None):
    try:
        id2cat = dataset.id2cat
    except:
        id2cat = {i: i for i in range(dataset.num_classes)}
    iu_false_positive = hist.sum(axis=1) - np.diag(hist)
    iu_false_negative = hist.sum(axis=0) - np.diag(hist)
    iu_true_positive = np.diag(hist)

    logging.info('IoU:')
    logging.info('label_id      label    iU    Precision Recall TP     FP    FN')
    for idx, i in enumerate(iu):
        idx_string = "{:2d}".format(idx)
        class_name = "{:>13}".format(id2cat[idx]) if idx in id2cat else ''
        iu_string = '{:5.2f}'.format(i * 100)
        total_pixels = hist.sum()
        tp = '{:5.2f}'.format(100 * iu_true_positive[idx] / total_pixels)
        fp = '{:5.2f}'.format(
            iu_false_positive[idx] / iu_true_positive[idx])
        fn = '{:5.2f}'.format(iu_false_negative[idx] / iu_true_positive[idx])
        precision = '{:5.2f}'.format(
            iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_positive[idx]))
        recall = '{:5.2f}'.format(
            iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_negative[idx]))
        logging.info('{}    {}   {}  {}     {}  {}   {}   {}'.format(
            idx_string, class_name, iu_string, precision, recall, tp, fp, fn))

        writer.add_scalar('val_class_iu/{}'.format(id2cat[idx]), i * 100, epoch)
        writer.add_scalar('val_class_precision/{}'.format(id2cat[idx]),
                          iu_true_positive[idx] / (iu_true_positive[idx] +
                                                   iu_false_positive[idx]), epoch)
        writer.add_scalar('val_class_recall/{}'.format(id2cat[idx]),
                          iu_true_positive[idx] / (iu_true_positive[idx] +
                                                   iu_false_negative[idx]), epoch)
