"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.must import get_dataloader_sample

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill_hsmot as train
from helper.pretrain import init

import logging
from torchvision import models
from models.detr_backbone import build_backbone 

from log.logger import Logger

def parse_option():


    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    # parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--learning_rate', type=float, default=6e-5, help='learning rate')# 
    # parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_epochs', type=str, default='5,10,15', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='hint', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    # parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])
    
    parser.add_argument('--hint_layer_list', default=[2, 3, 4], type=list)
    parser.add_argument('--hint_loss_weights', default=[1,1,1], type=list)

    parser.add_argument('--first_conv_lr_scale', default=10, type=float)

    # exp
    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # opt.model_t = get_teacher_name(opt.path_t)

    # opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                # opt.gamma, opt.alpha, opt.beta, opt.trial)
    # opt.model_name = 'HingWeight:{}_{}_{}_FconvLrScale:{}'.format(opt.hint_weights_layer2, opt.hint_weights_layer3, opt.hint_weights_layer4, opt.first_conv_lr_scale)
    opt.model_name = f'hintlayer:{"_".join(str(x) for x in opt.hint_layer_list)}_hintweights:{"_".join(str(x) for x in opt.hint_loss_weights)}_firstconvLrScale:{opt.first_conv_lr_scale}'

    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # if not os.path.isdir(opt.tb_folder):
        # os.makedirs(opt.tb_folder)

    # opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    # if not os.path.isdir(opt.save_folder):
        # os.makedirs(opt.save_folder)


    opt.work_dir = '/data/users/litianhao/hsmot_code/workdir/distill'
    opt.exp_dir = os.path.join(opt.work_dir, f'{opt.model_name}_{opt.trial}')
    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]



class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    
def load_detr_pretrain(path, model):
    """
    load the pre-trained DETR model
    :param path: path to the pre-trained model
    :param model: the model to load the pre-trained weights into
    :return: None
    """
    print('==> loading pre-trained DETR model')
    checkpoint = torch.load(path)
    state_dict = checkpoint['model']
    useful_dict = {}
    # remove 'module.' prefix from state_dict keys
    for k, v in state_dict.items():
        if ('backbone.0.body.') in k:
            useful_dict[k.replace('backbone.0.', '')] = v

    missing_keys = []
    mismatched_keys = []
    extra_keys = []

    # 检查匹配情况
    for k, v in model.state_dict().items():
        if k not in useful_dict:
            missing_keys.append(k)
            print('missing key: ', k)
        else:
            if v.shape != useful_dict[k].shape:
                mismatched_keys.append(k)
                print('shape mismatch: ', k, v.shape, useful_dict[k].shape)
    for k, v in useful_dict.items():
        if k not in model.state_dict():
            extra_keys.append(k)
            print('extra key: ', k)
    
    # fc.weight and fc.bias
    assert len(missing_keys) == 0 or len(missing_keys)==2, 'missing keys: {}'.format(missing_keys) 
    assert len(mismatched_keys) == 0 or len(mismatched_keys)==1, 'mismatched keys: {}'.format(mismatched_keys)
    assert len(extra_keys) == 0, 'extra keys: {}'.format(extra_keys)

    if mismatched_keys:
        useful_dict.pop(mismatched_keys[0])

    model.load_state_dict(useful_dict, strict=False)
    print('==> done')



def main():
    opt = parse_option()

    # tensorboard logger
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    logger = Logger(
        logdir=opt.exp_dir,
        use_tensorboard=False,
        use_wandb=False,
        only_main=True,
        config=None
    )

    logger.print_config(config = vars(opt), prompt="Runtim Configs: ")
    # dataloader

    train_loader, n_data = get_dataloader_sample(dataset=opt.dataset, batch_size=opt.batch_size,num_workers=opt.num_workers)

    # model
    # 构造教师网络和学生网络
    model_t = build_backbone(input_channels = 3)
    model_s = build_backbone(input_channels = 8)

    # 加载DETR权重
    path = '/data3/litianhao/hsmot/motr/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth'

    load_detr_pretrain(path, model_t)
    load_detr_pretrain(path, model_s)

    # model_t = load_teacher(opt.path_t, n_cls)
    # model_s = model_dict[opt.model_s](num_classes=n_cls)

    data = torch.randn(2, 3, 768, 992)
    data_spec = torch.randn(2, 8, 768, 992)
    model_t.eval()
    model_s.eval()

    '''
        out_shape
        0 :[bs, 64, w/4, h/4]
        1 :[bs, 256, w/4, h/4]
        2 :[bs, 512, w/8, h/8]
        3 :[bs, 1024, w/16, h/16]
        4 :[bs, 2048, w/32, h/32]
        5 :[bs, 2048] 

    '''
    feat_t, _ = model_t(data, preact=True)
    feat_s, _ = model_s(data_spec, preact=True)

    # feat_t, _ = model_t(data, is_feat=True)
    # feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s_list = [ConvReg(feat_s[hint_layer].shape, feat_t[hint_layer].shape) for hint_layer in opt.hint_layer_list]
        module_list.extend(regress_s_list)
        trainable_list.extend(regress_s_list)

        # regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        # module_list.append(regress_s)
        # trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    #! cls 和 div损失都不能使用， 在train函数中排除
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss
    
    def match_names(name, key_names):
        for key in key_names:
            if key in name:
                return True
        return False
    
    firsr_conv_name = '0.body.conv1.weight'

    # 分组
    groups = [
        {
            "params": [p for n, p in trainable_list.named_parameters() if match_names(n, firsr_conv_name) and p.requires_grad],
            "lr_scale": opt.first_conv_lr_scale,
            "lr": opt.learning_rate * opt.first_conv_lr_scale,
        },
        {
            "params": [p for n, p in trainable_list.named_parameters() if not match_names(n, firsr_conv_name) and p.requires_grad],
            "lr_scale": 1,
            "lr": opt.learning_rate,
        }
    ]



    # optimizer
    # optimizer = optim.SGD(trainable_list.parameters(),
    #                       lr=opt.learning_rate,
    #                       momentum=opt.momentum,
    #                       weight_decay=opt.weight_decay)
    optimizer = optim.AdamW(params=groups, 
                            lr=opt.learning_rate, 
                            weight_decay=opt.weight_decay,)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    # teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    # print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt, logger)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        print('train_loss: ', train_loss)

        
        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
            }
            save_file = os.path.join(opt.exp_dir, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)


    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.exp_dir, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
