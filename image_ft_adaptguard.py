import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from adversarial_attack import adversarial_sample_generate

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def normalize(x):
    x = torch.clamp(x, 0, 1)
    x_normalized = torchvision.transforms.functional.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return x_normalized

def log_write(args, log_str):
    assert log_str
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    return

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        # normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["target2"] = ImageList_idx(txt_tar, transform=image_test())
    dset_loaders["target2"] = DataLoader(dsets["target2"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders


def cal_acc(args, loader, netF, netB, netC, flag=False, perturbation=False, v=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if perturbation:
                inputs = inputs + torch.tensor(v)
            inputs = transforms.functional.normalize(inputs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            inputs = normalize(inputs)
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    return accuracy*100, mean_ent


def train_target(args):
    dset_loaders = data_load(args)

    source_netF = network.ResBase(res_name=args.net).cuda()
    source_netB = network.feat_bootleneck(type=args.classifier, feature_dim=source_netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    source_netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    source_modelpath = os.path.join(args.output_dir_src, 'source_F.pt') 
    source_netF.load_state_dict(torch.load(source_modelpath))
    source_modelpath = os.path.join(args.output_dir_src, 'source_B.pt') 
    source_netB.load_state_dict(torch.load(source_modelpath))
    source_modelpath = os.path.join(args.output_dir_src, 'source_C.pt') 
    source_netC.load_state_dict(torch.load(source_modelpath))
    source_netF.eval()
    source_netB.eval()
    source_netC.eval()

    #### source preprocess

    args.topk = args.class_num

    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders["target2"])
        for i in range(len(dset_loaders["target2"])):
            data = iter_test.next()
            inputs, labels = data[0], data[1]
            inputs = inputs.cuda()
            inputs = normalize(inputs)
            outputs = source_netC(source_netB(source_netF(inputs)))
            outputs = nn.Softmax(dim=1)(outputs)
            _, src_idx = torch.sort(outputs, 1, descending=True)
            if args.topk > 0:
                topk = np.min([args.topk, args.class_num])
                for i in range(outputs.size()[0]):
                    outputs[i, src_idx[i, topk:]] = (1.0 - outputs[i, src_idx[i, :topk]].sum())/ (outputs.size()[1] - topk)

            if start_test:
                all_output = outputs.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
        
        mem_P = all_output.detach()

    source_netF.cpu()
    source_netB.cpu()
    source_netC.cpu()

    del source_netF
    del source_netB
    del source_netC

    
    netF = network.ResBase(res_name=args.net)
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck)
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck)

    netF = netF.cuda()
    netB = netB.cuda()
    netC = netC.cuda()

    netF.train()
    netB.train()
    netC.train()

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.kd_max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        if args.ema < 1.0 and iter_num > 0 and iter_num % interval_iter == 0:
            print('KD: ',iter_num,'/',max_iter)
            netF.eval()
            netB.eval()
            netC.eval()
            start_test = True
            with torch.no_grad():
                iter_test = iter(dset_loaders["target2"])
                for i in range(len(dset_loaders["target2"])):
                    data = iter_test.next()
                    inputs = data[0]
                    inputs = inputs.cuda()
                    inputs = normalize(inputs)
                    outputs = netC(netB(netF(inputs)))
                    outputs = nn.Softmax(dim=1)(outputs)
                    if start_test:
                        all_output = outputs.float()
                        start_test = False
                    else:
                        all_output = torch.cat((all_output, outputs.float()), 0)
                mem_P = mem_P * args.ema + all_output.detach() * (1 - args.ema)
            netF.train()
            netB.train()
            netC.train()

        try:
            inputs_target, y, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, y, tar_idx = iter_target.next()

        if inputs_target.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter, power=1.5)
        inputs_target = inputs_target.cuda()
        with torch.no_grad():
            outputs_target_by_source = mem_P[tar_idx, :] / args.kd_temp
            pred_by_source = torch.max(outputs_target_by_source, dim=1)[1].detach()

        t = iter_num/max_iter
        adjust_eps = args.eps * t
        adjust_step_size = 2.5 * (adjust_eps / args.iterations)

        if args.kd_adv_par > 0:
            inputs_target_adv = adversarial_sample_generate(netF, netB, netC, x=inputs_target, iterations=args.iterations, eps=adjust_eps, step_size=adjust_step_size, target=pred_by_source)
            inputs_target_adv = normalize(inputs_target_adv)
            outputs_target_adv  = netC(netB(netF(inputs_target_adv)))
            outputs_target_adv = outputs_target_adv / args.kd_temp
            outputs_target_adv = torch.nn.Softmax(dim=1)(outputs_target_adv)

        inputs_target = normalize(inputs_target)
        outputs_target = netC(netB(netF(inputs_target)))
        outputs_target = outputs_target / args.kd_temp
        outputs_target = torch.nn.Softmax(dim=1)(outputs_target)

        classifier_loss = nn.KLDivLoss(reduction='batchmean')(outputs_target.log(), outputs_target_by_source) * args.kd_par
        if args.kd_adv_par > 0:
            classifier_loss += nn.KLDivLoss(reduction='batchmean')(outputs_target_adv.log(), outputs_target_by_source) * args.kd_adv_par

        optimizer.zero_grad()

        classifier_loss.backward()
        optimizer.step()

    netF.eval()
    netB.eval()
    netC.eval()
    
    acc_s_te, _ = cal_acc(args, dset_loaders['test'], netF, netB, netC, False)
    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    torch.save(netF.state_dict(), osp.join(args.output_dir, "source_F.pt"))
    torch.save(netB.state_dict(), osp.join(args.output_dir, "source_B.pt"))
    torch.save(netC.state_dict(), osp.join(args.output_dir, "source_C.pt"))

    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    # parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'office-home', 'DomainNet126'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output_src', type=str, default='') # SHOT source model
    parser.add_argument('--output', type=str, default='') 
    parser.add_argument('--da', type=str, default='uda', choices=['uda'])

    parser.add_argument('--kd_max_epoch', type=int, default=50)
    parser.add_argument('--kd_temp', type=float, default=1.0)

    parser.add_argument('--kd_par', type=float, default=0.5)
    parser.add_argument('--kd_adv_par', type=float, default=0.5)

    parser.add_argument('--eps', type=float, default=4)
    parser.add_argument('--step_size', type=float, default=0)
    parser.add_argument('--iterations', type=int, default=7)
    parser.add_argument('--ema', type=float, default=0.6)

    parser.add_argument('--backdoor', type=str, default='None', choices=['None', 'Blended', 'SIG'])

    args = parser.parse_args()
    assert args.output_src != ''
    args.output_src = args.output_src.replace('ckps', 'ckps_seed'+str(args.seed))

    args.output = args.output.replace('ckps', 'ckps_seed'+str(args.seed))

    # PGD setting
    args.eps = args.eps / 255.0 if args.eps >= 1.0 else args.eps
    args.step_size = 2.5 * (args.eps / args.iterations) if args.step_size == 0 else args.step_size

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'DomainNet126':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = 'data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    if args.dset == 'DomainNet126':
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_test.txt'

    args.output_dir_src = osp.join(args.output_src, args.da, args.dset, args.backdoor, names[args.s][0].upper())
    args.output_dir = osp.join(args.output, args.da, args.dset, args.backdoor, names[args.s][0].upper()+names[args.t][0].upper())
    args.name = names[args.s][0].upper()+names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(osp.join(args.output_dir, 'log_defense_stage.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target(args)

