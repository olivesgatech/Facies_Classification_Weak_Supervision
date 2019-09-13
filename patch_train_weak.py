import argparse
import os
from datetime import datetime
from os.path import join as pjoin
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm

import core.loss
import torchvision.utils as vutils
from core.augmentations_weak import (
    Compose, RandomHorizontallyFlip, RandomRotate, AddNoise)
from core.loader.data_loader import *
from core.metrics import runningScore
from core.models import get_model
import random



torch.backends.cudnn.deterministic = True
torch.manual_seed(2019)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2019)
np.random.seed(seed=2019)


def np_to_tb(array):
    # if 2D :
    if array.ndim == 2:
        # HW => CHW
        array = np.expand_dims(array, axis=0)
        # CHW => NCHW
        array = np.expand_dims(array, axis=0)
    elif array.ndim == 3:
        # HWC => CHW
        array = array.transpose(2, 0, 1)
        # CHW => NCHW
        array = np.expand_dims(array, axis=0)

    array = torch.from_numpy(array)
    array = vutils.make_grid(array, normalize=True, scale_each=True)
    return array


def split_train_val_weak(args, per_val=0.1):
    root = 'data/train_weak'

    root_img = pjoin(root, 'images')
    list_train_val = []
    for file in os.listdir(root_img):
        file_path = os.path.join(root_img, file)
        if os.path.splitext(file_path)[1] == '.mat':
            list_train_val.append(os.path.splitext(file)[0])
    # list_train_val.sort(key=lambda x: int(x[4:]))

    list_train, list_val = train_test_split(
        list_train_val, test_size=per_val, shuffle=True)

    # write to files trainval.txt train.txt val.txt
    file_object = open(
        pjoin(root, 'splits', 'trainval.txt'), 'w')
    file_object.write('\n'.join(list_train_val))
    file_object.close()
    file_object = open(pjoin(root, 'splits', 'train.txt'), 'w')
    file_object.write('\n'.join(list_train))
    file_object.close()
    file_object = open(pjoin(root, 'splits', 'val.txt'), 'w')
    file_object.write('\n'.join(list_val))
    file_object.close()


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate the train and validation sets for the model:
    split_train_val_weak(args, per_val=args.per_val)
    loader = patch_loader_weak

    current_time = datetime.now().strftime('%b%d_%H%M%S')
    log_dir = os.path.join('runs', current_time +
                           f"_{args.arch}_{args.model_name}")
    writer = SummaryWriter(log_dir=log_dir)
    # Setup Augmentations
    if args.aug:
        data_aug = Compose(
            [RandomRotate(15), RandomHorizontallyFlip(), AddNoise()])
    else:
        data_aug = None

    train_set = loader(is_transform=True,
                       split='train',
                       augmentations=data_aug)

    # Without Augmentation:
    val_set = loader(is_transform=True,
                     split='val',
                     patch_size=args.patch_size)


    #if args.mixup:
    #    train_set1 = loader(is_transform=True,
    #                       split='train',
    #                       augmentations=data_aug)




    n_classes = train_set.n_classes

    trainloader = data.DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=True)

#####################################################################
    #shuffle and load
    random.shuffle(train_set.patches['train']) #shuffle list of IDs
    alpha=0.5
    trainloader1 = data.DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  shuffle=True) #load shuffeled data again in another loader
######################################################################


    valloader = data.DataLoader(val_set,
                                batch_size=args.batch_size,
                                num_workers=4)

    # Setup Metrics
    running_metrics = runningScore(n_classes)
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            model = torch.load(args.resume)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    else:
        model = get_model(args.arch, args.pretrained, n_classes)

    # Use as many GPUs as we can
    model = torch.nn.DataParallel(
        model, device_ids=range(torch.cuda.device_count()))
    model = model.to(device)  # Send to GPU

    # PYTROCH NOTE: ALWAYS CONSTRUCT OPTIMIZERS AFTER MODEL IS PUSHED TO GPU/CPU,

    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        print('Using custom optimizer')
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)

    loss_fn = core.loss.focal_loss2d

    if args.class_weights:
        # weights are inversely proportional to the frequency of the classes in the training set
        class_weights = torch.tensor(
            [0, 0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852], device=device, requires_grad=False)
    else:
        class_weights = None

    best_iou = -100.0
    class_names = ['null', 'upper_ns', 'middle_ns',
                   'lower_ns', 'rijnland_chalk', 'scruff', 'zechstein']

    for arg in vars(args):
        text = arg + ': ' + str(getattr(args, arg))
        writer.add_text('Parameters/', text)

    # training
    for epoch in range(args.n_epoch):
        # Training Mode:
        model.train()
        loss_train, total_iteration = 0, 0

        for (i, (images, labels, confs, sims)),(i1, (images1, labels1, confs1, sims1)) in zip(enumerate(trainloader), enumerate(trainloader1)):

            N, c, w, h = labels.shape
            one_hot = torch.FloatTensor(N, 7, w, h).zero_()
            labels_hot = one_hot.scatter_(1, labels.data, 1)  # create one hot representation for the labels

            if args.mixup: #if mixup is true then mix
                lam=torch.from_numpy(np.random.beta(alpha,alpha,(N,1,1,1))).float() #sampling lambda
                one_hot = torch.FloatTensor(N, 7, w, h).zero_()
                labels_hot1 = one_hot.scatter_(1, labels1.data, 1)  # create one hot representation for the labels
                images, labels,labels_hot, confs, sims= (lam*images+(1-lam)*images1),(lam*labels.float()+(1-lam)*labels1.float()), (lam*labels_hot+(1-lam)*labels_hot1), (lam*confs.squeeze()+(1-lam)*confs1.squeeze()), (lam.squeeze()*sims.float()+(1-lam).squeeze()*sims1.float()) #mixup

            image_original = images #TODO Q: Are the passed original lables correct? in the context of following comaprison in line 233
            images, labels_hot, confs, sims = images.to(device), labels_hot.to(device), confs.to(device), sims.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            pred = outputs.detach().max(1)[1].cpu().numpy()
            labels_original=confs.squeeze().permute(0,3,1,2).detach().max(1)[1].cpu().numpy()
            running_metrics.update(labels_original, pred)
            loss = loss_fn(input=outputs, target=labels_hot, conf=confs, alpha=class_weights, sim=sims,
                           gamma=args.gamma, loss_type=args.loss_parameters, soft_dev=args.soft_dev)
            loss_train += loss.item()
            loss.backward()

            # gradient clipping
            if args.clip != 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            total_iteration = total_iteration + 1

            if (i) % 20 == 0:
                print("Epoch [%d/%d] training Loss: %.4f" %
                      (epoch + 1, args.n_epoch, loss.item()))

            numbers = [0, 14, 29]
            if i in numbers:

                tb_original_image = vutils.make_grid(
                    image_original[i][0], normalize=True, scale_each=True)
                writer.add_image('train/original_image',
                                 tb_original_image, epoch + 1)

                # tb_confs_original = vutils.make_grid(confs_tb, normalize=True, scale_each=True)
                # writer.add_image('train/confs_original',tb_confs_original, epoch +1)

                labels_original = labels_original[i]
                correct_label_decoded = train_set.decode_segmap(
                    np.squeeze(labels_original))
                writer.add_image('train/original_label',
                                 np_to_tb(correct_label_decoded), epoch + 1)
                out = F.softmax(outputs, dim=1)

                # this returns the max. channel number:
                prediction = out.max(1)[1].cpu().numpy()[0]
                # this returns the confidence:
                confidence = out.max(1)[0].cpu().detach()[0]
                tb_confidence = vutils.make_grid(
                    confidence, normalize=True, scale_each=True)

                decoded = train_set.decode_segmap(np.squeeze(prediction))
                writer.add_image('train/predicted',
                                 np_to_tb(decoded), epoch + 1)
                writer.add_image('train/confidence', tb_confidence, epoch + 1)

                unary = outputs.cpu().detach()
                unary_max = torch.max(unary)
                unary_min = torch.min(unary)
                unary = unary.add((-1*unary_min))
                unary = unary/(unary_max - unary_min)

                for channel in range(0, len(class_names)):
                    decoded_channel = unary[0][channel]
                    tb_channel = vutils.make_grid(
                        decoded_channel, normalize=True, scale_each=True)
                    writer.add_image(
                        f'train_classes/_{class_names[channel]}', tb_channel, epoch + 1)

        # Average metrics, and save in writer()
        loss_train /= total_iteration
        score, class_iou = running_metrics.get_scores()
        writer.add_scalar('train/Pixel Acc', score['Pixel Acc: '], epoch+1)
        writer.add_scalar('train/Mean Class Acc',
                          score['Mean Class Acc: '], epoch+1)
        writer.add_scalar('train/Freq Weighted IoU',
                          score['Freq Weighted IoU: '], epoch+1)
        writer.add_scalar('train/Mean_IoU', score['Mean IoU: '], epoch+1)

        confusion = score['confusion_matrix']
        writer.add_image(f'train/confusion matrix',
                         np_to_tb(confusion), epoch + 1)

        running_metrics.reset()
        writer.add_scalar('train/loss', loss_train, epoch+1)

        if args.per_val != 0:
            with torch.no_grad():  # operations inside don't track history
                # Validation Mode:
                model.eval()
                loss_val, total_iteration_val = 0, 0

                for i_val, (images_val, labels_val, conf_val, sim_val) in tqdm(enumerate(valloader)):

                    N, c, w, h = labels_val.shape
                    one_hot = torch.FloatTensor(N, 7, w, h).zero_()
                    labels_hot_val = one_hot.scatter_(1, labels_val.data, 1)  # create one hot representation for the labels

                    image_original, labels_original = images_val, labels_val
                    images_val, labels_hot_val, conf_val, sim_val = images_val.to(
                        device), labels_hot_val.to(device), conf_val.to(device), sim_val.to(device)

                    outputs_val = model(images_val)
                    pred = outputs_val.detach().max(1)[1].cpu().numpy()
                    gt = labels_val.numpy()

                    running_metrics_val.update(gt, pred)

                    loss = loss_fn(input=outputs_val, target=labels_hot_val, conf=conf_val, alpha=class_weights,
                                   sim=sim_val, gamma=args.gamma, loss_type=args.loss_parameters, soft_dev=args.soft_dev)

                    total_iteration_val = total_iteration_val + 1

                    if (i_val) % 20 == 0:
                        print("Epoch [%d/%d] validation Loss: %.4f" %
                              (epoch, args.n_epoch, loss.item()))

                    numbers = [0]
                    if i_val in numbers:

                        tb_original_image = vutils.make_grid(
                            image_original[i_val][0], normalize=True, scale_each=True)
                        writer.add_image('val/original_image',
                                         tb_original_image, epoch)
                        labels_original = labels_original.numpy()[0]
                        correct_label_decoded = train_set.decode_segmap(
                            np.squeeze(labels_original))
                        writer.add_image(
                            'val/original_label', np_to_tb(correct_label_decoded), epoch + 1)

                        out = F.softmax(outputs_val, dim=1)

                        # this returns the max. channel number:
                        prediction = out.max(1)[1].cpu().detach().numpy()[0]
                        # this returns the confidence:
                        confidence = out.max(1)[0].cpu().detach()[0]
                        tb_confidence = vutils.make_grid(
                            confidence, normalize=True, scale_each=True)

                        decoded = train_set.decode_segmap(
                            np.squeeze(prediction))
                        writer.add_image(
                            'val/predicted', np_to_tb(decoded), epoch + 1)
                        writer.add_image('val/confidence',
                                         tb_confidence, epoch + 1)

                        unary = outputs.cpu().detach()
                        unary_max, unary_min = torch.max(
                            unary), torch.min(unary)
                        unary = unary.add((-1*unary_min))
                        unary = unary/(unary_max - unary_min)

                        for channel in range(0, len(class_names)):
                            tb_channel = vutils.make_grid(
                                unary[0][channel], normalize=True, scale_each=True)
                            writer.add_image(
                                f'val_classes/_{class_names[channel]}', tb_channel, epoch + 1)

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)

                writer.add_scalar(
                    'val/Pixel Acc', score['Pixel Acc: '], epoch+1)
                writer.add_scalar('val/Mean IoU', score['Mean IoU: '], epoch+1)
                writer.add_scalar('val/Mean Class Acc',
                                  score['Mean Class Acc: '], epoch+1)
                writer.add_scalar('val/Freq Weighted IoU',
                                  score['Freq Weighted IoU: '], epoch+1)

                confusion = score['confusion_matrix']
                writer.add_image(f'val/confusion matrix',
                                 np_to_tb(confusion), epoch + 1)
                writer.add_scalar('val/loss', loss.item(), epoch+1)
                running_metrics_val.reset()

                if score['Mean IoU: '] >= best_iou:
                    best_iou = score['Mean IoU: ']
                    model_dir = os.path.join(
                        log_dir, f"{args.arch}_model_best.pkl")
                    torch.save(model, model_dir)

                if epoch % 10 == 0:
                    model_dir = os.path.join(
                        log_dir, f"{args.arch}_ep{epoch}_model.pkl")
                    torch.save(model, model_dir)

        else:  # validation is turned off:
            # just save the latest model:
            if epoch % 10 == 0:
                model_dir = os.path.join(
                    log_dir, f"{args.arch}_ep{epoch+1}_model.pkl")
                torch.save(model, model_dir)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='patch_deconvnet_skip',
                        help='Architecture to use [\'patch_deconvnet, patch_deconvnet_skip\']')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=1500,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=64,
                        help='Batch Size')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--clip', nargs='?', type=float, default=0,
                        help='Max norm of the gradients if clipping. Set to zero to disable. ')
    parser.add_argument('--per_val', nargs='?', type=float, default=0.05,
                        help='percentage of the training data for validation')
    parser.add_argument('--patch_size', nargs='?', type=int, default=75,
                        help='The size of each patch')
    parser.add_argument('--pretrained', nargs='?', type=bool, default=True,
                        help='Pretrained models not supported. Keep as False for now.')
    parser.add_argument('--aug', nargs='?', type=bool, default=True,
                        help='Whether to use data augmentation.')
    parser.add_argument('--gamma', nargs='?', type=int, default=2,
                        help='Switch between cross entropy "0"  focal loss "2"')
    parser.add_argument('--class_weights', nargs='?', type=bool, default=None,
                        help='Whether to use class weights to reduce the effect of class imbalance')
    parser.add_argument('--model_name', nargs='?', type=str, default='test',
                        help='name to associate with the model')
    parser.add_argument('--loss_parameters', nargs='?', type=str, default='LinConf_LinSim',
                        help='Loss function parameters: for fully supervised model select  ["baseline", "LinConf_NonSim", "LinConf_LinSim", "SoftConf_SoftSim"]')
    parser.add_argument('--soft_dev', nargs=2, type=int, default=None,
                        help='Deviation inside softmax c1 and c2')
    parser.add_argument('--mixup', nargs='?', type=bool, default=True,
                        help='Activate or deactivate mixup')
    args = parser.parse_args()
    train(args)


