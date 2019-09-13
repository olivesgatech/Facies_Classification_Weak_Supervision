import argparse
import os
from datetime import datetime
import torch.nn as nn
from os.path import join as pjoin
import itertools

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm
from core.augmentations import (Compose, RandomHorizontallyFlip, RandomRotate, AddNoise)
import torchvision.utils as vutils
from core.loader.data_loader import * #Import all different loaders
from core.metrics import runningScore
from core.models import get_model
import core.loss #import the loss function





torch.backends.cudnn.deterministic = True
torch.manual_seed(2019)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(2019)
np.random.seed(seed=2019)

def split_train_val(args, per_val=0.1):
    # create inline and crossline sections for training and validation:
    loader_type = 'section' #section loader
    labels = np.load(pjoin('data', 'train', '' 'y')) #
    i_list = list(range(labels.shape[0])) #list of inline section numbers
    i_list = ['i_'+str(inline) for inline in i_list] #create list of inline section names

    x_list = list(range(labels.shape[1])) #create a list of crossline section numbers
    x_list = ['x_'+str(crossline) for crossline in x_list] #create list of crossline section names

    list_train_val = i_list + x_list #concatinate inline and crossline list names

    # create train and test splits:
    list_train, list_val = train_test_split(list_train_val, test_size=per_val, shuffle=True) #sklearn ready made function that creartes train validation splits

    # write to files to disK:
    file_object = open(pjoin('data', 'splits', loader_type + '_train_val.txt'), 'w') #path to store train and validation lists
    file_object.write('\n'.join(list_train_val))
    file_object.close()
    file_object = open(pjoin('data', 'splits', loader_type + '_train.txt'), 'w') #path to store train list only
    file_object.write('\n'.join(list_train))
    file_object.close()
    file_object = open(pjoin('data', 'splits', loader_type + '_val.txt'), 'w') #path to store validation only
    file_object.write('\n'.join(list_val))
    file_object.close()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Selects Torch Device
    split_train_val(args, per_val=args.per_val)     #Generate the train and validation sets for the model as text files:

    current_time = datetime.now().strftime('%b%d_%H%M%S') #Gets Current Time and Date
    log_dir = os.path.join('runs', current_time + f"_{args.arch}_{args.model_name}")  #Greate the log directory
    writer = SummaryWriter(log_dir=log_dir) #Initialize the tensorboard summary writer

    # Setup Augmentations
    if args.aug: #if augmentation is true
        data_aug = Compose([RandomRotate(10), RandomHorizontallyFlip(), AddNoise()]) #compose some augmentation functions
    else:
        data_aug = None

    loader = section_loader #name the loader
    train_set = loader(is_transform=True,split='train',augmentations=data_aug) #use custom data loader to get the training set (instance of the loader class)
    val_set = loader(is_transform=True,split='val') #use custom made data  loader to get the validation

    n_classes = train_set.n_classes #initalize the number of classes which is hard coded in the dataloader

    # Create sampler:

    shuffle = False  # must turn False if using a custom sampler
    with open(pjoin('data', 'splits', 'section_train.txt'), 'r') as f:train_list = f.read().splitlines() #load the section train list previously stored in a text file created by split_train_val() function
    with open(pjoin('data', 'splits', 'section_val.txt'), 'r') as f:val_list = f.read().splitlines() #load the section train list previously stored in a text file created by split_train_val() function

    class CustomSamplerTrain(torch.utils.data.Sampler): #create a custom sampler
        def __iter__(self):
            char = ['i' if np.random.randint(2) == 1 else 'x'] #choose randomly between letter i and letter x
            self.indices = [idx for (idx, name) in enumerate(train_list) if char[0] in name] #choose index all inlines or all crosslines from the training list created by split_train_val() function
            return (self.indices[i] for i in torch.randperm(len(self.indices))) #shuffle the indices and return them

    class CustomSamplerVal(torch.utils.data.Sampler):
        def __iter__(self):
            char = ['i' if np.random.randint(2) == 1 else 'x'] #choose randomly between letter i and letter x
            self.indices = [idx for (idx, name) in enumerate(val_list) if char[0] in name] #choose index all inlines or all crosslines from the validation list created by split_train_val() function
            return (self.indices[i] for i in torch.randperm(len(self.indices))) #shuffle the indices and return them


    trainloader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=12, shuffle=True) #use pytorch data loader to get the batches of training set
    valloader = data.DataLoader(val_set, batch_size=args.batch_size, num_workers=12) #use pytorch data loader to get the batches of validation set


    # Setup Metrics
    running_metrics = runningScore(n_classes) #initialize class instance for evaluation metrics for training
    running_metrics_val = runningScore(n_classes) #initialize class instance for evaluation meterics for validation


    # Setup Model
    if args.resume is not None: #Check if we have a stored model or not
        if os.path.isfile(args.resume): #if yes then load the stored model
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            model = torch.load(args.resume)
        else:
            print("No checkpoint found at '{}'".format(args.resume)) #if stored model requested with invalid path
    else: #if  no stord model then load the requested model
        #n_classes=64
        model = get_model(name=args.arch, pretrained=args.pretrained,batch_size=args.batch_size,growth_rate=32,drop_rate=0,n_classes=n_classes) #get the stored model





    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))     #Use as many GPUs as we can
    model = model.to(device)  # Send to GPU


    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        print('Using custom optimizer')
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr = args.lr,amsgrad=True,
                                     weight_decay=args.weight_decay,
                                     eps=args.eps) #if no specified optimizer then load the defualt optimizer


    loss_fn = core.loss.focal_loss2d #initialize a function loss function

    if args.class_weights: #if class weights are to be used then intailize them
        # weights are inversely proportional to the frequency of the classes in the training set
        class_weights = torch.tensor([0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852], device=device, requires_grad=False)
    else:
        class_weights = None #if no class weights then no need to use them

    best_iou = -100.0
    class_names = ['null', 'upper_ns', 'middle_ns', 'lower_ns', 'rijnland_chalk', 'scruff', 'zechstein'] #initialize the name of different classes

    for arg in vars(args): #Before training start writting the summary of the parameters
        text = arg + ': ' + str(getattr(args, arg)) #get the attribute name and value, make them as string
        writer.add_text('Parameters/', text) #store the whole string

    # training
    for epoch in range(args.n_epoch): #for loop on the number of epochs
        # Training Mode:
        model.train() #initialize training mode
        loss_train, total_iteration = 0, 0 # intialize training loss and total number of iterations

        for i, (images, labels) in enumerate(trainloader): #start the epoch then initialize the number of iterations per epoch i is the batch number
            image_original, labels_original = images, labels #store the image and label batch in new varaibles
            images, labels = images.to(device), labels.to(device) #move images and labels to the GPU

            optimizer.zero_grad() #intialize the optimizer
            outputs = model(images) #feed forward the images through the model (outputs is a 7 channel o/p)

            pred = outputs.detach().max(1)[1].cpu().numpy() #get the model o/p from GPU, select the index of the maximum channel and send it back to CPU
            gt = labels.detach().cpu().numpy()  #get the true lablels from GPU and send them to CPU
            running_metrics.update(gt, pred) #call the function update and pass the ground truth and the predicted classes

            loss = loss_fn(input=outputs, target=labels, gamma=args.gamma, loss_type=args.loss_parameters) #call the loss fuction to calculate the loss
            loss_train += loss.item() #gets the scalar value held in the loss.
            loss.backward() # Use autograd to compute the backward pass. This call will compute the gradient of loss with respect to all Tensors with requires_grad=True.


            # gradient clipping
            if args.clip != 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip) #The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.




            optimizer.step() #step the optimizer (update the model weights with the new gradients)
            total_iteration = total_iteration + 1 #increment the total number of iterations by 1

            if (i) % 20 == 0: #if 20% of the total number of iterations pass then
                print("Epoch [%d/%d] training Loss: %.4f" % (epoch + 1, args.n_epoch, loss.item())) #print the current epoch, total number of epochs and the current training loss

            numbers = [0, 14, 29, 49, 99] #select some numbers
            if i in numbers:  #if the current batch number is in numbers
                # number 0 image in the batch
                tb_original_image = vutils.make_grid(image_original[0][0], normalize=True, scale_each=True) #select the first image in the batch create a tensorboard grid form the image tensor
                writer.add_image('train/original_image', tb_original_image, epoch + 1) #send the image to writer

                labels_original = labels_original.numpy()[0]  #convert the ground truth lablels of the first image in the batch to numpy array
                correct_label_decoded = train_set.decode_segmap(np.squeeze(labels_original)) #Decode segmentation class labels into a color image
                writer.add_image('train/original_label', np_to_tb(correct_label_decoded), epoch + 1)  #send the image to the writer
                out = F.softmax(outputs, dim=1) #softmax of the network o/p
                prediction = out.max(1)[1].cpu().numpy()[0] #get the index of the maximum value after softmax
                confidence = out.max(1)[0].cpu().detach()[0] # this returns the confidence in the chosen class

                tb_confidence = vutils.make_grid(confidence, normalize=True, scale_each=True) #convert the confidence from tensor to image

                decoded = train_set.decode_segmap(np.squeeze(prediction)) #Decode predicted classes to colours
                writer.add_image('train/predicted', np_to_tb(decoded), epoch + 1) #send predicted map to writer along with the epoch number
                writer.add_image('train/confidence', tb_confidence, epoch + 1) #send the confidence to writer along with the epoch number




                unary = outputs.cpu().detach() #get the Nw o/p for the whole batch
                unary_max = torch.max(unary) #normalize the Nw o/p w.r.t whole batch
                unary_min = torch.min(unary)
                unary = unary.add((-1 * unary_min))
                unary = unary / (unary_max - unary_min)

                for channel in range(0, len(class_names)):
                    decoded_channel = unary[0][channel] #get the normalized o/p for the first image in the batch
                    tb_channel = vutils.make_grid(decoded_channel, normalize=True, scale_each=True) #prepare a image from tensor
                    writer.add_image(f'train_classes/_{class_names[channel]}', tb_channel, epoch + 1) #send image to writer




        # Average metrics after finishing all batches for the whole epoch, and save in writer()
        loss_train /= total_iteration #total loss for all iterations/ number of iterations
        score, class_iou = running_metrics.get_scores() #returns a dictionary of the calculated accuracy metrics and class iu
        writer.add_scalar('train/Pixel Acc', score['Pixel Acc: '], epoch + 1) # store the epoch metrics in the tensorboard writer
        writer.add_scalar('train/Mean Class Acc',score['Mean Class Acc: '], epoch + 1)
        writer.add_scalar('train/Freq Weighted IoU',score['Freq Weighted IoU: '], epoch + 1)
        writer.add_scalar('train/Mean_IoU', score['Mean IoU: '], epoch + 1)
        confusion = score['confusion_matrix']
        writer.add_image(f'train/confusion matrix', np_to_tb(confusion), epoch + 1)

        running_metrics.reset() #resets the confusion matrix
        writer.add_scalar('train/loss', loss_train, epoch + 1) #store the training loss
        #Finished one epoch of training, starting one epoch of testing
        if args.per_val != 0: # if validation is required
            with torch.no_grad():  # operations inside don't track history
                # Validation Mode:
                model.eval() #start validation mode
                loss_val, total_iteration_val = 0, 0 # initialize validation loss and total number of iterations

                for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)): #start validation testing
                    image_original, labels_original = images_val, labels_val #store original validation errors
                    images_val, labels_val = images_val.to(device), labels_val.to(device) #send validation images and labels to GPU

                    outputs_val = model(images_val) #feedforward the image
                    pred = outputs_val.detach().max(1)[1].cpu().numpy() #get the network class prediction
                    gt = labels_val.detach().cpu().numpy() #get the ground truth from the GPU

                    running_metrics_val.update(gt, pred) #run metrics on the validation data

                    loss = loss_fn(input=outputs_val, target=labels_val, gamma=args.gamma,loss_type=args.loss_parameters) #calculate the loss function
                    total_iteration_val = total_iteration_val + 1 #increment the loop counter

                    if (i_val) % 20 == 0: #After 20% of batches for validation print the validation loss
                        print("Epoch [%d/%d] validation Loss: %.4f" %(epoch, args.n_epoch, loss.item()))

                    numbers = [0]
                    if i_val in numbers: #select batch number 0
                        # number 0 image in the batch
                        tb_original_image = vutils.make_grid(image_original[0][0], normalize=True, scale_each=True) #make first tensor in the batch as image
                        writer.add_image('val/original_image',tb_original_image, epoch) #send image to writer
                        labels_original = labels_original.numpy()[0] #get origianl labels of image 0
                        correct_label_decoded = train_set.decode_segmap(np.squeeze(labels_original)) #convert the labels to colour map
                        writer.add_image('val/original_label', np_to_tb(correct_label_decoded), epoch + 1) #send the coloured map to writer

                        out = F.softmax(outputs_val, dim=1) #get soft max of the network 7 channel o/p

                        # this returns the max. channel number:
                        prediction = out.max(1)[1].cpu().detach().numpy()[0] #get the position of the max o/p across different channels
                        # this returns the confidence:
                        confidence = out.max(1)[0].cpu().detach()[0] #get the maximum o/p of the Nw across different channels
                        tb_confidence = vutils.make_grid(confidence, normalize=True, scale_each=True) #convert tensor to image

                        decoded = train_set.decode_segmap(np.squeeze(prediction)) #convert predicted classes to colour maps
                        writer.add_image('val/predicted', np_to_tb(decoded), epoch + 1) #send prediction to writer
                        writer.add_image('val/confidence', tb_confidence, epoch + 1) #send confidence to writer

                        unary = outputs.cpu().detach() #get Nw o/p of the current batch
                        unary_max, unary_min = torch.max(unary), torch.min(unary) #normalize across all the Nw o/p
                        unary = unary.add((-1 * unary_min))
                        unary = unary / (unary_max - unary_min)

                        for channel in range(0, len(class_names)): #for all the 7 channels of the Nw op
                            tb_channel = vutils.make_grid(unary[0][channel], normalize=True, scale_each=True) #convert the channel o/p of the class to image
                            writer.add_image(f'val_classes/_{class_names[channel]}', tb_channel, epoch + 1) #send image to writer
                # finished one cycle of validation after iterating over all validation batched
                score, class_iou = running_metrics_val.get_scores() #returns a dictionary of the calculated accuracy metrics and class iu
                for k, v in score.items(): #??
                    print(k, v)

                writer.add_scalar('val/Pixel Acc', score['Pixel Acc: '], epoch + 1) #send metrics to writer
                writer.add_scalar('val/Mean IoU', score['Mean IoU: '], epoch + 1)
                writer.add_scalar('val/Mean Class Acc', score['Mean Class Acc: '], epoch + 1)
                writer.add_scalar('val/Freq Weighted IoU', score['Freq Weighted IoU: '], epoch + 1)
                confusion = score['confusion_matrix']
                writer.add_image(f'val/confusion matrix', np_to_tb(confusion), epoch + 1)
                writer.add_scalar('val/loss', loss.item(), epoch + 1)
                running_metrics_val.reset() #reset confusion matrix


                if score['Mean IoU: '] >= best_iou: #compare with the validation mean iou of current epoch with the best stored validation mean IoU
                    best_iou = score['Mean IoU: '] #if better, then store the better and store the current model as the best model
                    model_dir = os.path.join(log_dir, f"{args.arch}_model_best.pkl")
                    torch.save(model, model_dir)

                if epoch % 10 == 0: #every 10 epochs store the current model
                    model_dir = os.path.join(log_dir, f"{args.arch}_ep{epoch}_model.pkl")
                    torch.save(model, model_dir)


        else:  # validation is turned off:
            # just save the latest model every 10 epochs:
            if (epoch + 1) % 10 == 0:
                model_dir = os.path.join(log_dir, f"{args.arch}_ep{epoch + 1}_model.pkl")
                torch.save(model, model_dir)

    writer.close() #close the writer


parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--arch', nargs='?', type=str, default='DenseNet',
                    help='Architecture to use [\'DenseNet\',\'Inception\',\'ResNet\',\'ResNext\']')
parser.add_argument('--n_epoch', nargs='?', type=int, default=40,
                    help='# of the epochs')
parser.add_argument('--batch_size', nargs='?', type=int, default=10,
                    help='Batch Size')
parser.add_argument('--resume', nargs='?', type=str, default=None,
                    help='Path to previous saved model to restart from')
parser.add_argument('--pretrained', nargs='?', type=bool, default=False,
                    help='Pretrained models not supported.')
parser.add_argument('--per_val', nargs='?', type=bool, default=True,
                    help='Do validation or not. keep true.')
parser.add_argument('--model_name', nargs='?', type=str, default='Inception',
                    help='name to associate with the model')
parser.add_argument('--lr',nargs='?', type=float, default=0.0005,
                    help='Learning rate')
parser.add_argument('--weight_decay',nargs='?', type=float,default=0.0,help='Weight decay (L2 penalty)')
parser.add_argument('--eps',nargs='?',type=float,default=1e-08,
                    help='Term added to the denominator to improve numerical stability')
args = parser.parse_args()
train(args)



