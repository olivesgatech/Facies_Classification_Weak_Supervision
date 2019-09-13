import argparse
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import torchvision.utils as vutils
from core.loader.data_loader import *
from core.metrics import runningScore

def np_to_tb(array): #convert from numpy array to tensorboard image
    # if 2D :
    if array.ndim == 2: #if array is 2D
        # HW => CHW
        array = np.expand_dims(array,axis=0) #expand it to 3D
        # CHW => NCHW
        array = np.expand_dims(array,axis=0) #expand it even further to 4D
    elif array.ndim == 3: #if array is 3D
        # HWC => CHW
        array = array.transpose(2, 0, 1)  # transpose it
        # CHW => NCHW
        array = np.expand_dims(array,axis=0) #expand it to 4D
    
    array = torch.from_numpy(array) #convert np to tensor
    array = vutils.make_grid(array, normalize=True, scale_each=True) #convert tensor to image
    return array

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #get device name

    log_dir, model_name = os.path.split(args.model_path) #split the model directory
    # load model:
    model = torch.load(args.model_path)  #load model
    model = model.to(device)  # Send to GPU if available
    writer = SummaryWriter(log_dir=log_dir) #open summary writer

    class_names = ['null', 'upper_ns', 'middle_ns', 'lower_ns', 'rijnland_chalk', 'scruff', 'zechstein'] #class names
    running_metrics_overall = runningScore(6) #ToDo

    splits = [args.split if 'both' not in args.split else 'test1', 'test2'] #check if both tests are required
    for sdx, split in enumerate(splits): # sdx: test index, split name (For loop on the number of tests)
        # define indices of the array
        labels = np.load(pjoin('data', 'test_once', split + '_labels.npy')) #load labels of the required test
        irange, xrange, depth = labels.shape #get the number of test images in that test

        if args.inline: # if inline mode is required
            i_list = list(range(irange)) #create a list for inline indces
            i_list = ['i_'+str(inline) for inline in i_list] #create a list of inline names
        else:
            i_list = [] #else send an empty list

        if args.crossline: #if cross lines are required
            x_list = list(range(xrange)) #create a list of cross line indces
            x_list = ['x_'+str(crossline) for crossline in x_list] #create a list of cross line names
        else:
            x_list = []

        list_test = i_list + x_list #combine inline and crossline indces

        file_object = open(pjoin('data', 'splits', 'section_' + split + '.txt'), 'w') #open a text file  with the name of the test in write mode
        file_object.write('\n'.join(list_test)) #write the list in the file
        file_object.close() #close the list

        test_set = section_loader(is_transform=True,split=split,augmentations=None) #call the costume made data loader
        n_classes = test_set.n_classes #set the number of classes
        test_loader = data.DataLoader(test_set,batch_size=1,num_workers=4,shuffle=False)#ToDo: batch size equal 1, each section is a batch by itself #call pytorch data loader and pass in the costume made data loader with the overwritten methodes

        # print the results of this split:
        running_metrics_split = runningScore(n_classes) #function calling #ToDo

        # testing mode: Start testind
        with torch.no_grad():  #stop gradient tarcking
            model.eval() #start evaluation mode
            total_iteration = 0 #iteration variable
            for i, (images, labels) in enumerate(test_loader): #load batches one by one
                #print(labels.shape)
                print(f'split: {split}, section: {i}')  #print test name and section number
                total_iteration = total_iteration + 1 #increment the total number of iterations
                image_original, labels_original = images, labels #copy images and labels
                images, labels = images.to(device), labels.to(device) #move images and labels to GPU
                outputs = model(images) #Feed forward the image to the model
                pred = outputs.detach().max(1)[1].cpu().numpy() #get the predicted class
                #print('iteration',i,'images',images.shape,'labels',labels,'prediction', pred)
                gt = labels.detach().cpu().numpy() #get the ground truth labels


                running_metrics_split.update(gt, pred) #send the predicted class and the labels to metrics intialized on 7 classes
                running_metrics_overall.update(gt, pred)  #send the predicted class and the labels to metrics intialized on 6 classes

                numbers = [0, 99, 149, 399, 499] #images to consider during testing
                if i in numbers:
                    tb_original_image = vutils.make_grid(image_original[0][0], normalize=True, scale_each=True) #convert tensor to image
                    writer.add_image('test/original_image',tb_original_image, i) #send image to writer
                
                    labels_original=labels_original.numpy()[0] #get the ground truth labels
                    correct_label_decoded = test_set.decode_segmap(np.squeeze(labels_original)) #get the color map of the ground truth
                    writer.add_image('test/original_label',np_to_tb(correct_label_decoded), i) #send the color map to writer
                    out = F.softmax(outputs, dim=1) #do soft max on Nw op

                    # this returns the max. channel number:
                    prediction = out.max(1)[1].cpu().numpy()[0] #get the predicions
                    # this returns the confidence:
                    confidence = out.max(1)[0].cpu().detach()[0] #get the confidence
                    tb_confidence = vutils.make_grid(confidence, normalize=True, scale_each=True) #convert the Nw op to confidence to Image

                    decoded = test_set.decode_segmap(np.squeeze(prediction)) #get the colour map of the prediction
                    writer.add_image('test/predicted', np_to_tb(decoded), i) #send colour map to writer
                    writer.add_image('test/confidence', tb_confidence, i) #send confidence to writer

                    # uncomment if you want to visualize the different class heatmaps
                    # unary = outputs.cpu().detach()
                    # unary_max = torch.max(unary)
                    # unary_min = torch.min(unary)
                    # unary = unary.add((-1*unary_min))
                    # unary = unary/(unary_max - unary_min)

                    # for channel in range(0, len(class_names)):
                    #     decoded_channel = unary[0][channel]
                    #     tb_channel = vutils.make_grid(decoded_channel, normalize=True, scale_each=True)
                    #     writer.add_image(f'test_classes/_{class_names[channel]}', tb_channel, i)

        # get scores and save in writer()
        #after finishing one test
        score, class_iou = running_metrics_split.get_scores() #ToDo:
        #print('score',score)
        # Add split results to TB:
        writer.add_text(f'test__{split}/',f'Pixel Acc: {score["Pixel Acc: "]:.3f}', 0) #send scores to writer
        for cdx, class_name in enumerate(class_names[1:]):
            #print('index',cdx,'class name',class_name)
            writer.add_text(f'test__{split}/', f'  {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}', 0) #send individual  class scores to the writer

        writer.add_text(f'test__{split}/', f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}',0) #send averages to the writer
        writer.add_text(f'test__{split}/', f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}',0)
        writer.add_text(f'test__{split}/', f'Mean IoU: {score["Mean IoU: "]:0.3f}',0)
        confusion = score['confusion_matrix']
        writer.add_image(f'test/confusion matrix', np_to_tb(confusion), 0)

        running_metrics_split.reset() #clear the confusion matrix
    #after finishing both tests
    # FINAL TEST RESULTS:
    score, class_iou = running_metrics_overall.get_scores() # get scores of both tests

    # Add split results to TB:
    writer.add_text('test_final', f'Pixel Acc: {score["Pixel Acc: "]:.3f}', 0) #send results to writer
    for cdx, class_name in enumerate(class_names[1:]):
        writer.add_text('test_final', f'  {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}', 0) #send individual class scores to writer

    writer.add_text('test_final', f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}', 0) #send average test results to the writer
    writer.add_text('test_final', f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}', 0)
    writer.add_text('test_final', f'Mean IoU: {score["Mean IoU: "]:0.3f}', 0)
    writer.add_text('test_final', f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}',0)
    writer.add_text('test_final', f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}',0)
    writer.add_text('test_final', f'Mean IoU: {score["Mean IoU: "]:0.3f}',0)
    confusion = score['confusion_matrix']
    writer.add_image(f'test/FINAL confusion matrix', np_to_tb(confusion),0)

    print('--------------- FINAL RESULTS -----------------') #print the results
    print(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
    for cdx, class_name in enumerate(class_names[1:]):
        print(f'     {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}')
    print(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
    print(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
    print(f'Mean IoU: {score["Mean IoU: "]:0.3f}')

    confusion = score['confusion_matrix']
    np.savetxt(pjoin(log_dir,'confusion.csv'), confusion, delimiter=" ") #save confusion Mtx as text

    writer.close() #close writer
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='runs/Mar29_111105_patch_deconvnet_skip_section_full_skip_CE_noaug_baseline/patch_deconvnet_skip_model_best.pkl',
                        help='Path to the saved model')
    parser.add_argument('--split', nargs='?', type=str, default='both',
                        help='Choose from: "test1", "test2", or "both" to change which region to test on')
    parser.add_argument('--crossline', nargs='?', type=bool, default=True,
                        help='whether to test in crossline mode')
    parser.add_argument('--inline', nargs='?', type=bool, default=True,
                        help='whether to test inline mode')
    args = parser.parse_args()
    test(args)
