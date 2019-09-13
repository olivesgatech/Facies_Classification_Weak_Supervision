import argparse
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import torchvision.utils as vutils
from core.loader.data_loader import *
from core.metrics import runningScore
from core.utils import test, test_time_augmentation

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

torch.backends.cudnn.deterministic = True
torch.manual_seed(2019)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(2019)
np.random.seed(seed=2019)


def np_to_tb(array):
    # if 2D :
    if array.ndim == 2:
        # HW => CHW
        array = np.expand_dims(array,axis=0)
        # CHW => NCHW
        array = np.expand_dims(array,axis=0)
    elif array.ndim == 3:
        # HWC => CHW
        array = array.transpose(2, 0, 1)
        # CHW => NCHW
        array = np.expand_dims(array,axis=0)
    
    array = torch.from_numpy(array)
    array = vutils.make_grid(array, normalize=True, scale_each=True)
    return array

def patch_label_2d(model, img, patch_size, stride, testing):
    img = torch.squeeze(img)
    h, w = img.shape  # height and width

    # Pad image with patch_size/2:
    ps = int(np.floor(patch_size/2))  # pad size
    img_p = F.pad(img, pad=(ps, ps, ps, ps), mode='constant', value=0) # padded image

    num_classes = 7
    output_p = torch.zeros([1, num_classes, h+2*ps, w+2*ps])

    # generate output:
    for hdx in range(0, h+2*ps-patch_size, stride):
        for wdx in range(0, w+2*ps-patch_size, stride):
            patch = img_p[hdx: hdx + patch_size, wdx: wdx + patch_size]
            patch = patch.unsqueeze(dim=0) # channel dim
            patch = patch.unsqueeze(dim=0) # batch dim

            assert (patch.shape == (1, 1, patch_size, patch_size))
            model_output = testing(model,patch)
            output_p[:, :, hdx: hdx + patch_size, wdx: wdx + patch_size] += torch.squeeze(model_output.detach().cpu())

    # crop the output_p in the middle
    output = output_p[:, :, ps:-ps, ps:-ps]
    return output


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.tta:
        testing_code = test_time_augmentation
    else:
        testing_code = test

    log_dir, model_name = os.path.split(args.model_path)
    # load model:
    model = torch.load(args.model_path)
    model = model.to(device)  # Send to GPU if available
    writer = SummaryWriter(log_dir=log_dir)

    class_names = ['null', 'upper_ns', 'middle_ns', 'lower_ns', 'rijnland_chalk', 'scruff', 'zechstein']
    running_metrics_overall = runningScore(7)

    splits = [args.split if 'both' not in args.split else 'test1', 'test2']

    for sdx, split in enumerate(splits):
        # define indices of the array
        labels = np.load(pjoin('data', 'test_once', split + '_labels.npy'))
        irange, xrange, depth = labels.shape

        if split == 'test1':
            result_volume = torch.zeros([7,200,701,255], dtype=torch.float32, device='cpu', requires_grad=False)
        elif split == 'test2':
            result_volume = torch.zeros([7,601,200,255], dtype=torch.float32, device='cpu', requires_grad=False)

        if args.inline:
            i_list = list(range(irange))
            i_list = ['i_'+str(inline) for inline in i_list]
        else:
            i_list = []

        if args.crossline:
            x_list = list(range(xrange))
            x_list = ['x_'+str(crossline) for crossline in x_list]
        else:
            x_list = []

        num_iline = len(i_list) 
        num_xline = len(x_list) 

        list_test = i_list + x_list

        file_object = open(
            pjoin('data', 'splits', 'section_' + split + '.txt'), 'w')
        file_object.write('\n'.join(list_test))
        file_object.close()

        test_set = section_loader(is_transform=True,
                                  split=split,
                                  augmentations=None)
        n_classes = test_set.n_classes

        test_loader = data.DataLoader(test_set,
                                      batch_size=1,
                                      num_workers=8,
                                      shuffle=False)

        running_metrics_split = runningScore(n_classes)

        # testing mode:
        with torch.no_grad():  # operations inside don't track history
            model.eval()
            total_iteration = 0
            for i, (imgs, lbls) in enumerate(test_loader):
                print(f'split: {split}, section: {i}')
                total_iteration = total_iteration + 1


                # get sections labaled (7 channels/sections outputed for each section)
                outputs = patch_label_2d(model=model,
                                         img=imgs,
                                         patch_size=args.train_patch_size,
                                         stride=args.test_stride, 
                                         testing=testing_code)

                # detach, send to cpu and add to corresponding location: 
                update = outputs.detach().cpu()
                # this is the tricky part: (how to know where to add it:)
                if split == 'test1':
                    if i < num_iline and args.inline: # inline -- split 1
                        assert update.shape == torch.Size([1,7,255,701]), 'Hmm, something is wrong.'
                        result_volume[:,i,:,:] += update.squeeze().permute((0, 2, 1))
                    elif i < num_iline+num_xline and args.crossline: # crossline -- split 1
                        assert update.shape == torch.Size([1,7,255,200]), 'Hmm, something is wrong.'
                        result_volume[:,:,i-num_iline,:] += update.squeeze().permute((0, 2, 1))
                    else: # ???
                        raise ValueError('Something is wrong with the value of i')
                elif split == 'test2':
                    if i < num_iline and args.inline: # inline -- split 2
                        assert update.shape == torch.Size([1,7,255,200]), 'Hmm, something is wrong.'
                        result_volume[:,i,:,:] += update.squeeze().permute((0, 2, 1))
                    elif i < num_iline+num_xline and args.crossline: # crossline -- split 2
                        assert update.shape == torch.Size([1,7,255,601]), 'Hmm, something is wrong.' 
                        result_volume[:,:,i-num_iline,:] += update.squeeze().permute((0, 2, 1))
                    else: # ???
                        raise ValueError('Something is wrong with the value of i')
                    
                if i == num_iline - 1:  # last iteration in inline: 
                    np.save(pjoin(log_dir,f'volume_split_{split}_iline.npy'), result_volume) 
                elif i == num_iline+num_xline - 1:  # last iteration in crossline: 
                    np.save(pjoin(log_dir,f'volume_split_{split}_xline.npy'), result_volume) 


            # FLATTEN THE VOLUMES (GT AND PRED), AND compute the metrics:
            final_volume = result_volume.max(0)[1].numpy() 
            pred = final_volume.flatten()
            gt = labels.flatten() + 1 # make 1-indexed like pred
            running_metrics_split.update(gt, pred)
            running_metrics_overall.update(gt, pred)

            # SAVE THE RESULTING LABELS AS NP NDARRAY: (TO VISUALIZE RESULT LATER)
            np.save(pjoin(log_dir,f'final_volume_split_{split}.npy'), final_volume) 


        # ------------------------- Split sdx is done ------------------------- 

        # get scores and save in writer()
        score, class_iou = running_metrics_split.get_scores()

        # Add split results to TB: 
        writer.add_text(f'test__{split}/', f'Pixel Acc: {score["Pixel Acc: "]:.3f}',0)
        for cdx, class_name in enumerate(class_names[1:]):
            writer.add_text(f'test__{split}/', f'  {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}',0)

        writer.add_text(f'test__{split}/', f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}',0)
        writer.add_text(f'test__{split}/', f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}',0)
        writer.add_text(f'test__{split}/', f'Mean IoU: {score["Mean IoU: "]:0.3f}',0)
        running_metrics_split.reset()

    # FINAL TEST RESULTS:
    score, class_iou = running_metrics_overall.get_scores()

    # Add split results to TB: 
    writer.add_text('test_final', f'Pixel Acc: {score["Pixel Acc: "]:.3f}',0)
    for cdx, class_name in enumerate(class_names[1:]):
        writer.add_text('test_final', f'  {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}',0)

    writer.add_text(
        'test_final', f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}', 0)
    writer.add_text(
        'test_final', f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}', 0)
    writer.add_text('test_final', f'Mean IoU: {score["Mean IoU: "]:0.3f}', 0)
    writer.close()

    print('--------------- FINAL RESULTS -----------------')
    print(f'Pixel Acc: {score["Pixel Acc: "]:.3f}')
    for cdx, class_name in enumerate(class_names[1:]):
        print(f'     {class_name}_accuracy {score["Class Accuracy: "][cdx]:.3f}')
    print(f'Mean Class Acc: {score["Mean Class Acc: "]:.3f}')
    print(f'Freq Weighted IoU: {score["Freq Weighted IoU: "]:.3f}')
    print(f'Mean IoU: {score["Mean IoU: "]:0.3f}')

    confusion = score['confusion_matrix']
    np.savetxt(pjoin(log_dir,'confusion.csv'), confusion, delimiter=" ") 

    proper_class_names = ['Upper N.S.', 'Middle N.S.', 'Lower N.S.', 'Rijnland/Chalk', 'Scruff', 'Zechstein']
    # normalize confidence matrix:
    confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(confusion, index = [i for i in proper_class_names],
                    columns = [i for i in proper_class_names])
    plt.figure(figsize = (10,7))
    ax = sn.heatmap(df_cm, annot=True)
    fig = ax.get_figure()
    fig.savefig(pjoin(log_dir,'confusion.png'), dpi=300)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='runs/Apr17_161626_patch_deconvnet_skip_test/patch_deconvnet_skip_model_best.pkl',
                        help='Path to the saved model')
    parser.add_argument('--split', nargs='?', type=str, default='both',
                        help='Choose from: "test1", "test2", or "both" to change which region to test on')
    parser.add_argument('--crossline', nargs='?', type=bool, default=True,
                        help='whether to test in crossline mode')
    parser.add_argument('--inline', nargs='?', type=bool, default=True,
                        help='whether to test inline mode')
    parser.add_argument('--tta', nargs='?', type=bool, default=True,
                        help='whether to use Test Time Augmentation')
    parser.add_argument('--train_patch_size', nargs='?', type=int, default=75,
                        help='The size of the patches that were used for training.'\
                        'This must be correct, or will cause errors.')
    parser.add_argument('--test_stride', nargs='?', type=int, default=5,
                        help='The size of the stride of the sliding window at test time.'\
                        'The smaller, the better the results, but the slower they are computed.')
    args = parser.parse_args()
    test(args)
