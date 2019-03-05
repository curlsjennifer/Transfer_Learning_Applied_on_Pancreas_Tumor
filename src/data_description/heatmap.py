import itertools

import numpy as np
from skimage import measure, morphology
import matplotlib.pyplot as plt

from data_loader.preprocessing import (minmax_normalization, smoothing,
                                       windowing)


def ceil_div(num, den):
    num = int(num)
    den = int(den)
    return (num + den - 1) // den


def get_score(img_shape, results, patch_size, stride):
    sp = img_shape
    num = (ceil_div(sp[0], stride[0]), ceil_div(sp[1], stride[1]), sp[2])
    # sp = (num[0] * stride[0], num[1] * stride[1], sp[2])

    score = np.zeros(sp)
    factor = np.zeros(sp)
    index = 0
    assert (num[0] * num[1] * num[2] == results.shape[0])
    for sx, sy, z in itertools.product(
            range(0, sp[0], stride[0]), range(0, sp[1], stride[1]),
            range(sp[2])):
        ex = int(np.min([sx + patch_size[0], sp[0]]))
        ey = int(np.min([sy + patch_size[1], sp[1]]))
        score[sx:ex, sy:ey, z] = score[sx:ex, sy:ey, z] + results[index]
        factor[sx:ex, sy:ey, z] = factor[sx:ex, sy:ey, z] + 1
        index = index + 1
    score[factor != 0] = score[factor != 0] / factor[factor != 0]
    return score


def slides_to_patches(pancreas, lesion, patch_size, stride):
    assert (pancreas.shape == lesion.shape)
    X = []
    y = []
    for sx, sy in itertools.product(
            range(0, pancreas.shape[0], stride[0]),
            range(0, pancreas.shape[1], stride[1])):
        img_pan = np.zeros(patch_size + (pancreas.shape[2],))
        mask_les = np.zeros(patch_size + (pancreas.shape[2],))
        ex = int(np.min([sx + patch_size[0], pancreas.shape[0]]))
        ey = int(np.min([sy + patch_size[1], pancreas.shape[1]]))
        img_pan[:ex - sx, :ey - sy, :] = pancreas[sx:ex, sy:ey, :]
        mask_les[:ex - sx, :ey - sy, :] = lesion[sx:ex, sy:ey, :]
        score = np.sum(mask_les, axis=(0, 1)) > 0
        X.append(img_pan)
        y.append(score.astype(int))
    return np.moveaxis(
        np.expand_dims(np.concatenate(X, axis=-1), axis=-1), 2,
        0), np.concatenate(
            y, axis=-1)


def split_to_patches(pan_list, les_list, patch_size, stride, includebg=True):
    X = []
    y = []
    for pan, les in zip(pan_list, les_list):
        for sx, sy in itertools.product(
                range(0, pan.shape[0], stride[0]),
                range(0, pan.shape[1], stride[1])):
            img_pan = np.zeros(patch_size + (pan.shape[2],))
            mask_les = np.zeros(patch_size + (pan.shape[2],))
            ex = int(np.min([sx + patch_size[0], pan.shape[0]]))
            ey = int(np.min([sy + patch_size[1], pan.shape[1]]))
            img_pan[:ex - sx, :ey - sy, :] = pan[sx:ex, sy:ey, :]
            mask_les[:ex - sx, :ey - sy, :] = les[sx:ex, sy:ey, :]
            idx = np.ones(img_pan.shape[2], dtype=np.bool)
            if not includebg:
                idx = np.sum(img_pan, axis=(0, 1)) > 0
            score = np.sum(mask_les, axis=(0, 1)) > 0
            X.append(img_pan[:, :, idx])
            y.append(score[idx].astype(int))
    return np.moveaxis(
        np.expand_dims(np.concatenate(X, axis=-1), axis=-1), 2,
        0), np.concatenate(
            y, axis=-1)


def split_to_seg_patches(pan_list, les_list, patch_size, stride,
                         includebg=True):
    X = []
    y = []
    for pan, les in zip(pan_list, les_list):
        for sx, sy in itertools.product(
                range(0, pan.shape[0], stride[0]),
                range(0, pan.shape[1], stride[1])):
            img_pan = np.zeros(patch_size + (pan.shape[2],))
            mask_les = np.zeros(patch_size + (pan.shape[2],))
            ex = int(np.min([sx + patch_size[0], pan.shape[0]]))
            ey = int(np.min([sy + patch_size[1], pan.shape[1]]))
            img_pan[:ex - sx, :ey - sy, :] = pan[sx:ex, sy:ey, :]
            mask_les[:ex - sx, :ey - sy, :] = les[sx:ex, sy:ey, :]
            idx = np.ones(img_pan.shape[2], dtype=np.bool)
            if not includebg:
                idx = np.sum(img_pan, axis=(0, 1)) > 0
            # score = np.sum(mask_les, axis=(0, 1)) > 0
            X.append(img_pan[:, :, idx])
            y.append(mask_les[:, :, idx])
    return np.moveaxis(
        np.expand_dims(np.concatenate(X, axis=-1), axis=-1), 2, 0), np.moveaxis(
            np.expand_dims(np.concatenate(y, axis=-1), axis=-1), 2, 0)


def load_slices(data_path, case_list):
    img_list = []
    pan_list = []
    les_list = []
    for ID in case_list:
        img_path = data_path + '/' + ID + '/ctscan.npy'
        pan_path = data_path + '/' + ID + '/pancreas.npy'
        pancreas = np.load(pan_path)
        pancreas = smoothing(pancreas)
        img = np.load(img_path)
        img = windowing(img)
        img = minmax_normalization(img)
        lesion = np.zeros(img.shape)
        if ID[:2] != 'NP' and ID[:2] != 'AD':
            les_path = data_path + '/' + ID + '/lesion.npy'
            lesion = np.load(les_path)
            pancreas[np.where(lesion == 1)] = 1

        # pancreas is union of pancreas and lesion by hands.
        img_pan = img * pancreas
        img_list.append(img)
        pan_list.append(img_pan)
        les_list.append(lesion)

    return img_list, pan_list, les_list


def heatmap(model,
            pancreas,
            lesion,
            patch_size,
            stride,
            is_padding=False,
            batch_size=128):

    X = pancreas
    Y = lesion
    if is_padding:
        X = np.pad(X, ((patch_size[0], patch_size[0]),
                       (patch_size[1], patch_size[1]), (0, 0)), 'constant')
        Y = np.pad(Y, ((patch_size[0], patch_size[0]),
                       (patch_size[1], patch_size[1]), (0, 0)), 'constant')
    X_patches, Y_patches = slides_to_patches(X, Y, patch_size, stride)
    ground_truth = get_score(X.shape, Y_patches, patch_size, stride)
    results = model.predict(X_patches, batch_size=batch_size)
    print("results shape = ", results.shape)
    score = get_score(X.shape, results, patch_size, stride)
    if is_padding:
        ground_truth = ground_truth[patch_size[0]:patch_size[0] + pancreas.
                                    shape[0], patch_size[1]:patch_size[1] +
                                    pancreas.shape[1], :]
        score = score[patch_size[0]:patch_size[0] +
                      pancreas.shape[0], patch_size[1]:patch_size[1] +
                      pancreas.shape[1], :]
    return ground_truth, score


def draw_heatmap(img,
                 pancreas,
                 ground_truth,
                 prediction,
                 z,
                 threshold,
                 ismask=False,
                 alpha=0.5):
    fig, axs = plt.subplots(1, 4, figsize=(16, 10))
    axs[0].imshow(img[:, :, z], cmap='gray')
    axs[1].imshow(img[:, :, z], cmap='gray')
    axs[2].imshow(img[:, :, z], cmap='gray')
    axs[3].imshow(img[:, :, z], cmap='gray')
    pan = pancreas[:, :, z]
    pan_mask = pan > 0
    pan_img = np.stack([
        np.zeros(pan_mask.shape),
        np.zeros(pan_mask.shape), pan, pan_mask * alpha
    ],
                       axis=-1)
    axs[0].imshow(pan_img)
    # gound truth use green
    gt = ground_truth[:, :, z]
    gt_mask = gt == 1
    if ismask:
        gt_mask = gt_mask * (pancreas[:, :, z] > 0)
    gt_img = np.stack([
        np.zeros(gt_mask.shape), gt_mask,
        np.zeros(gt_mask.shape), gt_mask * alpha
    ],
                      axis=-1)
    axs[1].imshow(gt_img)
    pd = prediction[:, :, z]
    pd_mask = pd >= threshold
    if ismask:
        pd_mask = pd_mask * (pancreas[:, :, z] > 0)
    pd_val = pd * pd_mask
    pd_img = np.stack([
        pd_val,
        np.zeros(pd_mask.shape),
        np.zeros(pd_mask.shape), pd_mask * alpha
    ],
                      axis=-1)
    axs[2].imshow(pd_img)

    pan_bd = morphology.dilation(pan_mask, np.ones((5, 5))) ^ pan_mask
    gt_bd = morphology.dilation(gt_mask, np.ones((5, 5))) ^ gt_mask
    pd_bd = morphology.dilation(pd_mask, np.ones((5, 5))) ^ pd_mask
    pan_bd_img = np.stack([
        np.zeros(pd_mask.shape),
        np.zeros(pd_mask.shape), pan_bd, pan_bd * alpha
    ],
                          axis=-1)
    gt_bd_img = np.stack(
        [np.zeros(gt_bd.shape), gt_bd,
         np.zeros(gt_bd.shape), gt_bd * alpha],
        axis=-1)
    pd_bd_img = np.stack(
        [pd_bd,
         np.zeros(pd_bd.shape),
         np.zeros(pd_bd.shape), pd_bd * alpha],
        axis=-1)
    axs[3].imshow(pan_bd_img)
    axs[3].imshow(gt_bd_img)
    axs[3].imshow(pd_bd_img)
    axs[2].imshow(pan_bd_img)
    axs[1].imshow(pan_bd_img)
