import torch
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compute_ssim

def psnr_metric(img_pred, img_gt):
    mse = np.mean((img_pred - img_gt)**2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr

def ssim_metric(img_pred, img_gt, mask_at_box):
    x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
    img_pred = img_pred[y:y + h, x:x + w]
    img_gt = img_gt[y:y + h, x:x + w]

    # compute the ssim
    ssim = compute_ssim(img_pred, img_gt, multichannel=True)
    return ssim

def lpips_metric(img_pred, img_gt, mask_at_box, loss_fn_vgg, device):
    x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
    img_pred = img_pred[y:y + h, x:x + w]
    img_gt = img_gt[y:y + h, x:x + w]

    # compute the lpips
    img_pred = torch.tensor(img_pred, dtype=torch.float32, device=device).reshape(1, h, w, 3).permute(0, 3, 1, 2)
    img_gt = torch.tensor(img_gt, dtype=torch.float32, device=device).reshape(1, h, w, 3).permute(0, 3, 1, 2)

    score = loss_fn_vgg(img_pred, img_gt, normalize=True)
    return score.item()
