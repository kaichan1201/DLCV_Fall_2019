import os
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms

import parser
import models
import data
from trainer import evaluate

transform = transforms.Compose([
                               transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                               ])


if __name__ == '__main__':
    
    args = parser.arg_parse()

    print('===> Loading infer image...')
    img = cv2.imread(args.infer_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = img.view(1, *img.shape)

    print('===> Preparing model...')
    model = models.get_model(args)

    checkpoint = torch.load(args.pretrained)
    model.load_state_dict(checkpoint)

    print('===> Inferring...')
    model.eval()
    out = model(img)
    _, seg = torch.max(out, dim=1)

    print('===> Saving inferred image to infer.png...')
    cv2.imwrite('infer.png', seg.numpy().squeeze())
