import os
import torch
import numpy as np
import pandas as pd

import lib.parser as parser
import lib.model as model
import lib.data as data

if __name__ == '__main__':

    args = parser.arg_parse()

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    # prepare data
    val_loader = torch.utils.data.DataLoader(data.DigitDataset_INFER(args),
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False)

    # prepare model
    if args.no_transfer:
        my_model = model.DANN_NoTransfer()
    else:
        my_model = model.DANN()

    if use_cuda:
        my_model.cuda()
    my_model.load_state_dict(torch.load(args.pretrained))

    name_list = []
    pred_list = []
    # gt_list = []

    my_model.eval()
    with torch.no_grad():  # do not need to calculate information for gradient during eval
        for idx, (imgs, img_names) in enumerate(val_loader):
            if use_cuda:
                # imgs, gt = imgs.cuda(), gt.cuda()
                imgs = imgs.cuda()
            if args.no_transfer:
                out = my_model(imgs)
            else:
                out, _ = my_model(imgs, 0)
            _, pred = torch.max(out, dim=1)

            name_list += img_names
            pred_list += pred.tolist()
            # gt_list += gt.view(-1).tolist()

        name_list = np.array(name_list)
        pred_list = np.array(pred_list)
        # gt_list = np.array(gt_list)

        pred_data = np.stack((name_list, pred_list), axis=0).T
        pred_df = pd.DataFrame(pred_data, columns=['image_name', 'label'])
        pred_df.to_csv(args.csv_path, index=False, columns=['image_name', 'label'])

        # gt_data = np.stack((name_list, gt_list), axis=0).T
        # gt_df = pd.DataFrame(gt_data, columns=['img_names', 'labels'])
        # gt_df.to_csv(args.csv_path, index=False, columns=['img_names', 'labels'])
