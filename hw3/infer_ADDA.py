import torch
import numpy as np
import pandas as pd

import lib.parser as parser
import lib.model_ADDA as model_ADDA
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
    my_model = {
        'E': model_ADDA.E(),
        'C': model_ADDA.C(),
    }

    if use_cuda:
        for _, m in my_model.items():
            m.cuda()
            m.eval()

    checkpoint = torch.load(args.pretrained)
    my_model['E'].load_state_dict(checkpoint['E'])

    pretrained_src = torch.load(args.pretrained_src)
    my_model['C'].load_state_dict(pretrained_src['C'])

    name_list = []
    pred_list = []
    # gt_list = []

    with torch.no_grad():  # do not need to calculate information for gradient during eval
        for idx, (imgs, img_names) in enumerate(val_loader):
            if use_cuda:
                # imgs, gt = imgs.cuda(), gt.cuda()
                imgs = imgs.cuda()
            out = my_model['C'](my_model['E'](imgs))
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
