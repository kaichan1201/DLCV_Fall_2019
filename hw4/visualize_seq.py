import os
import cv2
import numpy as np

pred_dir = './results/'
gt_list_dir = './hw4_data/FullLengthVideos/labels/valid/'
video_list_dir = './hw4_data/FullLengthVideos/videos/valid/'

n_classes = 11
frame_num = 10
color_map = {
    0: np.array([0, 0, 0]),
    1: np.array([0, 255, 0]),
    2: np.array([0, 0, 255]),
    3: np.array([255, 255, 0]),
    4: np.array([255, 0, 255]),
    5: np.array([0, 255, 255]),
    6: np.array([128, 0, 0]),
    7: np.array([0, 128, 0]),
    8: np.array([0, 0, 128]),
    9: np.array([128, 128, 0]),
    10: np.array([128, 0, 128]),
}


def draw_bar(lbls, out_w, out_h=35):
    if out_w < lbls.shape[0]:
        lbls = lbls[:out_w]
        bar = np.zeros((out_h, out_w, 3))
    else:
        bar = np.zeros((out_h, lbls.shape[0], 3))

    for i in range(n_classes):
        idx = np.where(lbls == i)
        bar[:, idx, :] = color_map[i]

    if out_w > lbls.shape[0]:
        bar = cv2.resize(bar, dsize=(out_w, out_h))
    return bar


def sample_rescale_frames(video_dir, out_w):
    """return frame_num frames concatenated side by side, rescaled to a certain width"""
    frame_id_list = os.listdir(video_dir)
    idx_list = np.rint(np.arange(frame_num) / frame_num * len(frame_id_list)).astype(int)

    frame = cv2.imread(os.path.join(video_dir, frame_id_list[0]))
    h, w, _ = frame.shape
    frame_w = out_w // frame_num
    frame_h = int(h / w * frame_w)

    frames = []
    for i in idx_list:
        frame = cv2.imread(os.path.join(video_dir, frame_id_list[i]))
        frame = cv2.resize(frame, dsize=(frame_w, frame_h))
        frames.append(frame)
    frames = np.concatenate(frames, axis=1)

    pad = 5
    frames_padded = np.full((frame_h + 2*pad, out_w, 3), 255)
    frames_padded[pad:frame_h+pad, :, :] = frames
    return frames_padded


def draw(pred, gt, video_dir):
    pred_bar = draw_bar(pred, out_w=1250)
    gt_bar = draw_bar(gt, out_w=1250)
    frames = sample_rescale_frames(video_dir, out_w=pred_bar.shape[1])
    return np.concatenate([pred_bar, frames, gt_bar], axis=0)


def create_block(color, w, pad):
    block = np.full((w+2*pad, w+2*pad, 3), 255)
    block[pad:w+pad, pad:w+pad, :] = color
    return block


def draw_legend():
    rows = []
    for i in range(3):
        row = []
        for j in range(4):
            if i == 2 and j == 3:
                block = create_block(np.array([255, 255, 255]), w=30, pad=30)
            else:
                block = create_block(color_map[i*4+j], w=30, pad=30)
            row.append(block)
        row = np.concatenate(row, axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)


if __name__ == '__main__':
    # find all pred txt files
    gt_list = sorted(os.listdir(gt_list_dir))
    for gt_file in gt_list:
        # read in pred and gt
        with open(os.path.join(pred_dir, gt_file)) as f:
            pred = f.read().splitlines()
            pred = np.array(list(map(int, pred)))
        with open(os.path.join(gt_list_dir, gt_file)) as f:
            gt = f.read().splitlines()
            gt = np.array(list(map(int, gt)))

        name = gt_file[:-4]
        # draw
        canvas = draw(pred, gt, video_dir=os.path.join(video_list_dir, name))
        cv2.imwrite(os.path.join('./vis', '{}.jpg'.format(name)), canvas)

    # plot legend
    legend = draw_legend()
    cv2.imwrite(os.path.join('./vis/legend.jpg'), legend)
