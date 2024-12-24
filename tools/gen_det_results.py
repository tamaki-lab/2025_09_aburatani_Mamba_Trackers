import numpy as np
import torch
import cv2 
from PIL import Image
from tqdm import tqdm

import argparse
import os

from time import gmtime, strftime

# from ultralytics import YOLO  # yolo v8
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.data.data_augment import preproc

from loguru import logger

def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', required=True, type=str, default='visdrone')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--data_root', required=True, type=str, default='/data/wujiapeng/datasets/VisDrone2019/VisDrone2019/images')
    parser.add_argument('--exp_file', required=True, type=str, default='')
    parser.add_argument('--model_path', required=True, type=str, default='weights/yolo')

    parser.add_argument('--img_size', nargs='+', type=int, default=[800, 1440], help='[train, test] image sizes')
    parser.add_argument('--high_thresh', type=float, default=0.5)

    parser.add_argument('--save_dir', type=str, default='det_results/{dataset_name}/{split}')

    parser.add_argument('--device', type=str, default='1')

    parser.add_argument('--fp16', type=bool, default=False)

    parser.add_argument('--vis', action='store_true')

    parser.add_argument('--generate_meta_data', action='store_true', help='generate meta data for ssm tracker, such as img h and w')

    return parser.parse_args()

def select_device(device):
    """ set device 
    Args:
        device: str, 'cpu' or '0' or '1,2,3'-like

    Return:
        torch.device
    
    """

    if device == 'cpu':
        logger.info('Use CPU for training')

    elif ',' in device:  # multi-gpu
        logger.error('Multi-GPU currently not supported')
    
    else:
        logger.info(f'set gpu {device}')
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available()

    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    return device

def postprocess_yolox(out, num_classes, conf_thresh, img, ori_img):
    """
    convert out to  -> (tlbr, conf, cls)
    """

    out = postprocess(out, num_classes, conf_thresh, )[0]  # (tlbr, obj_conf, cls_conf, cls)

    if out is None: return out

    # merge conf 
    out[:, 4] *= out[:, 5]
    out[:, 5] = out[:, -1]
    out = out[:, :-1]

    # scale to origin size 

    img_size = [img.shape[-2], img.shape[-1]]  # h, w
    ori_img_size = [ori_img.shape[0], ori_img.shape[1]]  # h0, w0
    img_h, img_w = img_size[0], img_size[1]

    scale = min(float(img_h) / ori_img_size[0], float(img_w) / ori_img_size[1])

    out[:, :4] /= scale 

    return out

def save_results(folder_name, seq_name, result_dict, data_type='default'):
    """
    write results to txt file

    """
    

    if not os.path.exists(folder_name):
        os.makedirs(folder_name) 

    with open(os.path.join(folder_name, seq_name + '.txt'), 'w') as f:
        
        for frame_id, output in result_dict.items():

            for det in output:
                f.write(f'{frame_id},-1,{det[0]:.2f},{det[1]:.2f},{det[2]:.2f},{det[3]:.2f},{det[4]:.2f},-1,-1,-1\n')

    f.close()

    return folder_name

def save_meta_data(folder_name, meta_data):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name) 

    with open(os.path.join(folder_name, 'meta_data.txt'), 'w') as f:
        for k, v in meta_data.items():
            line = k + ','
            for item in v: line += str(item) + ','
            line = line[:-1]  # drop the last comma

            f.write(line + '\n')
    f.close()

def plot_img(img, frame_id, results, save_dir):
    """
    visualization
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if len(img.shape) > 3:
        img = img.squeeze(0)

    img_ = np.ascontiguousarray(np.copy(img))

    for det in results:
        tlwh, s = det[: 4], det[4]
        tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
        text = '{:.2f}'.format(s)

        cv2.rectangle(img_, tlbr[:2], tlbr[2:], (0, 255, 0), thickness=3, )

        cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                        color=(255, 164, 0), thickness=2)
        
    cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=img_)

def main(args):
    
    exp = get_exp(args.exp_file, args.dataset_name)

    device = select_device(args.device)
    model = exp.get_model().to(device)

    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    # load ckpt
    ckpt = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    # default fuse the model 
    logger.info("\tFusing model...")
    model = fuse_model(model)

    if args.fp16:
        model = model.half()
    
    data_root = os.path.join(args.data_root, args.split)
    save_dir = args.save_dir.format(dataset_name=args.dataset_name, split=args.split)

    seqs = os.listdir(data_root)  # all seq names

    # NOTE for debug
    # seqs = ['uav0000013_00000_v', 'uav0000072_05448_v']

    meta_data = dict()  # meta data, including image h and w

    for seq in seqs:
        logger.info(f'detecting seq {seq}')

        imgs = os.listdir(os.path.join(data_root, seq))
        imgs = sorted(imgs)

        det_result_dict = dict()
        frame_id = 1

        img_ori = cv2.imread(os.path.join(data_root, seq, imgs[0]))
        meta_data[seq] = [img_ori.shape[0], img_ori.shape[1]]  # h, w, 

        for img_name in tqdm(imgs):
            img_ori = cv2.imread(os.path.join(data_root, seq, img_name))

            # save meta data
            if not seq in meta_data.keys():
                meta_data[seq] = [img_ori.shape[0], img_ori.shape[1]]  # h, w, 

            img, ratio = preproc(img_ori, args.img_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

            img = torch.from_numpy(img).unsqueeze(0).float().to(device)

            if args.fp16:
                img = img.half()

            with torch.no_grad():
                output = model(img)

                output = postprocess_yolox(output, exp.num_classes, 0.05, img, img_ori)  # [N, 5] 

                # tlbr to tlwh
                output[:, 2] -= output[:, 0]
                output[:, 3] -= output[:, 1]

            if args.vis:
                plot_img(img_ori, frame_id, output, save_dir=os.path.join(save_dir, 'vis_results'))

            det_result_dict[frame_id] = output.cpu().numpy()

            frame_id += 1


        # write result
        logger.info(f'write results of seq {seq}')

        save_results(save_dir, seq, det_result_dict)

    # save meta data
    if args.generate_meta_data:
        save_meta_data(save_dir, meta_data)
                


if __name__ == '__main__':
    args = get_args()
    main(args)


