from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json
from typing import Dict, Optional
import sys, os

WILOR_ROOT = os.path.expanduser(os.environ.get("WILOR_ROOT", "~/src/WiLoR"))
sys.path.insert(0, WILOR_ROOT)

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import cam_crop_to_full
from ultralytics import YOLO 



def main():
    parser = argparse.ArgumentParser(description='WiLoR demo for robot arm - no rendering, video input')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--out', type=str, required=True, help='Output .npz path')
    parser.add_argument('--focal_px', type=float, default=None, help='Phone focal length in pixels (from calibration)')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--fast',   dest='fast', action='store_true', default=False, help='Use FP16 and layer dropping to accelerate inference')
    args = parser.parse_args()

    # model load - start of wilor copy
    print("Loading WiLoR model...")
    model, model_cfg = load_wilor(checkpoint_path='./pretrained_models/wilor_final.ckpt', cfg_path='./pretrained_models/model_config.yaml')
    if args.fast:     
        torch.set_float32_matmul_precision('high')
        model = model.half()
        model.backbone = torch.compile(model.backbone)
        model.backbone.skip_blocks = True 
        
    print("Loading hand detector...")
    detector = YOLO(os.path.join(WILOR_ROOT, 'pretrained_models/detector.pt'))
    
    device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model    = model.to(device)
    detector = detector.to(device)
    model.eval()
    # end of copy

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Found {total} frames")
    # start of inference
    T = total
    keypoints   = np.full((T, 21, 3), np.nan, dtype=np.float32)   # camera frame
    global_rot  = np.full((T, 3, 3),  np.nan, dtype=np.float32)
    confidence  = np.zeros(T, dtype=np.float32)
    right_flags = np.zeros(T, dtype=bool)

    for frame_idx in range(total):
        ret, img_cv2 = cap.read()
        if not ret:
            continue

        detections = detector(img_cv2, conf=0.3, verbose=False)[0]
        bboxes, is_right, conf = [], [], []
        for det in detections:
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())
            conf.append(float(Bbox[4]))
        if not bboxes:
            continue                                   # row stays NaN

        # keep one hand: prefer right, then highest confidence
        order = sorted(range(len(bboxes)), key=lambda i: (is_right[i], conf[i]), reverse=True)
        k = order[0]
        boxes = np.array([bboxes[k]])
        right = np.array([is_right[k]])

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right,
                                rescale_factor=args.rescale_factor, fp16=args.fast)
        batch = recursive_to(next(iter(torch.utils.data.DataLoader(dataset, batch_size=1))), device)

        with torch.no_grad():
            out = model(batch)

        multiplier    = (2*batch['right']-1)
        pred_cam      = out['pred_cam']
        pred_cam[:,1] = multiplier*pred_cam[:,1]
        img_size      = batch["img_size"].float()
        focal = args.focal_px if args.focal_px else \
                model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        cam_t = cam_crop_to_full(pred_cam, batch["box_center"].float(), batch["box_size"].float(),
                                 img_size, focal).detach().cpu().numpy()[0]

        joints = out['pred_keypoints_3d'][0].detach().cpu().numpy()
        is_right_hand = bool(batch['right'][0].item())
        joints[:,0] = (1 if is_right_hand else -1) * joints[:,0]

        rot = out['pred_mano_params']['global_orient'][0].detach().cpu().numpy().reshape(3, 3)

        keypoints[frame_idx]   = joints + cam_t
        global_rot[frame_idx]  = rot
        confidence[frame_idx]  = conf[k]
        right_flags[frame_idx] = is_right_hand

    cap.release()
    np.savez_compressed(
        args.out,
        t=np.arange(T) / fps, fps=fps, width=width, height=height,
        focal_px=float(focal) if args.focal_px else np.nan,
        keypoints=keypoints, global_rot=global_rot,
        conf=confidence, is_right=right_flags,
    )

if __name__ == '__main__':
    main()
