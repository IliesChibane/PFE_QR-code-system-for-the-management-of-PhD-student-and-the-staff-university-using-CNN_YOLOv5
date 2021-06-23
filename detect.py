import argparse
import time
from pathlib import Path
import shutil
import os
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

@torch.no_grad()
def detect(weights='YOLOv5ExtraLargeBest.pt',  # model.pt path(s)
           source='0',  # file/dir/URL/glob, 0 for webcam
           imgsz=416,  # inference size (pixels)
           conf_thres=0.4,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=False,  # show results
           save_txt=False,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=True,  # hide labels
           hide_conf=True,  # hide confidences
           half=False,  # use FP16 half-precision inference
           ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric()

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        if conf >= 0.80:
                            roi = im0[int(xyxy[1].item()):int(xyxy[3].item()), int(xyxy[0].item()):int(xyxy[2].item())]
                            cv2.destroyAllWindows()
                            return roi
                    else:
                        break
            # Stream results
            if view_img:
                cv2.imshow('QR-CODE Scanner', im0)
                key = cv2.waitKey(1)  # 1 millisecond
                if key == 27:
                    cv2.destroyAllWindows()
                    return None



