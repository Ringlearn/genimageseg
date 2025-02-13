from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
# from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
import sys
import os
from collections import OrderedDict
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from src.utils import ptp_utils
from src.utils import seq_aligner
import cv2
import json
import torchvision
import argparse
import multiprocessing as mp
import torch.nn as nn
import threading
from random import choice
import os
import yaml
import argparse
from IPython.display import Image, display
from pytorch_lightning import seed_everything
from tqdm import tqdm
from src.data.datadm import *
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from src.model.datadm.unet import UNet2D,get_feature_dic,clear_feature_dic
from src.model.datadm.segment.transformer_decoder_semantic import seg_decorder_open_word
import torch.optim as optim
from train.datadm.train_instance_coco import dict2obj,instance_inference
from train.datadm.train import AttentionStore
import torch.nn.functional as F
from scipy.special import softmax
from detectron2.utils.memory import retry_if_cuda_oom
from random import choice
from src.utils.vis.vis_VOC import visualise_segmentation

classes = {
        0: 'background',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        12: 'stop sign',
        13: 'parking meter',
        14: 'bench',
        15: 'bird',
        16: 'cat',
        17: 'dog',
        18: 'horse',
        19: 'sheep',
        20: 'cow',
        21: 'elephant',
        22: 'bear',
        23: 'zebra',
        24: 'giraffe',
        25: 'backpack',
        26: 'umbrella',
        27: 'handbag',
        28: 'tie',
        29: 'suitcase',
        30: 'frisbee',
        31: 'skis',
        32: 'snowboard',
        33: 'sports ball',
        34: 'kite',
        35: 'baseball bat',
        36: 'baseball glove',
        37: 'skateboard',
        38: 'surfboard',
        39: 'tennis racket',
        40: 'bottle',
        41: 'wine glass',
        42: 'cup',
        43: 'fork',
        44: 'knife',
        45: 'spoon',
        46: 'bowl',
        47: 'banana',
        48: 'apple',
        49: 'sandwich',
        50: 'orange',
        51: 'broccoli',
        52: 'carrot',
        53: 'hot dog',
        54: 'pizza',
        55: 'donut',
        56: 'cake',
        57: 'chair',
        58: 'couch',
        59: 'potted plant',
        60: 'bed',
        61: 'dining table',
        62: 'toilet',
        63: 'tv',
        64: 'laptop',
        65: 'computer mouse',
        66: 'remote',
        67: 'keyboard',
        68: 'cell phone',
        69: 'microwave',
        70: 'oven',
        71: 'toaster',
        72: 'sink',
        73: 'refrigerator',
        74: 'book',
        75: 'clock',
        76: 'vase',
        77: 'scissors',
        78: 'teddy bear',
        79: 'hair drier',
        80: 'toothbrush'
    }

classes_check = {
    0: [],
    1: ['aeroplane'],
    2: ['bicycle'],
    3: ['bird'],
    4: ['boat'],
    5: ['bottle'],
    6: ['bus'],
    7: ['car'],
    8: ['cat'],
    9: ['chair'],
    10: ['cow'],
    11: ['diningtable'],
    12: ['dog'],
    13: ['horse'],
    14: ['motorbike'],
    15: ['person'],
    16: ['pottedplant'],
    17: ['sheep'],
    18: ['sofa'],
    19: ['train'],
    20: ['tvmonitor']
}

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def voc_inference(prompts_list, outpath):    
    LOW_RESOURCE = False 
    NUM_DIFFUSION_STEPS = 50
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    freeze_params(vae.parameters())
    vae=vae.to(device)
    vae.eval()
    unet = UNet2D.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    freeze_params(unet.parameters())
    unet=unet.to(device)
    unet.eval()
    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
    freeze_params(text_encoder.parameters())
    text_encoder=text_encoder.to(device)
    text_encoder.eval()
    scheduler = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device).scheduler
    num_classes = 80
    num_queries = 100
    seg_model=seg_decorder_open_word(num_classes=num_classes, 
                           num_queries=num_queries).to(device)
    grounding_ckpt = "weights/datadm/voc/COCO_400RealImage.pth"
    base_weights = torch.load(grounding_ckpt, map_location=device)
    try:
        seg_model.load_state_dict(base_weights, strict=True)
    except:
        new_state_dict = OrderedDict()
        for k, v in base_weights.items():
            name = k[7:]   # remove `vgg.`
            new_state_dict[name] = v 
        seg_model.load_state_dict(new_state_dict, strict=True)
    # outpath = "Results/datadm/voc"
    os.makedirs(outpath, exist_ok=True)
    Image_path = os.path.join(outpath, "Image")
    os.makedirs(Image_path, exist_ok=True)
    # prompts_list = ["a car in city"]       
    Mask_path = os.path.join(outpath, "Mask")
    os.makedirs(Mask_path, exist_ok=True)
    controller = AttentionStore()
    ptp_utils.register_attention_control(unet, controller)
    seed = 0
    n_samples = 1
    W = H = 512
    C = 4
    f = 8
    with torch.no_grad():
        for eachprompt in prompts_list:
            clear_feature_dic()
            controller.reset()
            g_cpu = torch.Generator().manual_seed(seed)
            prompts = [eachprompt]
            start_code = torch.randn([n_samples, C, H // f, W // f], device=device)
            img_name = "_".join(word for word in eachprompt.split(' '))
            images_here, x_t = ptp_utils.text2image(unet,vae,tokenizer,text_encoder,scheduler, prompts, controller,
                                num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=7, generator=g_cpu,
                                low_resource=LOW_RESOURCE, Train=False)
            ptp_utils.save_images(images_here,out_put = "{}/{}_{}.jpg".format(Image_path,img_name,seed))
            full_arr = np.zeros((81, 512,512), np.float32)
            full_arr[0]=0.5
            for idxx in classes:
                if idxx==0:
                    continue

                class_name = classes[idxx]
                print(class_name)
                if class_name not in classes.values():
                    continue   
                    
                # train segmentation
                query_text = class_name            
                text_input = tokenizer(
                        query_text,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                text_embeddings = text_encoder(text_input.input_ids.to(unet.device))[0]
                class_embedding=text_embeddings
                
                if class_embedding.size()[1] > 1:
                    class_embedding = torch.unsqueeze(class_embedding.mean(1),1)

                diffusion_features=get_feature_dic()
#                     class_target  class_name
                outputs=seg_model(diffusion_features,controller,prompts,tokenizer,class_embedding)
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                mask_pred_results = F.interpolate(
                                    mask_pred_results,
                                    size=(512, 512),
                                    mode="bilinear",
                                    align_corners=False,
                                    )
                for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
                    
                    instance_r = retry_if_cuda_oom(instance_inference)(mask_cls_result, mask_pred_result,class_n = num_classes,test_topk_per_image=3,query_n = num_queries)
                        
                    pred_masks = instance_r.pred_masks.cpu().numpy().astype(np.uint8)
                    pred_boxes = instance_r.pred_boxes
                    scores = instance_r.scores 
                    pred_classes = instance_r.pred_classes 


                    import heapq
                    topk_idx = heapq.nlargest(1, range(len(scores)), scores.__getitem__)
                    mask_instance = (pred_masks[topk_idx[0]]>0.5 * 1).astype(np.uint8) 
                    full_arr[idxx] = np.array(mask_instance)
            full_arr = softmax(full_arr, axis=0)
            mask = np.argmax(full_arr, axis=0)
            cv2.imwrite("{}/{}_{}.png".format(Mask_path,img_name,seed), mask)
            seed+=1