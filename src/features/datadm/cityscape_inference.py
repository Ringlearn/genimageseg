import torch
import os
import cv2
import torch.nn.functional as F
from random import choice
from detectron2.utils.memory import retry_if_cuda_oom
from src.utils import ptp_utils
from collections import OrderedDict
from train.datadm.train import AttentionStore
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from src.model.datadm.unet import UNet2D,get_feature_dic,clear_feature_dic
from src.model.datadm.segment.transformer_decoder import seg_decorder
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

classes = {
                0: 'road',
                1: 'sidewalk',
                2: 'building',
                3: 'wall',
                4: 'fence',
                5: 'pole',
                6: 'traffic light',
                7: 'traffic sign',
                8: 'vegetation',
                9: 'terrain',
                10: 'sky',
                11: 'person',
                12: 'rider',
                13: 'car',
                14: 'truck',
                15: 'bus',
                16: 'train',
                17: 'motorcycle',
                18: 'bicycle'
            }
ADE20K_150_CATEGORIES = [
    {"color": [128, 64, 128], "id": 0, "isthing": 0, "name": "road"},
    {"color": [244, 35, 232], "id": 1, "isthing": 0, "name": "sidewalk"},
    {"color": [70, 70, 70], "id": 2, "isthing": 0, "name": "building"},
    {"color": [102, 102, 156], "id": 3, "isthing": 0, "name": "wall"},
    {"color": [190, 153, 153], "id": 4, "isthing": 0, "name": "fence"},
    {"color": [153, 153, 153], "id": 5, "isthing": 0, "name": "pole"},
    {"color": [250, 170, 30], "id": 6, "isthing": 0, "name": "traffic light"},
    {"color": [220, 220, 0], "id": 7, "isthing": 1, "name": "traffic sign"},
    {"color": [107, 142, 35], "id": 8, "isthing": 1, "name": "vegetation "},
    {"color": [152, 251, 152], "id": 9, "isthing": 0, "name": "terrain"},
    {"color": [70, 130, 180], "id": 10, "isthing": 1, "name": "sky"},
    {"color": [220, 20, 60], "id": 11, "isthing": 0, "name": "person"},
    {"color": [255, 0, 0], "id": 12, "isthing": 1, "name": "rider"},
    {"color": [0, 0, 142], "id": 13, "isthing": 0, "name": "car"},
    {"color": [0, 0, 70], "id": 14, "isthing": 1, "name": "truck"},
    {"color": [0, 60, 100], "id": 15, "isthing": 1, "name": "bus"},
    {"color": [0, 80, 100], "id": 16, "isthing": 0, "name": "train"},
    {"color": [0, 0, 230], "id": 17, "isthing": 0, "name": "motorcycle"},
    {"color": [119, 11, 32], "id": 18, "isthing": 1, "name": "bicycle"}
  
]

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def semantic_inference(mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
    for i in range(1,semseg.shape[0]):
        if (semseg[i]*(semseg[i]>0.5)).sum()<5000:
            semseg[i] = 0
    return semseg

def datadm_inference(prompts_list, out_dir):
    image_path = os.path.join(out_dir, "Image")
    mask_path = os.path.join(out_dir, "Mask")
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
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
    scheduler = DDPMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    # seg_model=seg_decorder().to(device)
    # base_weights = torch.load("/content/drive/MyDrive/GenDM/weights/datadm/Checkpoints/checkpoint_40.pth", map_location="cpu")
    # try:
    #     seg_model.load_state_dict(base_weights, strict=True)
    # except:
    #     new_state_dict = OrderedDict()
    #     for k, v in base_weights.items():
    #         name = k[7:]   # remove `vgg.`
    #         new_state_dict[name] = v 
    #     seg_model.load_state_dict(new_state_dict, strict=True)
    controller = AttentionStore()
    ptp_utils.register_attention_control(unet, controller)
    seed = 40
    n_samples = 1
    C = 3
    H = 512
    W = 512
    f = 8
    NUM_DIFFUSION_STEPS = 50
    LOW_RESOURCE = False
    # prompts = "a front view of car"
    with torch.no_grad():
        for prompts in prompts_list:
            clear_feature_dic()
            controller.reset()
            g_cpu = torch.Generator().manual_seed(seed)
            start_code = torch.randn([n_samples, C, H // f, W // f], device=device)
            images_here, x_t = ptp_utils.text2image(unet,vae,tokenizer,text_encoder,scheduler, prompts, controller,  
                                    num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=5, generator=g_cpu, 
                                    low_resource=LOW_RESOURCE, Train=False)
            ptp_utils.save_images(images_here,out_put = "{}/{}.jpg".format(out_dir,prompts))
            # diffusion_features=get_feature_dic()
            # outputs=seg_model(diffusion_features,controller,prompts,tokenizer)
            # mask_cls_results = outputs["pred_logits"]
            # mask_pred_results = outputs["pred_masks"]
            # mask_pred_results = F.interpolate(
            #                             mask_pred_results,
            #                             size=(512, 512),
            #                             mode="bilinear",
            #                             align_corners=False,
            #                             )
            # for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
            #     label_pred_prob = retry_if_cuda_oom(semantic_inference)(mask_cls_result, mask_pred_result)
            #     label_pred_prob = torch.argmax(label_pred_prob, axis=0)
            #     label_pred_prob = label_pred_prob.cpu().numpy()
            # cv2.imwrite("{}/{}.png".format(out_dir, prompts), label_pred_prob)
            # seed+=1

def _get_ade20k_full_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in ADE20K_150_CATEGORIES]
#     assert len(stuff_ids) == 847, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"].split(",")[0] for k in ADE20K_150_CATEGORIES]
    
    color = [k["color"] for k in ADE20K_150_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "color":color
    }
    return ret

def register_all_ade20k_full(root):
    root = os.path.join(root, "ADE20K_2021_17_01")
    meta = _get_ade20k_full_meta()
    for name, dirname in [("val", "validation")]:
        image_dir = os.path.join(root, "images_detectron2", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"sem_seg"
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            stuff_colors = meta["color"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=65535,  # NOTE: gt is saved in 16-bit TIFF images
        )

def visualize_segments(out_dir):
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ade20k_full(_root)
    root_image = os.path.join(out_dir, "Image")
    root_mask = os.path.join(out_dir, "Mask")
    vis_dir = os.path.join(out_dir, "Vis")
    os.makedirs(vis_dir, exist_ok=True)
    vis_list = os.listdir(root_image)
    mask_lsit = os.listdir(root_mask) 
    for image_name, mask_name in zip(vis_list, mask_lsit):
        mask_path = os.path.join(root_mask,mask_name)
        image_path = os.path.join(root_image,image_name)
        if not os.path.isfile(mask_path):
            continue  
        if not os.path.isfile(image_path):
            continue    
        mask = cv2.imread(mask_path)[:,:,0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w = image.shape[:2]
        metadata = MetadataCatalog.get("sem_seg")
        visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_sem_seg(
                            mask, alpha=0.5
                        )
        vis_output.save("{}/{}".format(vis_dir, image_name))