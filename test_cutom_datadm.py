import os
from src.features.datadm.cityscape_inference import datadm_inference, visualize_segments
input_prompts = "a city with car and bus"
out_dir = "Results"
os.makedirs(out_dir, exist_ok=True)
datadm_inference(input_prompts, out_dir)
visualize_segments(out_dir)