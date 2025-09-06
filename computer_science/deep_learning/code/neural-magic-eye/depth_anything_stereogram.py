#!/usr/bin/env python3
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
class WarningFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        
    def write(self, message):
        if ("Failed to load image Python extension" not in message and 
            "TensorFlow binary is optimized" not in message and 
            "torchvision.datapoints" not in message and 
            "transforms.v2 namespaces are still Beta" not in message and
            "oneDNN custom operations are on" not in message and
            "Using a slow image processor" not in message):
            self.original_stderr.write(message)
            
    def flush(self):
        self.original_stderr.flush()
sys.stderr = WarningFilter(sys.stderr)
import argparse
import numpy as np
import torch
import cv2
import glob
from tqdm import tqdm
import warnings
from networks import define_G
import torchvision
from PIL import Image
from transformers import pipeline
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
if hasattr(torchvision, 'disable_beta_transforms_warning'):
    torchvision.disable_beta_transforms_warning()
from style_transfer import (
    generate_random_texture,
    safe_stereogram_synthesis,
    get_effect_parameters
)

def extract_depth_from_image(image, model_size='Large'):
    h, w = image.shape[:2]
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    
    model_name = f"depth-anything/Depth-Anything-V2-{model_size}-hf"
    
    depth_estimator = pipeline(task="depth-estimation", model=model_name, use_fast=True)
    
    depth_result = depth_estimator(image_pil)
    
    depth_map = depth_result["depth"]
    depth_map = np.array(depth_map)
    
    if np.max(depth_map) > 0:
        depth_map = depth_map / np.max(depth_map)
    
    if depth_map.shape[:2] != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_CUBIC)
    
    return depth_map

def extract_depth_from_stereogram(stereogram, checkpoint_dir, in_size=256, device='cuda'):
    
    model_args = type('', (), {})()
    model_args.net_G = 'unet_256'
    model_args.norm_type = 'batch'
    model_args.with_disparity_conv = True
    model_args.in_size = in_size
    model_args.with_skip_connection = False
    
    model = define_G(model_args)
    
    checkpoint_path = checkpoint_dir
    if os.path.isdir(checkpoint_path):
        if os.path.exists(os.path.join(checkpoint_path, 'best_ckpt.pt')):
            checkpoint_path = os.path.join(checkpoint_path, 'best_ckpt.pt')
        elif os.path.exists(os.path.join(checkpoint_path, 'last_ckpt.pt')):
            checkpoint_path = os.path.join(checkpoint_path, 'last_ckpt.pt')
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_G_state_dict'])
    model.to(device)
    model.eval()
    
    org_h, org_w = stereogram.shape[:2]
    input_resized = cv2.resize(stereogram, (in_size, in_size), cv2.INTER_CUBIC)
    
    input_tensor = torch.FloatTensor(input_resized.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = model(input_tensor).squeeze().cpu().numpy()
    
    depth_map = np.clip(depth_map, a_min=0, a_max=1.0)
    
    depth_map = cv2.resize(depth_map, (org_w, org_h), cv2.INTER_CUBIC)
    
    return depth_map

def normalize_depth(depth_map, min_percentile=1, max_percentile=99, invert=True):
    min_val = np.percentile(depth_map, min_percentile)
    max_val = np.percentile(depth_map, max_percentile)
    
    if max_val - min_val > 0:
        depth_map = np.clip((depth_map - min_val) / (max_val - min_val), 0, 1)
    else:
        depth_map = np.zeros_like(depth_map)
    
    if invert:
        depth_map = 1.0 - depth_map
    
    return depth_map

def main():
    parser = argparse.ArgumentParser(description='Depth Anything Stereogram Generator')
    parser.add_argument('--input', type=str, required=True, help='path to input image, depth map, or stereogram')
    parser.add_argument('--input_type', type=str, required=True, choices=['image', 'depth_map', 'stereogram'],
                        help='specify the type of input (image, depth_map, or stereogram)')
    parser.add_argument('--output_dir', type=str, default='./results', help='output directory')
    parser.add_argument('--texture_dir', type=str, default='', help='optional textures directory')
    parser.add_argument('--textures', type=str, default='random,noise,stripes,dots,gradient', 
                        help='comma-separated list of texture types to generate')
    parser.add_argument('--output_depth', action='store_true', help='output depth map')
    parser.add_argument('--output_stereogram', action='store_true', help='output stereogram')
    parser.add_argument('--skip_normalization', action='store_true', help='skip depth map normalization')
    parser.add_argument('--invert_depth', action='store_true', help='invert the depth map (default for stereograms)')
    parser.add_argument('--input_size', type=int, default=0, help='input size for model inference (0 = full resolution)')
    args = parser.parse_args()
    checkpoint_decode = './checkpoints_decode_sp_u256_bn_df'
    model_size = 'Large'
    effect = 'high'
    
    if not args.output_depth and not args.output_stereogram:
        args.output_depth = True
        args.output_stereogram = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    img = cv2.imread(args.input)
    if img is None:
        raise ValueError(f"Cannot read image: {args.input}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    if args.input_type == "image":
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
        if args.input_size > 0:
            h, w = img_rgb.shape[:2]
            aspect_ratio = w / h
            if h > w:
                new_h = args.input_size
                new_w = int(new_h * aspect_ratio)
            else:
                new_w = args.input_size
                new_h = int(new_w / aspect_ratio)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        depth_map = extract_depth_from_image(
            img_rgb, model_size=model_size
        )
        
    elif args.input_type == "stereogram":
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
        depth_map = extract_depth_from_stereogram(
            img_rgb, checkpoint_decode, in_size=256, device=device
        )
        
    else:  # depth_map
        depth_map = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    
    if not args.skip_normalization:
        depth_map = normalize_depth(depth_map, invert=args.invert_depth)
    
    if args.output_depth:
        depth_out_path = os.path.join(args.output_dir, "depth_map.png")
        cv2.imwrite(depth_out_path, (depth_map * 255).astype(np.uint8))
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        depth_color_path = os.path.join(args.output_dir, "depth_map_color.png")
        plt.imsave(depth_color_path, depth_map, cmap='plasma')
            
    if args.output_stereogram:
        effect_params = get_effect_parameters(effect)
        textures = []
        
        if args.texture_dir and os.path.exists(args.texture_dir):
            print(f"Using textures from: {args.texture_dir}")
            texture_files = glob.glob(os.path.join(args.texture_dir, '*.jpg')) + \
                           glob.glob(os.path.join(args.texture_dir, '*.png'))
            
            for texture_path in texture_files:
                if os.path.basename(args.input) == os.path.basename(texture_path):
                    continue
                    
                texture = cv2.imread(texture_path)
                if texture is None:
                    print(f"Failed to read texture: {texture_path}")
                    continue
                    
                texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB) / 255.0
                texture_name = os.path.splitext(os.path.basename(texture_path))[0]
                textures.append((texture_name, texture))
        
        texture_types = args.textures.split(',')
        for texture_type in texture_types:
            texture = generate_random_texture(size=512, pattern_type=texture_type)
            textures.append((f"generated_{texture_type}", texture))
            
            texture_path = os.path.join(args.output_dir, f"texture_{texture_type}.png")
            cv2.imwrite(texture_path, cv2.cvtColor((texture * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        h, w = depth_map.shape[:2] if len(depth_map.shape) == 3 else depth_map.shape
        successful_count = 0
        
        for texture_name, texture in tqdm(textures, desc="Generating stereograms"):
            try:
                stereogram = safe_stereogram_synthesis(
                    texture, depth_map,
                    beta=effect_params['beta'],
                    tile_percentage=effect_params['tile_percentage']
                )
                
                output_path = os.path.join(args.output_dir, f"stereogram_{texture_name}.png")
                cv2.imwrite(output_path, cv2.cvtColor((stereogram * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                successful_count += 1
            except Exception as e:
                tqdm.write(f"Error with texture {texture_name}: {e}")
        
        print(f"Created {successful_count} stereograms in {args.output_dir}")

if __name__ == "__main__":
    main() 