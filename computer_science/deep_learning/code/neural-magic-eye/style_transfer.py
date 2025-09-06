import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import cv2
import glob
from networks import define_G
import utils
import random
from datetime import datetime
from tqdm import tqdm

def generate_random_texture(size=512, pattern_type='random'):
    random.seed(int(datetime.now().timestamp()) % 1000)
    
    if pattern_type == 'noise':
        texture = np.random.normal(0.5, 0.2, (size, size))
        texture = np.clip((texture - texture.min()) / (texture.max() - texture.min()), 0, 1)
        texture = np.stack([texture] * 3, axis=-1)
        
    elif pattern_type == 'gradient':
        x = np.linspace(0, 2*np.pi, size)
        y = np.linspace(0, 2*np.pi, size)
        xx, yy = np.meshgrid(x, y)
        
        texture = (np.sin(xx) + np.sin(yy)) / 2 + 0.5
        
        noise = np.random.normal(0, 0.03, texture.shape)
        fade_width = int(size * 0.05)
        if fade_width > 0:
            mask_x = np.ones(size)
            mask_y = np.ones(size)
            for i in range(fade_width):
                blend_factor = i / fade_width
                mask_x[i] = blend_factor
                mask_x[-i-1] = blend_factor
                mask_y[i] = blend_factor
                mask_y[-i-1] = blend_factor
            mask = np.outer(mask_y, mask_x)
            noise = noise * mask
            
        texture += noise
        texture = np.clip(texture, 0, 1)
        
        hsv = np.zeros((size, size, 3), dtype=np.float32)
        hsv[:, :, 0] = (texture * 0.2) + 0.7  # Hue varies with the gradient
        hsv[:, :, 1] = 0.6  # Saturation
        hsv[:, :, 2] = texture  # Value from the gradient
        
        # Convert to RGB
        texture_bgr = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        texture = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB) / 255.0
        
    elif pattern_type == 'stripes':
        freq = random.randint(5, 20)
        phase = random.random() * 2 * np.pi
        x = np.linspace(0, 2 * np.pi * freq, size)
        texture = 0.5 + 0.5 * np.sin(x + phase)
        texture = np.tile(texture[:, np.newaxis], (1, size))
        texture += np.random.normal(0, 0.05, texture.shape)
        texture = np.clip(texture, 0, 1)
        texture = np.stack([
            texture,
            np.roll(texture, shift=random.randint(5, 15), axis=0),
            np.roll(texture, shift=random.randint(5, 15), axis=1)
        ], axis=-1)
        
    elif pattern_type == 'dots':
        texture = np.zeros((size, size))
        dot_size = random.randint(2, 6)
        spacing = random.randint(10, 30)
        
        for i in range(0, size, spacing):
            for j in range(0, size, spacing):
                offset_i = random.randint(-spacing//4, spacing//4)
                offset_j = random.randint(-spacing//4, spacing//4)
                
                ii = (i + offset_i) % size
                jj = (j + offset_j) % size
                
                if ii+dot_size >= size or jj+dot_size >= size:
                    continue
                
                # Gaussian
                for di in range(-dot_size, dot_size+1):
                    for dj in range(-dot_size, dot_size+1):
                        if 0 <= ii+di < size and 0 <= jj+dj < size:
                            dist = np.sqrt(di**2 + dj**2)
                            if dist <= dot_size:
                                texture[ii+di, jj+dj] = np.exp(-0.5 * (dist/dot_size)**2)
        
        texture += np.random.normal(0, 0.05, texture.shape)
        texture = np.clip(texture, 0, 1)
        
        hue = random.random()
        hsv = np.zeros((size, size, 3), dtype=np.float32)
        hsv[:, :, 0] = hue
        hsv[:, :, 1] = 0.7 + 0.3 * texture
        hsv[:, :, 2] = texture
        
        texture_bgr = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        texture = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB) / 255.0
        
    else:  # random (default)
        texture = np.zeros((size, size, 3), dtype=np.float32)
        block_size = random.randint(4, 16)
        for i in range(0, size, block_size):
            for j in range(0, size, block_size):
                color = np.random.random(3)
                i_end = min(i + block_size, size)
                j_end = min(j + block_size, size)
                texture[i:i_end, j:j_end, :] = color
        
        texture = cv2.GaussianBlur(texture, (5, 5), 0)
    
    return texture

def safe_stereogram_synthesis(texture, depth_map, beta=0.05, tile_percentage=0.1):
    if len(depth_map.shape) == 3:
        depth_map = np.mean(depth_map, axis=-1)
    H, W = depth_map.shape
    
    tile_width = max(int(W * tile_percentage), 10)
    
    texture_h, texture_w = texture.shape[:2]
    texture_aspect = texture_w / texture_h
    tile_height = int(tile_width / texture_aspect)
    texture_resized = cv2.resize(texture, (tile_width, tile_height))
    
    repeats = W // tile_width + 2
    tiled_texture = np.tile(texture_resized, (1, repeats, 1))
    
    if tiled_texture.shape[0] != H:
        tiled_texture = cv2.resize(tiled_texture, (tiled_texture.shape[1], H))
    
    if tiled_texture.shape[1] > W:
        tiled_texture = tiled_texture[:, :W, :]
    
    stereogram = np.copy(tiled_texture)
    
    safe_margin = tile_width
    
    stereogram[:, :safe_margin, :] = tiled_texture[:, :safe_margin, :]
    
    for x in range(safe_margin, W):
        depth = depth_map[:, x]
        
        shift = np.clip(tile_width * (1 - depth * beta), 0, x - 1)
        
        for y in range(H):
            src_x = int(x - shift[y])
            if 0 <= src_x < W:
                stereogram[y, x, :] = stereogram[y, src_x, :]
    
    return stereogram

def get_model_parameters(model_preset):
    presets = {
        'fast': {
            'net_G': 'unet_64',
            'norm_type': 'batch',
            'with_disparity_conv': False
        },
        'balanced': {
            'net_G': 'unet_128',
            'norm_type': 'batch',
            'with_disparity_conv': True
        },
        'quality': {
            'net_G': 'unet_256',
            'norm_type': 'batch',
            'with_disparity_conv': True
        }
    }
    
    return presets.get(model_preset, presets['quality'])

def get_effect_parameters(effect_strength):
    presets = {
        'low': {
            'beta': 0.03,
            'tile_percentage': 0.08
        },
        'medium': {
            'beta': 0.05,
            'tile_percentage': 0.1
        },
        'high': {
            'beta': 0.08,
            'tile_percentage': 0.15
        }
    }
    
    return presets.get(effect_strength, presets['medium'])

def main():
    parser = argparse.ArgumentParser(description='Neural Magic Eye Style Transfer')
    parser.add_argument('--input_stereogram', type=str, required=True, 
                        help='path to input stereogram')
    parser.add_argument('--texture_dir', type=str, default='', 
                        help='optional directory with textures')
    parser.add_argument('--output_dir', type=str, default='./style_transfer_results', 
                        help='output directory')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints_decode_sp_u256_bn_df', 
                        help='checkpoint directory or file')
    parser.add_argument('--model', type=str, default='quality', 
                        choices=['fast', 'balanced', 'quality'],
                        help='model configuration preset')
    parser.add_argument('--effect', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='3D effect strength preset')
    parser.add_argument('--textures', type=str, default='random,noise,stripes,dots,gradient', 
                        help='comma-separated list of texture types to generate')
    parser.add_argument('--in_size', type=int, default=256, 
                        help='input size')
    args = parser.parse_args()
    
    model_params = get_model_parameters(args.model)
    effect_params = get_effect_parameters(args.effect)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.input_stereogram):
        print(f"Input file not found: {args.input_stereogram}")
        return
    
    model_args = type('', (), {})()
    model_args.net_G = model_params['net_G']
    model_args.norm_type = model_params['norm_type']
    model_args.with_disparity_conv = model_params['with_disparity_conv']
    model_args.in_size = args.in_size
    model_args.with_skip_connection = False
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = define_G(model_args)
        
        checkpoint_path = args.checkpoint
        if os.path.isdir(checkpoint_path):
            if os.path.exists(os.path.join(checkpoint_path, 'best_ckpt.pt')):
                checkpoint_path = os.path.join(checkpoint_path, 'best_ckpt.pt')
            elif os.path.exists(os.path.join(checkpoint_path, 'last_ckpt.pt')):
                checkpoint_path = os.path.join(checkpoint_path, 'last_ckpt.pt')
            else:
                print(f"No checkpoint found in {checkpoint_path}")
                return
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_G_state_dict'])
        model.to(device)
        model.eval()
        print(f"Model loaded from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    input_stereogram = cv2.imread(args.input_stereogram)
    if input_stereogram is None:
        print(f"Cannot read image")
        return
    
    input_stereogram = cv2.cvtColor(input_stereogram, cv2.COLOR_BGR2RGB) / 255.0
    org_h, org_w = input_stereogram.shape[:2]
    input_resized = cv2.resize(input_stereogram, (args.in_size, args.in_size), cv2.INTER_CUBIC)
    
    input_tensor = torch.FloatTensor(input_resized.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = model(input_tensor).squeeze().cpu().numpy()
    
    depth_map = np.clip(depth_map, a_min=0, a_max=1.0)
    depth_map = utils.normalize(depth_map, p_min=0.02, p_max=0.02)
    
    depth_out_path = os.path.join(args.output_dir, "extracted_depth.png")
    cv2.imwrite(depth_out_path, (depth_map * 255).astype(np.uint8))
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    depth_color_path = os.path.join(args.output_dir, "extracted_depth_color.png")
    plt.imsave(depth_color_path, depth_map, cmap='plasma')
    
    textures = []
    
    if args.texture_dir and os.path.exists(args.texture_dir):
        print(f"Using textures from: {args.texture_dir}")
        texture_files = glob.glob(os.path.join(args.texture_dir, '*.jpg')) + \
                       glob.glob(os.path.join(args.texture_dir, '*.png'))
        
        for texture_path in texture_files:
            if os.path.basename(args.input_stereogram) == os.path.basename(texture_path):
                continue
                
            texture = cv2.imread(texture_path)
            if texture is None:
                print(f"Failed to read texture: {texture_path}")
                continue
                
            texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB) / 255.0
            texture_name = os.path.splitext(os.path.basename(texture_path))[0]
            textures.append((texture_name, texture))
    else:
        texture_types = args.textures.split(',')
        print(f"Generating {len(texture_types)} textures...")
        
        for i, texture_type in enumerate(texture_types):
            print(f"Generating {texture_type} texture")
            texture = generate_random_texture(size=512, pattern_type=texture_type)
            textures.append((f"generated_{texture_type}", texture))
            
            texture_path = os.path.join(args.output_dir, f"texture_{texture_type}.png")
            cv2.imwrite(texture_path, cv2.cvtColor((texture * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    successful_count = 0
    for texture_name, texture in tqdm(textures, desc="Generating stereograms"):
        try:
            styled_stereogram = safe_stereogram_synthesis(
                texture, depth_map, 
                beta=effect_params['beta'], 
                tile_percentage=effect_params['tile_percentage']
            )
            
            styled_stereogram = cv2.resize(styled_stereogram, (org_w, org_h), cv2.INTER_CUBIC)
            
            output_path = os.path.join(args.output_dir, f"styled_{texture_name}.png")
            cv2.imwrite(output_path, cv2.cvtColor((styled_stereogram * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            successful_count += 1
        except Exception as e:
            tqdm.write(f"Error with texture {texture_name}: {e}")
if __name__ == "__main__":
    main()