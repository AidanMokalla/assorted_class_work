import os
import uuid
import numpy as np
import torch
import cv2
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from networks import define_G
import utils
from tqdm import tqdm
import base64
from io import BytesIO

from style_transfer import (
    generate_random_texture,
    safe_stereogram_synthesis,
    get_model_parameters,
    get_effect_parameters
)

app = Flask(__name__)
app.secret_key = 'neural_magic_eye_secret_key'


UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
DEFAULT_TEXTURES = ['random', 'noise', 'stripes', 'dots', 'gradient']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = None
device = None

def load_model():
    global model, device
    if model is None:
        model_args = type('', (), {})()
        model_params = get_model_parameters('quality')
        model_args.net_G = model_params['net_G']
        model_args.norm_type = model_params['norm_type']
        model_args.with_disparity_conv = model_params['with_disparity_conv']
        model_args.in_size = 256
        model_args.with_skip_connection = False
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = define_G(model_args)
        
        checkpoint_paths = [
            'neural-magic-eye/checkpoints_decode_sp_u256_bn_df/best_ckpt.pt',
            './checkpoints_decode_sp_u256_bn_df/best_ckpt.pt',
            'neural-magic-eye/checkpoints_decode_sp_u256_bn_df/last_ckpt.pt',
            './checkpoints_decode_sp_u256_bn_df/last_ckpt.pt'
        ]
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint_path = path
                print(f"Using checkpoint: {checkpoint_path}")
                break
                
        if checkpoint_path is None:
            raise FileNotFoundError("Couldn't find valid checkpoint")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_G_state_dict'])
        model.to(device)
        model.eval()
        print(f"Model loaded successfully with arch: {model_args.net_G}")
        
    return model, device

def extract_depth_map(image_path, p_min=0.02, p_max=0.02):
    try:
        model, device = load_model()
        
        input_stereogram = cv2.imread(image_path)
        if input_stereogram is None:
            print(f"Failed to load image from {image_path}")
            return None, None, None
                
        input_stereogram = cv2.cvtColor(input_stereogram, cv2.COLOR_BGR2RGB) / 255.0
        org_h, org_w = input_stereogram.shape[:2]
        
        # model's in_size parameter is consistent with style_transfer.py
        in_size = 256  # matches model_args.in_size in load_model
        input_resized = cv2.resize(input_stereogram, (in_size, in_size), cv2.INTER_CUBIC)
        
        # depth map
        input_tensor = torch.FloatTensor(input_resized.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            depth_map = model(input_tensor).squeeze().cpu().numpy()
                
        # Normalize
        depth_map = np.clip(depth_map, a_min=0, a_max=1.0)
        depth_map = utils.normalize(depth_map, p_min=p_min, p_max=p_max)
                
        # Resize
        depth_map_resized = cv2.resize(depth_map, (org_w, org_h), cv2.INTER_CUBIC)
        
        return depth_map, depth_map_resized, (org_h, org_w)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, None, None

def create_texture(texture_type, size=512):
    
    texture = generate_random_texture(size=size, pattern_type=texture_type)
    texture = texture.astype(np.float32)
    texture = np.clip(texture, 0, 1.0)
    
    return texture

def generate_stereogram(depth_map, texture, effect="medium", original_dims=None, custom_beta=None, custom_tile_percentage=None):
    effect_params = get_effect_parameters(effect)
    
    beta = custom_beta if custom_beta is not None else effect_params['beta']
    tile_percentage = custom_tile_percentage if custom_tile_percentage is not None else effect_params['tile_percentage']
    
    stereogram = safe_stereogram_synthesis(
        texture=texture,
        depth_map=depth_map,
        beta=beta,
        tile_percentage=tile_percentage
    )
    
    if original_dims:
        org_h, org_w = original_dims
        stereogram = cv2.resize(stereogram, (org_w, org_h), cv2.INTER_CUBIC)
    
    return stereogram

# base64 for embedding in HTML
def image_to_base64(image_array):
    if len(image_array.shape) == 2:  # Grayscale
        image_array = np.stack([image_array] * 3, axis=-1)
    
    # bytes
    success, buffer = cv2.imencode('.png', (image_array * 255).astype(np.uint8))
    if not success:
        return None
    
    # base64
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{encoded_image}"

@app.route('/') # flask decorators
def index():
    return render_template('index.html')

@app.route('/upload-stereogram', methods=['POST'])
def upload_stereogram():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # normalization parameters
        p_min = float(request.form.get('p_min', 0.02))
        p_max = float(request.form.get('p_max', 0.02))
        
        # depth map
        try:
            depth_map, depth_map_resized, original_dims = extract_depth_map(filepath, p_min=p_min, p_max=p_max)
            
            depth_filename = f"{unique_id}_depth.png"
            depth_filepath = os.path.join(app.config['RESULT_FOLDER'], depth_filename)
            cv2.imwrite(depth_filepath, (depth_map_resized * 255).astype(np.uint8))
            
            depth_base64 = image_to_base64(depth_map_resized)
            
            return jsonify({
                'success': True,
                'message': 'Depth map extracted successfully',
                'stereogram_id': unique_id,
                'stereogram_path': filepath,
                'depth_map_path': depth_filepath,
                'depth_map_preview': depth_base64,
                'p_min': p_min,
                'p_max': p_max
            })
        
        except Exception as e:
            import traceback
            error_msg = str(e)
            error_trace = traceback.format_exc()
            print(f"Error processing stereogram: {error_msg}")
            print(error_trace)
            return jsonify({
                'error': f'Error processing stereogram: {error_msg}',
                'details': error_trace
            }), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/upload-depth-map', methods=['POST'])
def upload_depth_map():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            depth_map = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if depth_map is None:
                return jsonify({'error': 'Failed to read depth map'}), 400
            
            depth_map = depth_map.astype(np.float32) / 255.0
            
            # base64 for preview
            depth_base64 = image_to_base64(depth_map)
            
            return jsonify({
                'success': True,
                'message': 'Depth map uploaded successfully',
                'depth_map_id': unique_id,
                'depth_map_path': filepath,
                'depth_map_preview': depth_base64
            })
        
        except Exception as e:
            return jsonify({'error': f'Error processing depth map: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/upload-texture', methods=['POST'])
def upload_texture():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            texture = cv2.imread(filepath)
            if texture is None:
                return jsonify({'error': 'Failed to read texture'}), 400
            
            texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            texture = np.clip(texture, 0, 1)
            
            cv2.imwrite(filepath, cv2.cvtColor((texture * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            
            # base64 for preview
            texture_base64 = image_to_base64(cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
            
            return jsonify({
                'success': True,
                'message': 'Texture uploaded successfully',
                'texture_id': unique_id,
                'texture_path': filepath,
                'texture_preview': texture_base64
            })
        
        except Exception as e:
            return jsonify({'error': f'Error processing texture: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/generate-stereogram', methods=['POST'])
def generate_stereogram_route():
    data = request.json
    
    depth_map_path = data.get('depth_map_path')
    if not depth_map_path or not os.path.exists(depth_map_path):
        return jsonify({'error': 'Invalid depth map path'}), 400
    
    texture_types = data.get('texture_types', DEFAULT_TEXTURES)
    
    custom_texture_paths = data.get('custom_texture_paths', [])
    
    effect = data.get('effect', 'medium')
    
    custom_beta = data.get('custom_beta')
    if custom_beta is not None:
        custom_beta = float(custom_beta)
    
    custom_tile_percentage = data.get('custom_tile_percentage')
    if custom_tile_percentage is not None:
        custom_tile_percentage = float(custom_tile_percentage)
    
    try:
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
        if depth_map is None:
            return jsonify({'error': 'Failed to read depth map'}), 400
        
        # Normalize
        depth_map = depth_map.astype(np.float32) / 255.0
        original_dims = depth_map.shape[:2]
        
        results = []
        
        # default textures
        for texture_type in texture_types:
            try:
                texture = create_texture(texture_type)
                
                stereogram = generate_stereogram(
                    depth_map, 
                    texture, 
                    effect, 
                    original_dims, 
                    custom_beta=custom_beta,
                    custom_tile_percentage=custom_tile_percentage
                )
                
                unique_id = str(uuid.uuid4())
                stereogram_filename = f"{unique_id}_{texture_type}.png"
                stereogram_filepath = os.path.join(app.config['RESULT_FOLDER'], stereogram_filename)
                cv2.imwrite(stereogram_filepath, cv2.cvtColor((stereogram * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                stereogram_base64 = image_to_base64(cv2.cvtColor(stereogram, cv2.COLOR_RGB2BGR))
                
                results.append({
                    'texture_type': texture_type,
                    'stereogram_path': stereogram_filepath,
                    'stereogram_preview': stereogram_base64
                })
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                continue
        
        # custom textures
        for texture_path in custom_texture_paths:
            if os.path.exists(texture_path):
                try:
                    texture = cv2.imread(texture_path)
                    if texture is None:
                        continue
                    
                    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    texture = np.clip(texture, 0, 1)
                    
                    stereogram = generate_stereogram(
                        depth_map, 
                        texture, 
                        effect, 
                        original_dims, 
                        custom_beta=custom_beta,
                        custom_tile_percentage=custom_tile_percentage
                    )
                    
                    unique_id = str(uuid.uuid4())
                    texture_name = os.path.basename(texture_path).split('_', 1)[1] if '_' in os.path.basename(texture_path) else os.path.basename(texture_path)
                    stereogram_filename = f"{unique_id}_{texture_name}"
                    stereogram_filepath = os.path.join(app.config['RESULT_FOLDER'], stereogram_filename)
                    cv2.imwrite(stereogram_filepath, cv2.cvtColor((stereogram * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    
                    stereogram_base64 = image_to_base64(cv2.cvtColor(stereogram, cv2.COLOR_RGB2BGR))
                    
                    results.append({
                        'texture_type': f"custom_{os.path.basename(texture_path)}",
                        'stereogram_path': stereogram_filepath,
                        'stereogram_preview': stereogram_base64
                    })
                except Exception as e:
                    print(f"Error generating custom texture stereogram: {str(e)}")
                    continue
        
        return jsonify({
            'success': True,
            'message': f'Generated {len(results)} stereograms',
            'results': results,
            'parameters': {
                'effect': effect,
                'custom_beta': custom_beta,
                'custom_tile_percentage': custom_tile_percentage
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Error generating stereograms: {str(e)}'}), 500

@app.route('/transfer-style', methods=['POST'])
def transfer_style_route():
    data = request.json
    
    stereogram_path = data.get('stereogram_path')
    if not stereogram_path or not os.path.exists(stereogram_path):
        return jsonify({'error': 'Invalid stereogram path'}), 400
    
    texture_types = data.get('texture_types', DEFAULT_TEXTURES)
    
    custom_texture_paths = data.get('custom_texture_paths', [])
    
    effect = data.get('effect', 'medium')
    
    p_min = float(data.get('p_min', 0.02))
    p_max = float(data.get('p_max', 0.02))
    
    custom_beta = data.get('custom_beta')
    if custom_beta is not None:
        custom_beta = float(custom_beta)
    
    custom_tile_percentage = data.get('custom_tile_percentage')
    if custom_tile_percentage is not None:
        custom_tile_percentage = float(custom_tile_percentage)
    
    try:
        depth_map, depth_map_resized, original_dims = extract_depth_map(
            stereogram_path, 
            p_min=p_min, 
            p_max=p_max
        )
        
        if depth_map is None:
            return jsonify({'error': 'Failed to extract depth map from stereogram. See server logs for details.'}), 500
        
        unique_id = str(uuid.uuid4())
        depth_filename = f"{unique_id}_depth.png"
        depth_filepath = os.path.join(app.config['RESULT_FOLDER'], depth_filename)
        cv2.imwrite(depth_filepath, (depth_map_resized * 255).astype(np.uint8))
        
        depth_base64 = image_to_base64(depth_map_resized)
        
        results = []
        
        for texture_type in texture_types:
            try:
                texture = create_texture(texture_type)                
                stereogram = generate_stereogram(
                    depth_map, 
                    texture, 
                    effect, 
                    original_dims,
                    custom_beta=custom_beta,
                    custom_tile_percentage=custom_tile_percentage
                )
                
                unique_id = str(uuid.uuid4())
                stereogram_filename = f"{unique_id}_{texture_type}.png"
                stereogram_filepath = os.path.join(app.config['RESULT_FOLDER'], stereogram_filename)
                cv2.imwrite(stereogram_filepath, cv2.cvtColor((stereogram * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                stereogram_base64 = image_to_base64(cv2.cvtColor(stereogram, cv2.COLOR_RGB2BGR))
                
                results.append({
                    'texture_type': texture_type,
                    'stereogram_path': stereogram_filepath,
                    'stereogram_preview': stereogram_base64
                })
            except Exception as e:
                import traceback
                error_msg = str(e)
                error_trace = traceback.format_exc()
                print(error_trace)
                continue
        
        for texture_path in custom_texture_paths:
            if os.path.exists(texture_path):
                try:
                    texture = cv2.imread(texture_path)
                    if texture is None:
                        continue
                    
                    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    texture = np.clip(texture, 0, 1) 
                    
                    stereogram = generate_stereogram(
                        depth_map, 
                        texture, 
                        effect, 
                        original_dims,
                        custom_beta=custom_beta,
                        custom_tile_percentage=custom_tile_percentage
                    )
                    
                    unique_id = str(uuid.uuid4())
                    texture_name = os.path.basename(texture_path).split('_', 1)[1] if '_' in os.path.basename(texture_path) else os.path.basename(texture_path)
                    stereogram_filename = f"{unique_id}_{texture_name}"
                    stereogram_filepath = os.path.join(app.config['RESULT_FOLDER'], stereogram_filename)
                    cv2.imwrite(stereogram_filepath, cv2.cvtColor((stereogram * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    
                    stereogram_base64 = image_to_base64(cv2.cvtColor(stereogram, cv2.COLOR_RGB2BGR))
                    
                    results.append({
                        'texture_type': f"custom_{os.path.basename(texture_path)}",
                        'stereogram_path': stereogram_filepath,
                        'stereogram_preview': stereogram_base64
                    })
                except Exception as e:
                    print(f"Error generating custom texture stereogram: {str(e)}")
                    continue
        
        return jsonify({
            'success': True,
            'message': f'Generated {len(results)} stereograms',
            'depth_map_path': depth_filepath,
            'depth_map_preview': depth_base64,
            'results': results,
            'parameters': {
                'p_min': p_min,
                'p_max': p_max,
                'effect': effect,
                'custom_beta': custom_beta,
                'custom_tile_percentage': custom_tile_percentage
            }
        })
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"Error in transfer_style: {error_msg}")
        print(error_trace)
        return jsonify({
            'error': f'Error transferring style: {error_msg}',
            'details': error_trace
        }), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9010) 