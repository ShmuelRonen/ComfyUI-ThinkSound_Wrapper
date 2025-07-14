import os
import torch
import json
import numpy as np
import tempfile
import shutil
from typing import Optional, Tuple
from torchvision.transforms import v2
import torch.nn.functional as F
from transformers import AutoProcessor

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file

script_directory = os.path.dirname(os.path.abspath(__file__))

# Add model folder for ThinkSound
if not "thinksound" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("thinksound", os.path.join(folder_paths.models_dir, "thinksound"))

# Import ThinkSound modules from the downloaded repository
print("🔍 DEBUG: Starting ThinkSound import process...")
print(f"🔍 DEBUG: Script directory = {script_directory}")

try:
    import sys
    
    # Add ThinkSound directory to Python path
    thinksound_path = script_directory
    if thinksound_path not in sys.path:
        sys.path.append(thinksound_path)
    print(f"🔍 DEBUG: Added to sys.path: {thinksound_path}")
    
    # Check what folders exist
    print("🔍 DEBUG: Contents of script_directory:")
    try:
        contents = os.listdir(script_directory)
        for item in contents:
            item_path = os.path.join(script_directory, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"  ❌ Error listing directory: {e}")
    
    # Try to import from ThinkSound module structure
    print("🔍 DEBUG: Trying ThinkSound module import...")
    import_success = False
    
    # Make sure script_directory is in sys.path so Python can find "thinksound" as a module
    if script_directory not in sys.path:
        sys.path.append(script_directory)
    
    try:
        thinksound_subfolder = os.path.join(script_directory, "thinksound")
        print(f"🔍 DEBUG: thinksound subfolder = {thinksound_subfolder}")
        print(f"🔍 DEBUG: thinksound subfolder exists = {os.path.exists(thinksound_subfolder)}")
        
        if os.path.exists(thinksound_subfolder):
            print("🔍 DEBUG: Contents of thinksound subfolder:")
            for item in os.listdir(thinksound_subfolder):
                print(f"  📁 {item}")
        
        # Create an alias so feature_utils_224.py can find "ThinkSound" imports
        import thinksound
        sys.modules['ThinkSound'] = thinksound
        print("🔍 DEBUG: Created ThinkSound alias for thinksound module")
        
        # Import from thinksound.data.v2a_utils.feature_utils_224 
        from thinksound.data.v2a_utils.feature_utils_224 import FeaturesUtils
        print("✅ SUCCESS: Found FeaturesUtils in thinksound.data.v2a_utils.feature_utils_224")
        import_success = True
        
    except ImportError as e:
        print(f"❌ FAILED: ThinkSound module import - {e}")
        
        # Fallback: try old data_utils approach
        print("🔍 DEBUG: Trying fallback data_utils import...")
        try:
            data_utils_path = os.path.join(script_directory, "data_utils")
            if os.path.exists(data_utils_path):
                sys.path.append(data_utils_path)
                from v2a_utils.feature_utils_224 import FeaturesUtils
                print("✅ SUCCESS: Found FeaturesUtils in data_utils (fallback)")
                import_success = True
        except ImportError as e2:
            print(f"❌ FAILED: Fallback import - {e2}")
    
    if not import_success:
        print("❌ CRITICAL: Could not import FeaturesUtils from any location!")
        raise ImportError("FeaturesUtils not found in any expected location")
    
    # Try to import models from thinksound module
    print("🔍 DEBUG: Trying models import...")
    try:
        # Import from thinksound module (parent directory is in sys.path)
        from thinksound.models.factory import create_model_from_config
        print("✅ SUCCESS: Found create_model_from_config in thinksound.models.factory")
    except ImportError:
        try:
            from thinksound.models import create_model_from_config
            print("✅ SUCCESS: Found create_model_from_config in thinksound.models")
        except ImportError as e:
            print(f"❌ FAILED: Could not find create_model_from_config: {e}")
            raise
    
    # Try to import utils from thinksound module
    print("🔍 DEBUG: Trying utils import...")
    try:
        from thinksound.models.utils import load_ckpt_state_dict
        print("✅ SUCCESS: Found load_ckpt_state_dict in thinksound.models.utils")
    except ImportError:
        try:
            from thinksound.training.utils import copy_state_dict
            print("✅ SUCCESS: Found copy_state_dict in thinksound.training.utils")
            # Create a wrapper function
            def load_ckpt_state_dict(ckpt_path, device='cpu', prefix=''):
                state_dict = torch.load(ckpt_path, map_location=device)
                if prefix:
                    # Remove prefix from keys
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith(prefix):
                            new_state_dict[k[len(prefix):]] = v
                        else:
                            new_state_dict[k] = v
                    return new_state_dict
                return state_dict
        except ImportError:
            print("⚠️ Using fallback load_ckpt_state_dict")
            def load_ckpt_state_dict(ckpt_path, device='cpu'):
                return torch.load(ckpt_path, map_location=device)
    
    # Try to import sampling functions from thinksound module
    print("🔍 DEBUG: Trying sampling import...")
    try:
        from thinksound.inference.sampling import sample, sample_discrete_euler
        print("✅ SUCCESS: Found sampling functions in thinksound.inference.sampling")
    except ImportError:
        try:
            from thinksound.inference.generate import sample, sample_discrete_euler
            print("✅ SUCCESS: Found sampling functions in thinksound.inference.generate")
        except ImportError as e:
            print(f"❌ FAILED: Could not find sampling functions: {e}")
            raise
    
    THINKSOUND_AVAILABLE = True
    print("🎉 ThinkSound modules imported successfully!")
    
except ImportError as e:
    print(f"❌ WARNING: ThinkSound modules not found: {e}")
    print("📁 Please ensure ThinkSound source code is properly placed")
    print(f"📁 Looking in: {script_directory}")
    
    # Create dummy classes to prevent errors
    class FeaturesUtils:
        def __init__(self, *args, **kwargs):
            raise ImportError("ThinkSound source code not installed. Please download from GitHub.")
    
    def create_model_from_config(*args, **kwargs):
        raise ImportError("ThinkSound source code not installed. Please download from GitHub.")
    
    def load_ckpt_state_dict(*args, **kwargs):
        raise ImportError("ThinkSound source code not installed. Please download from GitHub.")
    
    def sample(*args, **kwargs):
        raise ImportError("ThinkSound source code not installed. Please download from GitHub.")
    
    def sample_discrete_euler(*args, **kwargs):
        raise ImportError("ThinkSound source code not installed. Please download from GitHub.")
    
    THINKSOUND_AVAILABLE = False

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Constants from ThinkSound
_CLIP_SIZE = 224
_CLIP_FPS = 8.0
_SYNC_SIZE = 224  
_SYNC_FPS = 25.0

def process_video_tensor(video_tensor: torch.Tensor, duration_sec: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Process video tensor for both CLIP and sync processing - MMAudio style"""
    
    _CLIP_SIZE = 224
    _CLIP_FPS = 8.0
    _SYNC_SIZE = 224  
    _SYNC_FPS = 25.0
    
    # MMAudio-style transforms (simpler)
    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Handle input dimensions simply
    if len(video_tensor.shape) == 3:  # (H, W, C)
        video_tensor = video_tensor.unsqueeze(0)  # -> (1, H, W, C)
    
    total_frames = video_tensor.shape[0]
    
    # Calculate frame counts
    clip_frames_count = int(_CLIP_FPS * duration_sec)
    sync_frames_count = int(_SYNC_FPS * duration_sec)
    
    # Adjust if video too short
    if total_frames < max(clip_frames_count, sync_frames_count):
        log.warning(f'Video too short: {total_frames} frames for {duration_sec}s')
        actual_duration = total_frames / max(_CLIP_FPS, _SYNC_FPS)
        clip_frames_count = min(clip_frames_count, total_frames)
        sync_frames_count = min(sync_frames_count, total_frames)
        duration_sec = actual_duration
    
    # Extract frames
    clip_frames = video_tensor[:clip_frames_count]  # (T, H, W, C)
    sync_frames = video_tensor[:sync_frames_count]  # (T, H, W, C)
    
    # Convert to (T, C, H, W) format
    clip_frames = clip_frames.permute(0, 3, 1, 2)
    sync_frames = sync_frames.permute(0, 3, 1, 2)
    
    # Apply transforms
    clip_frames = torch.stack([clip_transform(frame) for frame in clip_frames])
    sync_frames = torch.stack([sync_transform(frame) for frame in sync_frames])

    return clip_frames, sync_frames, duration_sec

class ThinkSoundModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "thinksound_model": (folder_paths.get_filename_list("thinksound"), {
                    "tooltip": "ThinkSound main model (.ckpt files from 'ComfyUI/models/thinksound' folder)"
                }),
            },
        }

    RETURN_TYPES = ("THINKSOUND_MODEL",)
    RETURN_NAMES = ("thinksound_model",)
    FUNCTION = "load_model"
    CATEGORY = "ThinkSound"

    # Replace the load_model method in ThinkSoundModelLoader class (around line 285)

    def load_model(self, thinksound_model):
        if not THINKSOUND_AVAILABLE:
            raise ImportError("ThinkSound source code is not installed. Please download the ThinkSound repository from https://github.com/FunAudioLLM/ThinkSound and place it in the ComfyUI-ThinkSound folder.")
            
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        mm.soft_empty_cache()

        # Fixed to fp32 - ThinkSound requires fp32 for proper operation
        base_dtype = torch.float32

        # Load model config - try multiple paths
        config_path = os.path.join(script_directory, "configs", "thinksound.json")
        if not os.path.exists(config_path):
            # Try in thinksound directory
            config_path = os.path.join(script_directory, "thinksound", "configs", "model_configs", "thinksound.json")
            if not os.path.exists(config_path):
                # Use fallback config
                config_path = None
                
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                model_config = json.load(f)
        else:
            # Fallback config
            log.warning("Using fallback model config")
            model_config = {
                "model_type": "thinksound",
                "diffusion_objective": "rectified_flow",
                "io_channels": 64,
                "sample_rate": 44100
            }

        # Create model
        model = create_model_from_config(model_config)
        
        # Load weights
        thinksound_model_path = folder_paths.get_full_path_or_raise("thinksound", thinksound_model)
        model_sd = load_torch_file(thinksound_model_path, device=offload_device)
        
        # 🔧 FIX: Handle different key formats in model checkpoints
        def fix_state_dict_keys(state_dict):
            """Fix state dict keys for different ThinkSound model formats"""
            new_state_dict = {}
            
            # Check if we need to remove prefixes
            sample_key = list(state_dict.keys())[0]
            
            if sample_key.startswith('diffusion.'):
                # Remove 'diffusion.' prefix from all keys
                log.info("Removing 'diffusion.' prefix from model keys")
                for key, value in state_dict.items():
                    if key.startswith('diffusion.'):
                        new_key = key[len('diffusion.'):]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
            elif sample_key.startswith('model.'):
                # Keys are already in correct format
                new_state_dict = state_dict
            else:
                # Might need to add 'model.' prefix - check what the model expects
                model_keys = set(model.state_dict().keys())
                state_keys = set(state_dict.keys())
                
                # If no keys match, try adding 'model.' prefix
                if not model_keys.intersection(state_keys):
                    log.info("Adding 'model.' prefix to model keys")
                    for key, value in state_dict.items():
                        new_state_dict[f'model.{key}'] = value
                else:
                    new_state_dict = state_dict
            
            return new_state_dict
        
        # Apply key fixing
        model_sd = fix_state_dict_keys(model_sd)
        
        # Load with strict=False to handle missing/extra keys gracefully
        try:
            model.load_state_dict(model_sd, strict=False)
            log.info("✅ Model loaded successfully")
        except RuntimeError as e:
            log.error(f"❌ Model loading failed: {e}")
            # Try loading only matching keys
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(model_sd.keys())
            
            log.info(f"Model expects {len(model_keys)} keys, checkpoint has {len(checkpoint_keys)} keys")
            log.info(f"Matching keys: {len(model_keys.intersection(checkpoint_keys))}")
            
            # Load only matching keys
            filtered_sd = {k: v for k, v in model_sd.items() if k in model_keys}
            model.load_state_dict(filtered_sd, strict=False)
            log.warning("⚠️ Loaded model with partial weights")
        
        model = model.eval().to(device=device, dtype=base_dtype)
        
        log.info(f'Loaded ThinkSound model weights from {thinksound_model_path}')
        
        return (model,)

class ThinkSoundFeatureUtilsLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_model": (folder_paths.get_filename_list("thinksound"), {
                    "tooltip": "VAE model (.ckpt files from 'ComfyUI/models/thinksound' folder)"
                }),
                "synchformer_model": (folder_paths.get_filename_list("thinksound"), {
                    "tooltip": "Synchformer model (.pth files from 'ComfyUI/models/thinksound' folder)"
                }),
            },
        }

    RETURN_TYPES = ("THINKSOUND_FEATUREUTILS",)
    RETURN_NAMES = ("feature_utils",)
    FUNCTION = "load_feature_utils"
    CATEGORY = "ThinkSound"

    def load_feature_utils(self, vae_model, synchformer_model):
        if not THINKSOUND_AVAILABLE:
            raise ImportError("ThinkSound source code is not installed. Please download the ThinkSound repository from https://github.com/FunAudioLLM/ThinkSound and place it in the ComfyUI-ThinkSound folder.")
            
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # Fixed to fp32 - ThinkSound requires fp32 for proper operation
        dtype = torch.float32

        # Load VAE
        vae_path = folder_paths.get_full_path_or_raise("thinksound", vae_model)
        
        # Load Synchformer
        synchformer_path = folder_paths.get_full_path_or_raise("thinksound", synchformer_model)
        
        # VAE config path - try multiple locations
        vae_config_path = os.path.join(script_directory, "thinksound", "configs", "model_configs", "stable_audio_2_0_vae.json")
        if not os.path.exists(vae_config_path):
            vae_config_path = "thinksound/configs/model_configs/stable_audio_2_0_vae.json"
        
        # Create feature utils (this will auto-download MetaCLIP if needed)
        # IMPORTANT: Set vae_ckpt=None like in original code - VAE will be loaded separately in main model
        feature_utils = FeaturesUtils(
            vae_ckpt=None,  # ← Set to None like original!
            vae_config=vae_config_path,
            enable_conditions=True,
            synchformer_ckpt=synchformer_path
        ).eval()
        
        # Simple device/dtype conversion like original
        feature_utils = feature_utils.to(device=device, dtype=dtype)
        
        log.info(f'✅ Loaded ThinkSound FeatureUtils on {device} with dtype {dtype}')
        
        return (feature_utils,)

class ThinkSoundSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "thinksound_model": ("THINKSOUND_MODEL",),
                "feature_utils": ("THINKSOUND_FEATUREUTILS",),
                "duration": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 30.0, "step": 0.1, "tooltip": "Duration of the audio in seconds"}),
                "steps": ("INT", {"default": 24, "min": 1, "max": 100, "step": 1, "tooltip": "Number of denoising steps"}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Classifier-free guidance scale"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "caption": ("STRING", {"default": "", "multiline": False, "tooltip": "Short caption describing the desired audio"}),
                "cot_description": ("STRING", {"default": "", "multiline": True, "tooltip": "Chain-of-thought description for detailed audio generation"}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Offload models to save VRAM"}),
            },
            "optional": {
                "video": ("IMAGE", {"tooltip": "Input video frames (optional)"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_audio"
    CATEGORY = "ThinkSound"

    def generate_audio(self, thinksound_model, feature_utils, duration, steps, cfg_scale, seed, caption, cot_description, force_offload, video=None):
        if not THINKSOUND_AVAILABLE:
            raise ImportError("ThinkSound source code is not installed. Please download the ThinkSound repository from https://github.com/FunAudioLLM/ThinkSound and place it in the ComfyUI-ThinkSound folder.")
            
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Process video if provided
        clip_frames = None
        sync_frames = None
        
        if video is not None:
            # Ensure video is in the right format (batch, height, width, channels)
            if len(video.shape) == 3:
                video = video.unsqueeze(0)  # Add batch dimension
            
            video_tensor = video.cpu()  # Ensure video is on CPU for processing
            clip_frames, sync_frames, actual_duration = process_video_tensor(video_tensor, duration)
            duration = actual_duration
            
            # Ensure correct shape: clip_frames should be (T, C, H, W), then we add batch dim
            if len(clip_frames.shape) == 4:  # (T, C, H, W)
                clip_frames = clip_frames.unsqueeze(0)  # -> (1, T, C, H, W)
            elif len(clip_frames.shape) == 5:  # Already (B, T, C, H, W)
                pass  # Keep as is
            else:
                raise ValueError(f"Unexpected clip_frames shape: {clip_frames.shape}")
            
            if len(sync_frames.shape) == 4:  # (T, C, H, W) 
                sync_frames = sync_frames.unsqueeze(0)  # -> (1, T, C, H, W)
            elif len(sync_frames.shape) == 5:  # Already (B, T, C, H, W)
                pass  # Keep as is
            else:
                raise ValueError(f"Unexpected sync_frames shape: {sync_frames.shape}")
            
            # Move to device
            clip_frames = clip_frames.to(device)
            sync_frames = sync_frames.to(device)
            
            log.info(f"Processed video: clip_frames {clip_frames.shape}, sync_frames {sync_frames.shape}, duration {duration}")

        # Use CoT description if provided, otherwise use caption
        if cot_description.strip():
            cot = cot_description.strip()
        else:
            cot = caption.strip() if caption.strip() else "Generate audio"

        # Prepare data for processing
        data = {
            'caption': caption.strip() if caption.strip() else "",
            'caption_cot': cot,
            'clip_video': clip_frames,
            'sync_video': sync_frames,
        }

        # Move models to device (simple approach like original)
        feature_utils = feature_utils.to(device)
        thinksound_model = thinksound_model.to(device)
        
        log.info(f"Models moved to {device}")

        # Process features (simple approach like original)
        preprocessed_data = {}
        
        # Text features
        metaclip_global_text_features, metaclip_text_features = feature_utils.encode_text(data['caption'])
        preprocessed_data['metaclip_global_text_features'] = metaclip_global_text_features.detach().cpu().squeeze(0)
        preprocessed_data['metaclip_text_features'] = metaclip_text_features.detach().cpu().squeeze(0)

        t5_features = feature_utils.encode_t5_text(data['caption_cot'])
        preprocessed_data['t5_features'] = t5_features.detach().cpu().squeeze(0)

        # Process features using MMAudio approach - simpler and cleaner
        preprocessed_data = {}
        
        # Text features (same as before)
        metaclip_global_text_features, metaclip_text_features = feature_utils.encode_text(data['caption'])
        preprocessed_data['metaclip_global_text_features'] = metaclip_global_text_features.detach().cpu().squeeze(0)
        preprocessed_data['metaclip_text_features'] = metaclip_text_features.detach().cpu().squeeze(0)

        t5_features = feature_utils.encode_t5_text(data['caption_cot'])
        preprocessed_data['t5_features'] = t5_features.detach().cpu().squeeze(0)

        # Video features - MMAudio style processing
        if clip_frames is not None:
            # Process features directly without complex dimension handling
            clip_features = feature_utils.encode_video_with_clip(clip_frames)
            sync_features = feature_utils.encode_video_with_sync(sync_frames)
            
            # Store with proper dimensions (remove batch dim for metadata)
            preprocessed_data['metaclip_features'] = clip_features.detach().cpu().squeeze(0)
            preprocessed_data['sync_features'] = sync_features.detach().cpu().squeeze(0)
            preprocessed_data['video_exist'] = torch.tensor(True)
            
            log.info(f"Stored features - clip: {preprocessed_data['metaclip_features'].shape}, sync: {preprocessed_data['sync_features'].shape}")
        else:
            preprocessed_data['video_exist'] = torch.tensor(False)
            log.info("No video provided - using text-only generation")

        # Update sequence lengths
        if 'metaclip_features' in preprocessed_data:
            sync_seq_len = preprocessed_data['sync_features'].shape[0]
            clip_seq_len = preprocessed_data['metaclip_features'].shape[0]
        else:
            sync_seq_len = int(_SYNC_FPS * duration)
            clip_seq_len = int(_CLIP_FPS * duration)
            
        latent_seq_len = int(194/9 * duration)
        thinksound_model.model.model.update_seq_lengths(latent_seq_len, clip_seq_len, sync_seq_len)

        metadata = [preprocessed_data]

        # Generate conditioning (simple approach like original)
        with torch.amp.autocast(device_type=device.type):
            conditioning = thinksound_model.conditioner(metadata, device)
        
        # Handle missing video like in original code - but with better debugging
        video_exist = torch.stack([item['video_exist'] for item in metadata], dim=0)
        log.info(f"Video exist tensor: {video_exist}")
        
        # Debug conditioning shapes before empty feature handling
        for key, value in conditioning.items():
            if torch.is_tensor(value):
                log.info(f"Conditioning[{key}] shape: {value.shape}")
        
        # Check if model has empty features
        if hasattr(thinksound_model.model.model, 'empty_clip_feat'):
            log.info(f"Model empty_clip_feat shape: {thinksound_model.model.model.empty_clip_feat.shape}")
            log.info(f"Model empty_sync_feat shape: {thinksound_model.model.model.empty_sync_feat.shape}")
            
            # Only apply empty features if we actually have missing video
            if not video_exist.all():
                log.info("Some video missing - applying empty features")
                if 'metaclip_features' in conditioning:
                    log.info(f"Before: conditioning['metaclip_features'] shape: {conditioning['metaclip_features'].shape}")
                    conditioning['metaclip_features'][~video_exist] = thinksound_model.model.model.empty_clip_feat
                    log.info(f"After: conditioning['metaclip_features'] shape: {conditioning['metaclip_features'].shape}")
                
                if 'sync_features' in conditioning:
                    log.info(f"Before: conditioning['sync_features'] shape: {conditioning['sync_features'].shape}")
                    conditioning['sync_features'][~video_exist] = thinksound_model.model.model.empty_sync_feat
                    log.info(f"After: conditioning['sync_features'] shape: {conditioning['sync_features'].shape}")
            else:
                log.info("All video present - skipping empty feature application")
        else:
            log.warning("Model does not have empty features")

        # Generate audio (simple approach like original)
        cond_inputs = thinksound_model.get_conditioning_inputs(conditioning)
        noise = torch.randn([1, thinksound_model.io_channels, latent_seq_len], device=device)
        
        with torch.amp.autocast(device_type=device.type):
            if thinksound_model.diffusion_objective == "v":
                fakes = sample(thinksound_model.model, noise, steps, 0, **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True)
            elif thinksound_model.diffusion_objective == "rectified_flow":
                fakes = sample_discrete_euler(thinksound_model.model, noise, steps, **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True)
            else:
                raise ValueError(f"Unknown diffusion objective: {thinksound_model.diffusion_objective}")
                
        # Decode audio
        if thinksound_model.pretransform is not None:
            fakes = thinksound_model.pretransform.decode(fakes)

        # Convert to audio format
        audios = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).clamp(-1, 1).cpu()
        
        # Offload models if requested
        if force_offload:
            thinksound_model.to(offload_device)
            feature_utils.to(offload_device)
            mm.soft_empty_cache()

        # Prepare audio output for ComfyUI
        audio = {
            "waveform": audios,
            "sample_rate": 44100
        }

        log.info(f"Generated audio with shape {audios.shape} at {44100}Hz")
        
        return (audio,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ThinkSoundModelLoader": ThinkSoundModelLoader,
    "ThinkSoundFeatureUtilsLoader": ThinkSoundFeatureUtilsLoader, 
    "ThinkSoundSampler": ThinkSoundSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThinkSoundModelLoader": "ThinkSound Model Loader",
    "ThinkSoundFeatureUtilsLoader": "ThinkSound Feature Utils Loader",
    "ThinkSoundSampler": "ThinkSound Sampler",
}