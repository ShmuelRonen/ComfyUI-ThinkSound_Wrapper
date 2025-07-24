import os
import torch
import json
import numpy as np
import tempfile
import shutil
import time
from typing import Optional, Tuple, Dict, Any
from torchvision.transforms import v2
import torch.nn.functional as F
from transformers import AutoProcessor

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file

# Enhanced logging setup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))

# Add model folder for ThinkSound
if not "thinksound" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("thinksound", os.path.join(folder_paths.models_dir, "thinksound"))

# Enhanced ThinkSound module import with better error handling
print("🔍 DEBUG: Starting enhanced ThinkSound import process...")
print(f"🔍 DEBUG: Script directory = {script_directory}")

def safe_import_with_fallbacks():
    """Enhanced import function with multiple fallback strategies"""
    import sys
    
    # Add ThinkSound directory to Python path
    thinksound_path = script_directory
    if thinksound_path not in sys.path:
        sys.path.append(thinksound_path)
    print(f"🔍 DEBUG: Added to sys.path: {thinksound_path}")
    
    # Check directory structure
    print("🔍 DEBUG: Enhanced directory analysis:")
    try:
        contents = os.listdir(script_directory)
        for item in contents:
            item_path = os.path.join(script_directory, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
                # Show nested structure for important folders
                if item in ['thinksound', 'ThinkSound', 'data_utils']:
                    try:
                        nested = os.listdir(item_path)[:5]  # Show first 5 items
                        for nested_item in nested:
                            print(f"    📄 {nested_item}")
                        if len(os.listdir(item_path)) > 5:
                            print(f"    ... and {len(os.listdir(item_path)) - 5} more items")
                    except:
                        pass
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"  ❌ Error listing directory: {e}")
    
    # Enhanced import strategies
    import_results = {}
    
    try:
        # Strategy 1: Try thinksound module with alias
        thinksound_subfolder = os.path.join(script_directory, "thinksound")
        if os.path.exists(thinksound_subfolder):
            import thinksound
            sys.modules['ThinkSound'] = thinksound
            print("✅ Created ThinkSound alias for thinksound module")
            
            # Import FeaturesUtils
            from thinksound.data.v2a_utils.feature_utils_224 import FeaturesUtils
            import_results['FeaturesUtils'] = FeaturesUtils
            print("✅ SUCCESS: FeaturesUtils imported from thinksound.data.v2a_utils.feature_utils_224")
            
        else:
            # Strategy 2: Try data_utils fallback
            data_utils_path = os.path.join(script_directory, "data_utils")
            if os.path.exists(data_utils_path):
                sys.path.append(data_utils_path)
                from v2a_utils.feature_utils_224 import FeaturesUtils
                import_results['FeaturesUtils'] = FeaturesUtils
                print("✅ SUCCESS: FeaturesUtils imported from data_utils (fallback)")
    except ImportError as e:
        print(f"❌ FeaturesUtils import failed: {e}")
        import_results['FeaturesUtils'] = None
    
    # Import model creation functions with enhanced error handling
    try:
        from thinksound.models.factory import create_model_from_config
        import_results['create_model_from_config'] = create_model_from_config
        print("✅ SUCCESS: create_model_from_config from thinksound.models.factory")
    except ImportError:
        try:
            from thinksound.models import create_model_from_config
            import_results['create_model_from_config'] = create_model_from_config
            print("✅ SUCCESS: create_model_from_config from thinksound.models")
        except ImportError as e:
            print(f"❌ create_model_from_config import failed: {e}")
            import_results['create_model_from_config'] = None
    
    # Import utilities with enhanced fallbacks
    try:
        from thinksound.models.utils import load_ckpt_state_dict
        import_results['load_ckpt_state_dict'] = load_ckpt_state_dict
        print("✅ SUCCESS: load_ckpt_state_dict from thinksound.models.utils")
    except ImportError:
        # Enhanced fallback function
        def enhanced_load_ckpt_state_dict(ckpt_path, device='cpu', prefix=''):
            """Enhanced checkpoint loading with better error handling"""
            try:
                state_dict = torch.load(ckpt_path, map_location=device)
                if prefix:
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith(prefix):
                            new_state_dict[k[len(prefix):]] = v
                        else:
                            new_state_dict[k] = v
                    return new_state_dict
                return state_dict
            except Exception as e:
                log.error(f"Enhanced checkpoint loading failed: {e}")
                raise
        
        import_results['load_ckpt_state_dict'] = enhanced_load_ckpt_state_dict
        print("✅ Using enhanced fallback load_ckpt_state_dict")
    
    # Import sampling functions with fallbacks
    try:
        from thinksound.inference.sampling import sample, sample_discrete_euler
        import_results['sample'] = sample
        import_results['sample_discrete_euler'] = sample_discrete_euler
        print("✅ SUCCESS: Sampling functions from thinksound.inference.sampling")
    except ImportError:
        try:
            from thinksound.inference.generate import sample, sample_discrete_euler
            import_results['sample'] = sample
            import_results['sample_discrete_euler'] = sample_discrete_euler
            print("✅ SUCCESS: Sampling functions from thinksound.inference.generate")
        except ImportError as e:
            print(f"❌ Sampling functions import failed: {e}")
            import_results['sample'] = None
            import_results['sample_discrete_euler'] = None
    
    return import_results

# Execute enhanced imports
try:
    imports = safe_import_with_fallbacks()
    
    FeaturesUtils = imports['FeaturesUtils']
    create_model_from_config = imports['create_model_from_config']
    load_ckpt_state_dict = imports['load_ckpt_state_dict']
    sample = imports['sample']
    sample_discrete_euler = imports['sample_discrete_euler']
    
    # Check if all imports succeeded
    missing_imports = [k for k, v in imports.items() if v is None]
    if missing_imports:
        raise ImportError(f"Missing critical imports: {missing_imports}")
    
    THINKSOUND_AVAILABLE = True
    print("🎉 Enhanced ThinkSound modules imported successfully!")
    
except Exception as e:
    print(f"❌ CRITICAL: Enhanced ThinkSound import failed: {e}")
    print("📁 Please ensure ThinkSound source code is properly placed")
    print(f"📁 Looking in: {script_directory}")
    
    # Enhanced dummy classes with better error messages
    class FeaturesUtils:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ThinkSound source code not installed. "
                "Please download from https://github.com/FunAudioLLM/ThinkSound "
                f"and place in {script_directory}"
            )
    
    def create_model_from_config(*args, **kwargs):
        raise ImportError("ThinkSound models module not available")
    
    def load_ckpt_state_dict(*args, **kwargs):
        raise ImportError("ThinkSound utils module not available")
    
    def sample(*args, **kwargs):
        raise ImportError("ThinkSound sampling module not available")
    
    def sample_discrete_euler(*args, **kwargs):
        raise ImportError("ThinkSound sampling module not available")
    
    THINKSOUND_AVAILABLE = False

# Enhanced constants with validation
_CLIP_SIZE = 224
_CLIP_FPS = 8.0
_SYNC_SIZE = 224  
_SYNC_FPS = 25.0

def validate_video_tensor(video_tensor: torch.Tensor) -> torch.Tensor:
    """Enhanced video tensor validation with automatic fixes"""
    if video_tensor is None:
        return None
    
    original_shape = video_tensor.shape
    log.info(f"Input video tensor shape: {original_shape}")
    
    # Handle different input formats
    if len(video_tensor.shape) == 3:  # (H, W, C)
        video_tensor = video_tensor.unsqueeze(0)  # -> (1, H, W, C)
        log.info("Added time dimension to single frame")
    elif len(video_tensor.shape) == 5:  # (B, T, H, W, C)
        if video_tensor.shape[0] == 1:
            video_tensor = video_tensor.squeeze(0)  # -> (T, H, W, C)
            log.info("Removed batch dimension")
    
    # Ensure we have (T, H, W, C) format
    if len(video_tensor.shape) != 4:
        raise ValueError(f"Expected 4D tensor (T, H, W, C), got {video_tensor.shape}")
    
    # Validate channel dimension
    if video_tensor.shape[-1] not in [1, 3, 4]:
        log.warning(f"Unusual channel count: {video_tensor.shape[-1]}")
    
    # Ensure float32 and proper range
    if video_tensor.dtype != torch.float32:
        video_tensor = video_tensor.to(torch.float32)
        log.info(f"Converted dtype to float32")
    
    # Normalize to [0, 1] range if needed
    if video_tensor.max() > 1.1 or video_tensor.min() < -0.1:
        if video_tensor.max() > 10:  # Likely 0-255 range
            video_tensor = video_tensor / 255.0
            log.info("Normalized from [0, 255] to [0, 1] range")
        else:
            video_tensor = torch.clamp(video_tensor, 0, 1)
            log.info("Clamped to [0, 1] range")
    
    log.info(f"Final video tensor shape: {video_tensor.shape}")
    return video_tensor

def enhanced_pad_to_square(video_tensor: torch.Tensor) -> torch.Tensor:
    """Enhanced padding with better error handling"""
    if len(video_tensor.shape) != 4:
        raise ValueError(f"Expected 4D tensor (T, C, H, W), got {video_tensor.shape}")

    t, c, h, w = video_tensor.shape
    max_side = max(h, w)

    if h == w:
        return video_tensor  # Already square

    pad_h = max_side - h
    pad_w = max_side - w
    
    # Use symmetric padding
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    
    log.info(f"Padding video from ({h}, {w}) to ({max_side}, {max_side})")
    video_padded = F.pad(video_tensor, pad=padding, mode='constant', value=0)

    return video_padded

def enhanced_process_video_tensor(video_tensor: torch.Tensor, duration_sec: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Enhanced video processing with intelligent frame handling"""
    
    # Validate input
    video_tensor = validate_video_tensor(video_tensor)
    if video_tensor is None:
        return None, None, duration_sec
    
    # Enhanced transforms with better error handling
    try:
        clip_transform = v2.Compose([
            v2.Lambda(enhanced_pad_to_square),
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
        
        total_frames = video_tensor.shape[0]
        
        # Calculate frame counts with intelligent adjustments
        clip_frames_count = int(_CLIP_FPS * duration_sec)
        sync_frames_count = int(_SYNC_FPS * duration_sec)
        
        # Enhanced handling for short videos
        if total_frames < max(clip_frames_count, sync_frames_count):
            log.warning(f'Video too short: {total_frames} frames for {duration_sec}s')
            # Calculate what duration we can actually support
            max_possible_duration = total_frames / max(_CLIP_FPS, _SYNC_FPS)
            if max_possible_duration < duration_sec * 0.5:  # Less than half requested duration
                log.error(f"Video too short: {max_possible_duration:.2f}s available, {duration_sec}s requested")
                # Use minimum viable duration
                duration_sec = max(1.0, max_possible_duration)
            else:
                duration_sec = max_possible_duration
            
            clip_frames_count = min(clip_frames_count, total_frames)
            sync_frames_count = min(sync_frames_count, total_frames)
        
        # Enhanced frame extraction with padding logic from enhanced app.py
        def extract_frames_with_padding(total_frames: int, target_count: int, max_padding: int = 12) -> torch.Tensor:
            if total_frames >= target_count:
                # Simple case: enough frames available
                indices = torch.linspace(0, total_frames - 1, target_count).long()
                return video_tensor[indices]
            else:
                # Need padding - use all available frames plus padding
                frames = video_tensor  # Use all available frames
                
                padding_needed = target_count - total_frames
                if padding_needed > max_padding:
                    log.warning(f"Excessive padding needed: {padding_needed} > {max_padding}")
                    # Use only what we can reasonably pad
                    target_count = total_frames + max_padding
                    padding_needed = max_padding
                
                if padding_needed > 0:
                    last_frame = frames[-1:].expand(padding_needed, -1, -1, -1)
                    frames = torch.cat([frames, last_frame], dim=0)
                    log.info(f"Added {padding_needed} padding frames")
                
                return frames
        
        # Extract frames with enhanced padding
        clip_frames = extract_frames_with_padding(total_frames, clip_frames_count, max_padding=4)
        sync_frames = extract_frames_with_padding(total_frames, sync_frames_count, max_padding=12)
        
        # Convert to (T, C, H, W) format
        clip_frames = clip_frames.permute(0, 3, 1, 2)
        sync_frames = sync_frames.permute(0, 3, 1, 2)
        
        # Apply transforms with error handling
        try:
            clip_frames = torch.stack([clip_transform(frame) for frame in clip_frames])
        except Exception as e:
            log.error(f"CLIP transform failed: {e}")
            # Fallback: simple resize
            clip_frames = F.interpolate(clip_frames, size=(_CLIP_SIZE, _CLIP_SIZE), mode='bilinear')
        
        try:
            sync_frames = torch.stack([sync_transform(frame) for frame in sync_frames])
        except Exception as e:
            log.error(f"Sync transform failed: {e}")
            # Fallback: simple resize and normalize
            sync_frames = F.interpolate(sync_frames, size=(_SYNC_SIZE, _SYNC_SIZE), mode='bilinear')
            sync_frames = (sync_frames - 0.5) / 0.5  # Normalize to [-1, 1]

        log.info(f"Enhanced processing complete: clip {clip_frames.shape}, sync {sync_frames.shape}, duration {duration_sec:.2f}s")
        return clip_frames, sync_frames, duration_sec
        
    except Exception as e:
        log.error(f"Enhanced video processing failed: {e}")
        raise

class ThinkSoundModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "thinksound_model": (folder_paths.get_filename_list("thinksound"), {
                    "tooltip": "ThinkSound main model (.ckpt files from 'ComfyUI/models/thinksound' folder)"
                }),
                "precision": (["fp32", "fp16"], {
                    "default": "fp32",
                    "tooltip": "Model precision (fp32 recommended for stability)"
                }),
                "offload_device": (["cpu", "auto"], {
                    "default": "auto", 
                    "tooltip": "Device to offload model when not in use"
                }),
            },
        }

    RETURN_TYPES = ("THINKSOUND_MODEL",)
    RETURN_NAMES = ("thinksound_model",)
    FUNCTION = "load_model"
    CATEGORY = "ThinkSound"

    def load_model(self, thinksound_model, precision="fp32", offload_device="auto"):
        if not THINKSOUND_AVAILABLE:
            raise ImportError(
                "ThinkSound source code is not installed. "
                "Please download from https://github.com/FunAudioLLM/ThinkSound "
                f"and place in {script_directory}"
            )
            
        device = mm.get_torch_device()
        if offload_device == "auto":
            offload_device = mm.unet_offload_device()
        else:
            offload_device = torch.device(offload_device)
        
        mm.soft_empty_cache()

        # Enhanced precision handling
        if precision == "fp16" and device.type == "cuda":
            base_dtype = torch.float16
            log.info("Using fp16 precision for CUDA")
        else:
            base_dtype = torch.float32
            log.info("Using fp32 precision (recommended)")

        # Enhanced config loading with multiple fallback paths
        config_paths = [
            os.path.join(script_directory, "configs", "thinksound.json"),
            os.path.join(script_directory, "thinksound", "configs", "model_configs", "thinksound.json"),
            os.path.join(script_directory, "ThinkSound", "configs", "model_configs", "thinksound.json"),
        ]
        
        model_config = None
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        model_config = json.load(f)
                    log.info(f"✅ Loaded config from: {config_path}")
                    break
                except Exception as e:
                    log.warning(f"Failed to load config from {config_path}: {e}")
        
        if model_config is None:
            # Enhanced fallback config
            log.warning("Using enhanced fallback model config")
            model_config = {
                "model_type": "thinksound",
                "diffusion_objective": "rectified_flow", 
                "io_channels": 64,
                "sample_rate": 44100,
                "audio_channels": 2,
                "model": {
                    "pretransform": {
                        "type": "autoencoder"
                    }
                }
            }

        # Enhanced model creation with error handling
        try:
            model = create_model_from_config(model_config)
            log.info("✅ Model created from config")
        except Exception as e:
            log.error(f"❌ Model creation failed: {e}")
            raise RuntimeError(f"Failed to create ThinkSound model: {e}")
        
        # Enhanced weight loading
        thinksound_model_path = folder_paths.get_full_path_or_raise("thinksound", thinksound_model)
        
        try:
            model_sd = load_torch_file(thinksound_model_path, device=offload_device)
            log.info(f"✅ Loaded checkpoint from: {thinksound_model_path}")
        except Exception as e:
            log.error(f"❌ Checkpoint loading failed: {e}")
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Enhanced state dict key fixing
        def enhanced_fix_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Enhanced state dict key fixing with comprehensive pattern matching"""
            if not state_dict:
                raise ValueError("Empty state dict")
            
            new_state_dict = {}
            sample_keys = list(state_dict.keys())[:5]  # Check first 5 keys
            
            log.info(f"Sample checkpoint keys: {sample_keys}")
            
            # Pattern detection
            prefixes_to_remove = ['diffusion.', 'model.diffusion.', 'thinksound.']
            prefixes_to_add = ['model.']
            
            # Check for prefix removal
            for prefix in prefixes_to_remove:
                if any(key.startswith(prefix) for key in sample_keys):
                    log.info(f"Removing '{prefix}' prefix from model keys")
                    for key, value in state_dict.items():
                        if key.startswith(prefix):
                            new_key = key[len(prefix):]
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    return new_state_dict
            
            # Check if we need to add prefix
            model_keys = set(model.state_dict().keys())
            state_keys = set(state_dict.keys())
            matching_keys = model_keys.intersection(state_keys)
            
            log.info(f"Model keys: {len(model_keys)}, State keys: {len(state_keys)}, Matching: {len(matching_keys)}")
            
            if len(matching_keys) < len(model_keys) * 0.5:  # Less than 50% match
                for prefix in prefixes_to_add:
                    # Try adding prefix
                    test_keys = set(f"{prefix}{key}" for key in state_keys)
                    test_matching = model_keys.intersection(test_keys)
                    
                    if len(test_matching) > len(matching_keys):
                        log.info(f"Adding '{prefix}' prefix to model keys")
                        for key, value in state_dict.items():
                            new_state_dict[f"{prefix}{key}"] = value
                        return new_state_dict
            
            return state_dict
        
        # Apply enhanced key fixing
        try:  
            model_sd = enhanced_fix_state_dict_keys(model_sd)
            log.info("✅ State dict keys processed")
        except Exception as e:
            log.error(f"❌ State dict key fixing failed: {e}")
            # Continue with original keys
        
        # Enhanced model loading with detailed error reporting
        try:
            missing_keys, unexpected_keys = model.load_state_dict(model_sd, strict=False)
            
            if missing_keys:
                log.warning(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                log.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
            
            log.info("✅ Model weights loaded successfully")
            
        except Exception as e:
            log.error(f"❌ Model loading failed: {e}")
            # Try loading only compatible keys
            model_state = model.state_dict()
            compatible_sd = {k: v for k, v in model_sd.items() if k in model_state and v.shape == model_state[k].shape}
            
            if compatible_sd:
                model.load_state_dict(compatible_sd, strict=False)
                log.warning(f"⚠️ Loaded {len(compatible_sd)}/{len(model_state)} compatible weights")
            else:
                raise RuntimeError(f"No compatible weights found: {e}")
        
        # Move to device with proper error handling
        try:
            model = model.eval().to(device=device, dtype=base_dtype)
            log.info(f'✅ Model loaded on {device} with dtype {base_dtype}')
        except Exception as e:
            log.error(f"❌ Device transfer failed: {e}")
            raise
        
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
                "precision": (["fp32", "fp16"], {
                    "default": "fp32", 
                    "tooltip": "Feature extraction precision (fp32 recommended)"
                }),
                "enable_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable model offloading to save VRAM"
                }),
            },
        }

    RETURN_TYPES = ("THINKSOUND_FEATUREUTILS",)
    RETURN_NAMES = ("feature_utils",)
    FUNCTION = "load_feature_utils"
    CATEGORY = "ThinkSound"

    def load_feature_utils(self, vae_model, synchformer_model, precision="fp32", enable_offload=True):
        if not THINKSOUND_AVAILABLE:
            raise ImportError(
                "ThinkSound source code is not installed. "
                "Please download from https://github.com/FunAudioLLM/ThinkSound "  
                f"and place in {script_directory}"
            )
            
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device() if enable_offload else device

        # Enhanced precision handling
        if precision == "fp16" and device.type == "cuda":
            dtype = torch.float16
            log.info("Using fp16 precision for feature extraction")
        else:
            dtype = torch.float32
            log.info("Using fp32 precision for feature extraction")

        # Enhanced file validation
        try:
            vae_path = folder_paths.get_full_path_or_raise("thinksound", vae_model)
            if not os.path.exists(vae_path):
                raise FileNotFoundError(f"VAE model not found: {vae_path}")
            
            synchformer_path = folder_paths.get_full_path_or_raise("thinksound", synchformer_model)
            if not os.path.exists(synchformer_path):
                raise FileNotFoundError(f"Synchformer model not found: {synchformer_path}")
            
            log.info(f"✅ VAE path: {vae_path}")
            log.info(f"✅ Synchformer path: {synchformer_path}")
            
        except Exception as e:
            log.error(f"❌ Model file validation failed: {e}")
            raise
        
        # Enhanced VAE config search
        vae_config_paths = [
            os.path.join(script_directory, "configs", "stable_audio_2_0_vae.json"),
            os.path.join(script_directory, "thinksound", "configs", "model_configs", "stable_audio_2_0_vae.json"),
            os.path.join(script_directory, "ThinkSound", "configs", "model_configs", "stable_audio_2_0_vae.json"),
            "thinksound/configs/model_configs/stable_audio_2_0_vae.json",  # Relative path fallback
        ]
        
        vae_config_path = None
        for config_path in vae_config_paths:
            if os.path.exists(config_path):
                vae_config_path = config_path
                log.info(f"✅ Found VAE config: {config_path}")
                break
        
        if vae_config_path is None:
            log.warning("⚠️ VAE config not found, using relative path fallback")
            vae_config_path = "thinksound/configs/model_configs/stable_audio_2_0_vae.json"

        # Enhanced feature utils creation with comprehensive error handling
        try:
            feature_utils = FeaturesUtils(
                vae_ckpt=None,  # Important: Set to None as in original
                vae_config=vae_config_path,
                enable_conditions=True,
                synchformer_ckpt=synchformer_path
            ).eval()
            
            log.info("✅ FeatureUtils created successfully")
            
        except Exception as e:
            log.error(f"❌ FeatureUtils creation failed: {e}")
            # Try with absolute paths
            try:
                abs_vae_config = os.path.abspath(vae_config_path) if vae_config_path else None
                abs_synchformer = os.path.abspath(synchformer_path)
                
                feature_utils = FeaturesUtils(
                    vae_ckpt=None,
                    vae_config=abs_vae_config,
                    enable_conditions=True,
                    synchformer_ckpt=abs_synchformer
                ).eval()
                
                log.info("✅ FeatureUtils created with absolute paths")
                
            except Exception as e2:
                log.error(f"❌ FeatureUtils creation failed even with absolute paths: {e2}")
                raise RuntimeError(f"Failed to create FeatureUtils: {e2}")
        
        # Enhanced device/dtype management
        try:
            feature_utils = feature_utils.to(device=device, dtype=dtype)
            log.info(f'✅ FeatureUtils loaded on {device} with dtype {dtype}')
            
            # Test basic functionality
            test_text = "test"
            try:
                with torch.no_grad():
                    _ = feature_utils.encode_text(test_text)
                log.info("✅ FeatureUtils functionality verified")
            except Exception as e:
                log.warning(f"⚠️ FeatureUtils test failed: {e}")
            
        except Exception as e:
            log.error(f"❌ Device transfer failed: {e}")
            raise
        
        return (feature_utils,)

class ThinkSoundSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "thinksound_model": ("THINKSOUND_MODEL",),
                "feature_utils": ("THINKSOUND_FEATUREUTILS",),
                "duration": ("FLOAT", {
                    "default": 8.0, 
                    "min": 1.0, 
                    "max": 30.0, 
                    "step": 0.1, 
                    "tooltip": "Duration of generated audio in seconds"
                }),
                "steps": ("INT", {
                    "default": 24, 
                    "min": 1, 
                    "max": 100, 
                    "step": 1, 
                    "tooltip": "Number of denoising steps (more = better quality, slower)"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 5.0, 
                    "min": 1.0, 
                    "max": 20.0, 
                    "step": 0.1, 
                    "tooltip": "Classifier-free guidance scale (higher = more faithful to text)"
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible results"
                }),
                "caption": ("STRING", {
                    "default": "", 
                    "multiline": False, 
                    "tooltip": "Short description of desired audio (e.g., 'dog barking', 'ocean waves')"
                }),
                "cot_description": ("STRING", {
                    "default": "", 
                    "multiline": True, 
                    "tooltip": "Detailed chain-of-thought description for enhanced audio generation"
                }),
                "force_offload": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Offload models after generation to save VRAM"
                }),
                "performance_mode": (["balanced", "quality", "speed"], {
                    "default": "balanced",
                    "tooltip": "Generation performance profile"
                }),
            },
            "optional": {
                "video": ("IMAGE", {
                    "tooltip": "Input video frames for video-to-audio generation (optional)"
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_audio"
    CATEGORY = "ThinkSound"

    def generate_audio(self, thinksound_model, feature_utils, duration, steps, cfg_scale, seed, 
                      caption, cot_description, force_offload, performance_mode="balanced", video=None):
        if not THINKSOUND_AVAILABLE:
            raise ImportError(
                "ThinkSound source code is not installed. " 
                "Please download from https://github.com/FunAudioLLM/ThinkSound "
                f"and place in {script_directory}"
            )
            
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        # Enhanced performance mode settings
        if performance_mode == "quality":
            steps = max(steps, 32)  # Ensure minimum quality steps
            cfg_scale = max(cfg_scale, 7.0)  # Higher guidance
            log.info("🎯 Quality mode: Enhanced settings for better results")
        elif performance_mode == "speed":
            steps = min(steps, 16)  # Faster generation
            cfg_scale = min(cfg_scale, 3.0)  # Lower guidance for speed
            log.info("⚡ Speed mode: Optimized for faster generation")
        else:
            log.info("⚖️ Balanced mode: Standard quality/speed balance")
        
        # Enhanced seed management
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))
        log.info(f"🎲 Set random seed: {seed}")

        # Enhanced video processing with comprehensive error handling
        clip_frames = None
        sync_frames = None
        
        if video is not None:
            try:
                log.info(f"🎬 Processing input video: {video.shape}")
                video_tensor = video.cpu()  # Move to CPU for processing
                clip_frames, sync_frames, actual_duration = enhanced_process_video_tensor(video_tensor, duration)
                
                if actual_duration != duration:
                    log.info(f"📏 Duration adjusted: {duration:.2f}s → {actual_duration:.2f}s")
                    duration = actual_duration
                
                # Enhanced shape handling with validation
                def ensure_batch_dimension(tensor, name):
                    if tensor is None:
                        return None
                    
                    original_shape = tensor.shape
                    if len(tensor.shape) == 4:  # (T, C, H, W)
                        tensor = tensor.unsqueeze(0)  # -> (1, T, C, H, W)
                        log.info(f"Added batch dimension to {name}: {original_shape} → {tensor.shape}")
                    elif len(tensor.shape) == 5:  # Already (B, T, C, H, W)
                        if tensor.shape[0] != 1:
                            log.warning(f"Unexpected batch size for {name}: {tensor.shape[0]}")
                    else:
                        raise ValueError(f"Unexpected {name} shape: {tensor.shape}")
                    
                    return tensor.to(device)
                
                clip_frames = ensure_batch_dimension(clip_frames, "clip_frames")
                sync_frames = ensure_batch_dimension(sync_frames, "sync_frames")
                
                log.info(f"✅ Video processed: clip {clip_frames.shape}, sync {sync_frames.shape}")
                
            except Exception as e:
                log.error(f"❌ Video processing failed: {e}")
                log.warning("⚠️ Continuing with text-only generation")
                clip_frames = None
                sync_frames = None

        # Enhanced text processing
        if not caption.strip() and not cot_description.strip():
            log.warning("⚠️ No text input provided, using default")
            caption = "Generate audio"
            cot_description = "Generate appropriate audio for the given context"
        
        final_caption = caption.strip()
        final_cot = cot_description.strip() if cot_description.strip() else final_caption
        
        log.info(f"📝 Caption: '{final_caption}'")
        log.info(f"🧠 CoT: '{final_cot[:100]}{'...' if len(final_cot) > 100 else ''}'")

        # Enhanced model device management
        start_time = time.time()
        
        try:
            feature_utils = feature_utils.to(device)
            thinksound_model = thinksound_model.to(device)
            log.info(f"📱 Models moved to {device}")
        except Exception as e:
            log.error(f"❌ Model device transfer failed: {e}")
            raise

        # Enhanced feature extraction with comprehensive error handling
        try:
            log.info("🔍 Extracting text features...")
            
            # Text features with error handling
            try:
                metaclip_global_text_features, metaclip_text_features = feature_utils.encode_text(final_caption)
                t5_features = feature_utils.encode_t5_text(final_cot)
                
                log.info(f"✅ Text features extracted: MetaCLIP {metaclip_text_features.shape}, T5 {t5_features.shape}")
            except Exception as e:
                log.error(f"❌ Text feature extraction failed: {e}")
                raise RuntimeError(f"Text processing failed: {e}")
            
            # Prepare metadata
            preprocessed_data = {
                'metaclip_global_text_features': metaclip_global_text_features.detach().cpu().squeeze(0),
                'metaclip_text_features': metaclip_text_features.detach().cpu().squeeze(0),
                't5_features': t5_features.detach().cpu().squeeze(0),
                'video_exist': torch.tensor(clip_frames is not None),
            }
            
            # Enhanced video feature extraction
            if clip_frames is not None:
                try:
                    log.info("🎬 Extracting video features...")
                    
                    clip_features = feature_utils.encode_video_with_clip(clip_frames)
                    sync_features = feature_utils.encode_video_with_sync(sync_frames)
                    
                    preprocessed_data['metaclip_features'] = clip_features.detach().cpu().squeeze(0)
                    preprocessed_data['sync_features'] = sync_features.detach().cpu().squeeze(0)
                    
                    log.info(f"✅ Video features extracted: CLIP {clip_features.shape}, Sync {sync_features.shape}")
                    
                except Exception as e:
                    log.error(f"❌ Video feature extraction failed: {e}")
                    log.warning("⚠️ Falling back to text-only generation")
                    preprocessed_data['video_exist'] = torch.tensor(False)
            
        except Exception as e:
            log.error(f"❌ Feature extraction failed: {e}")
            raise

        # Enhanced sequence length calculation
        try:
            if 'metaclip_features' in preprocessed_data:
                sync_seq_len = preprocessed_data['sync_features'].shape[0]
                clip_seq_len = preprocessed_data['metaclip_features'].shape[0]
            else:
                sync_seq_len = int(_SYNC_FPS * duration)
                clip_seq_len = int(_CLIP_FPS * duration)
                
            latent_seq_len = int(194/9 * duration)
            
            log.info(f"📏 Sequence lengths: latent={latent_seq_len}, clip={clip_seq_len}, sync={sync_seq_len}")
            
            thinksound_model.model.model.update_seq_lengths(latent_seq_len, clip_seq_len, sync_seq_len)
            
        except Exception as e:
            log.error(f"❌ Sequence length update failed: {e}")
            raise

        # Enhanced conditioning with better error handling
        try:
            log.info("🔧 Preparing conditioning...")
            
            metadata = [preprocessed_data]
            
            with torch.amp.autocast(device_type=device.type):
                conditioning = thinksound_model.conditioner(metadata, device)
            
            # Enhanced empty feature handling
            video_exist = torch.stack([item['video_exist'] for item in metadata], dim=0)
            log.info(f"📹 Video exist tensor: {video_exist}")
            
            if hasattr(thinksound_model.model.model, 'empty_clip_feat') and hasattr(thinksound_model.model.model, 'empty_sync_feat'):
                if not video_exist.all():
                    log.info("🔄 Applying empty features for missing video")
                    
                    if 'metaclip_features' in conditioning:
                        conditioning['metaclip_features'][~video_exist] = thinksound_model.model.model.empty_clip_feat
                    if 'sync_features' in conditioning:
                        conditioning['sync_features'][~video_exist] = thinksound_model.model.model.empty_sync_feat
                else:
                    log.info("✅ All video features present")
            else:
                log.warning("⚠️ Model missing empty features - may affect text-only generation")
            
        except Exception as e:
            log.error(f"❌ Conditioning preparation failed: {e}")
            raise

        # Enhanced audio generation with progress tracking
        try:
            log.info(f"🎵 Generating audio: {steps} steps, CFG {cfg_scale}")
            generation_start = time.time()
            
            cond_inputs = thinksound_model.get_conditioning_inputs(conditioning)
            noise = torch.randn([1, thinksound_model.io_channels, latent_seq_len], device=device)
            
            with torch.amp.autocast(device_type=device.type):
                if thinksound_model.diffusion_objective == "v":
                    fakes = sample(thinksound_model.model, noise, steps, 0, **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True)
                elif thinksound_model.diffusion_objective == "rectified_flow":
                    fakes = sample_discrete_euler(thinksound_model.model, noise, steps, **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True)
                else:
                    raise ValueError(f"Unknown diffusion objective: {thinksound_model.diffusion_objective}")
            
            generation_time = time.time() - generation_start
            log.info(f"⏱️ Generation time: {generation_time:.2f}s")
            
        except Exception as e:
            log.error(f"❌ Audio generation failed: {e}")
            raise

        # Enhanced audio decoding and post-processing
        try:
            log.info("🔊 Decoding audio...")
            
            if thinksound_model.pretransform is not None:
                fakes = thinksound_model.pretransform.decode(fakes)
            
            # Enhanced audio normalization
            max_val = torch.max(torch.abs(fakes))
            if max_val > 0:
                audios = fakes.to(torch.float32).div(max_val).clamp(-1, 1).cpu()
            else:
                log.warning("⚠️ Generated audio is silent")
                audios = fakes.to(torch.float32).cpu()
            
            log.info(f"✅ Audio decoded: shape {audios.shape}, range [{audios.min():.3f}, {audios.max():.3f}]")
            
        except Exception as e:
            log.error(f"❌ Audio decoding failed: {e}")
            raise

        # Enhanced model offloading
        if force_offload:
            try:
                thinksound_model.to(offload_device)
                feature_utils.to(offload_device)
                mm.soft_empty_cache()
                log.info(f"💾 Models offloaded to {offload_device}")
            except Exception as e:
                log.warning(f"⚠️ Model offloading failed: {e}")

        # Enhanced audio output preparation
        audio_output = {
            "waveform": audios,
            "sample_rate": 44100
        }

        total_time = time.time() - start_time
        log.info(f"🎉 Audio generation complete! Total time: {total_time:.2f}s")
        log.info(f"📊 Performance: {duration:.1f}s audio in {total_time:.1f}s ({duration/total_time:.1f}x realtime)")
        
        return (audio_output,)

# Enhanced node mappings with better organization
NODE_CLASS_MAPPINGS = {
    "ThinkSoundModelLoader": ThinkSoundModelLoader,
    "ThinkSoundFeatureUtilsLoader": ThinkSoundFeatureUtilsLoader, 
    "ThinkSoundSampler": ThinkSoundSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThinkSoundModelLoader": "🎵 ThinkSound Model Loader",
    "ThinkSoundFeatureUtilsLoader": "🔧 ThinkSound Feature Utils Loader",
    "ThinkSoundSampler": "🎛️ ThinkSound Sampler",
}

# Enhanced version info
__version__ = "1.1.0"
__author__ = "Enhanced ThinkSound ComfyUI Integration"

log.info(f"🎵 Enhanced ThinkSound ComfyUI nodes loaded - version {__version__}")
if THINKSOUND_AVAILABLE:
    log.info("✅ All ThinkSound modules available and ready!")
else:
    log.warning("⚠️ ThinkSound modules not available - please install source code")