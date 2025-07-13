from typing import Literal, Optional
import json
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from open_clip import create_model_from_pretrained
from torchvision.transforms import Normalize
from ThinkSound.models.factory import create_model_from_config
from ThinkSound.models.utils import load_ckpt_state_dict
from ThinkSound.training.utils import copy_state_dict
from transformers import AutoModel
from transformers import AutoProcessor
from transformers import T5EncoderModel, AutoTokenizer
import logging
from data_utils.ext.synchformer import Synchformer
import os

log = logging.getLogger()

def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_get_text_features(self, input_ids=None, attention_mask=None, position_ids=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = text_outputs[0]
        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features, last_hidden_state

    clip_model.get_text_features = new_get_text_features.__get__(clip_model)
    return clip_model


class FeaturesUtils(nn.Module):
 
    def __init__(
        self,
        *, 
        vae_ckpt: Optional[str] = None,
        vae_config: Optional[str] = None,
        synchformer_ckpt: Optional[str] = None,
        enable_conditions: bool = True,
        need_vae_encoder: bool = True,
    ):
        super().__init__()

        if enable_conditions:
            # Try to use local models first, fallback to online
            try:
                # Import folder_paths to get ComfyUI models directory
                import folder_paths
                models_dir = folder_paths.models_dir
                print(f"🔍 DEBUG: Using ComfyUI models directory: {models_dir}")
            except ImportError:
                # Fallback if folder_paths not available
                models_dir = None
                print("⚠️ WARNING: folder_paths not available, using online models")
            
            # MetaCLIP model paths
            if models_dir:
                metaclip_local_path = os.path.join(models_dir, "thinksound", "metaclip-h14-fullcc2.5b")
                if os.path.exists(metaclip_local_path):
                    print(f"✅ Using local MetaCLIP model: {metaclip_local_path}")
                    metaclip_path = metaclip_local_path
                else:
                    print(f"❌ Local MetaCLIP not found at {metaclip_local_path}, using online")
                    metaclip_path = "facebook/metaclip-h14-fullcc2.5b"
            else:
                metaclip_path = "facebook/metaclip-h14-fullcc2.5b"
            
            # T5 model paths
            if models_dir:
                t5_local_path = os.path.join(models_dir, "t5-v1_1-xl")
                if os.path.exists(t5_local_path):
                    print(f"✅ Using local T5 model: {t5_local_path}")
                    t5_path = t5_local_path
                else:
                    print(f"❌ Local T5 not found at {t5_local_path}, using online")
                    t5_path = "google/t5-v1_1-xl"
            else:
                t5_path = "google/t5-v1_1-xl"
            
            # Load models with local/online paths
            try:
                print(f"🔄 Loading MetaCLIP model from: {metaclip_path}")
                self.clip_model = AutoModel.from_pretrained(
                    metaclip_path,
                    local_files_only=(models_dir is not None and os.path.exists(metaclip_path))
                )
                self.clip_model = patch_clip(self.clip_model)
                
                print(f"🔄 Loading MetaCLIP processor from: {metaclip_path}")
                self.clip_processor = AutoProcessor.from_pretrained(
                    metaclip_path,
                    local_files_only=(models_dir is not None and os.path.exists(metaclip_path))
                )
                print("✅ MetaCLIP model and processor loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load MetaCLIP: {e}")
                print("🔄 Trying without local_files_only flag...")
                try:
                    self.clip_model = AutoModel.from_pretrained(metaclip_path)
                    self.clip_model = patch_clip(self.clip_model)
                    self.clip_processor = AutoProcessor.from_pretrained(metaclip_path)
                    print("✅ MetaCLIP loaded with fallback method")
                except Exception as e2:
                    print(f"❌ Failed to load MetaCLIP with fallback: {e2}")
                    raise
            
            try:
                print(f"🔄 Loading T5 tokenizer from: {t5_path}")
                self.t5_tokenizer = AutoTokenizer.from_pretrained(
                    t5_path,
                    local_files_only=(models_dir is not None and os.path.exists(t5_path))
                )
                
                print(f"🔄 Loading T5 model from: {t5_path}")
                self.t5_model = T5EncoderModel.from_pretrained(
                    t5_path,
                    local_files_only=(models_dir is not None and os.path.exists(t5_path))
                )
                print("✅ T5 model and tokenizer loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load T5: {e}")
                print("🔄 Trying without local_files_only flag...")
                try:
                    self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_path)
                    self.t5_model = T5EncoderModel.from_pretrained(t5_path)
                    print("✅ T5 loaded with fallback method")
                except Exception as e2:
                    print(f"❌ Failed to load T5 with fallback: {e2}")
                    raise
            
            # Load Synchformer
            print(f"🔄 Loading Synchformer from: {synchformer_ckpt}")
            self.synchformer = Synchformer()
            self.synchformer.load_state_dict(
                torch.load(synchformer_ckpt, weights_only=True, map_location='cpu'))
            print("✅ Synchformer loaded successfully")

            # self.tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')  # same as 'ViT-H-14'
        else:
            self.clip_model = None
            self.synchformer = None
            self.tokenizer = None

        if vae_ckpt is not None:
            print(f"🔄 Loading VAE config from: {vae_config}")
            with open(vae_config) as f:
                vae_config = json.load(f)
            self.vae = create_model_from_config(vae_config)
            print(f"🔄 Loading VAE checkpoint from: {vae_ckpt}")
            # Load checkpoint
            copy_state_dict(self.vae, load_ckpt_state_dict(vae_ckpt,prefix='autoencoder.'))#,prefix='autoencoder.'
            print("✅ VAE loaded successfully")
        else:
            self.vae = None

    def compile(self):
        if self.clip_model is not None:
            self.clip_model.encode_image = torch.compile(self.clip_model.encode_image)
            self.clip_model.encode_text = torch.compile(self.clip_model.encode_text)
        if self.synchformer is not None:
            self.synchformer = torch.compile(self.synchformer)


    def train(self, mode: bool) -> None:
        return super().train(False)

    @torch.inference_mode()
    def encode_video_with_clip(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        # x: (B, T, C, H, W) H/W: 384
        b, t, c, h, w = x.shape
        
        assert c == 3 and h == 224 and w == 224
        # x = self.clip_preprocess(x)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        outputs = []
        if batch_size < 0:
            batch_size = b * t
        for i in range(0, b * t, batch_size):
            outputs.append(self.clip_model.get_image_features(x[i:i + batch_size]))
        x = torch.cat(outputs, dim=0)
        # x = self.clip_model.encode_image(x, normalize=True)
        x = rearrange(x, '(b t) d -> b t d', b=b)
        return x

    @torch.inference_mode()
    def encode_video_with_sync(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.synchformer is not None, 'Synchformer is not loaded'
        # x: (B, T, C, H, W) H/W: 384
        b, t, c, h, w = x.shape
        # import ipdb
        # ipdb.set_trace()
        assert c == 3 and h == 224 and w == 224

        # partition the video
        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size:i * step_size + segment_size])
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

        outputs = []
        if batch_size < 0:
            batch_size = b
        x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
        for i in range(0, b * num_segments, batch_size):
            outputs.append(self.synchformer(x[i:i + batch_size]))
        x = torch.cat(outputs, dim=0)
        x = rearrange(x, '(b s) 1 t d -> b (s t) d', b=b)
        return x

    @torch.inference_mode()
    def encode_text(self, text: list[str]) -> torch.Tensor:
        assert self.clip_model is not None, 'CLIP is not loaded'
        # assert self.tokenizer is not None, 'Tokenizer is not loaded'
        # x: (B, L)
        tokens = self.clip_processor(text=text, truncation=True, max_length=77, padding="max_length",return_tensors="pt").to(self.device)
        return self.clip_model.get_text_features(**tokens)

    @torch.inference_mode()
    def encode_t5_text(self, text: list[str]) -> torch.Tensor:
        assert self.t5_model is not None, 'T5 model is not loaded'
        assert self.t5_tokenizer is not None, 'T5 Tokenizer is not loaded'
        # x: (B, L)
        inputs = self.t5_tokenizer(text,
            truncation=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt").to(self.device)
        return self.t5_model(**inputs).last_hidden_state

    @torch.inference_mode()
    def encode_audio(self, x) -> torch.Tensor:
        x = self.vae.encode(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype