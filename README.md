# ComfyUI-ThinkSound_Wrapper

A ComfyUI wrapper implementation of **ThinkSound** - an advanced AI model for generating high-quality audio from text descriptions and video content using Chain-of-Thought (CoT) reasoning.

## 🎵 Features

- **Text-to-Audio Generation**: Create audio from detailed text descriptions
- **Video-to-Audio Generation**: Generate synchronized audio that matches video content  
- **Chain-of-Thought Reasoning**: Use detailed CoT prompts for precise audio control
- **Multimodal Understanding**: Combines visual and textual information for better results
- **ComfyUI Integration**: Easy-to-use nodes that integrate seamlessly with ComfyUI workflows

## 🎬 What Makes ThinkSound Special

ThinkSound uses **multimodal AI** to understand both text and video:
- **MetaCLIP** for visual scene understanding
- **Synchformer** for temporal motion analysis  
- **T5** for detailed language understanding
- **Advanced diffusion model** for high-quality audio synthesis

## 📋 Requirements

### System Requirements
- **NVIDIA GPU** with at least 8GB VRAM (12GB+ recommended)
- **Python 3.8+**
- **ComfyUI** installed and working
- **Windows/Linux** (tested on Windows)

### Dependencies
The following Python packages will be installed automatically:
```
torch>=2.0.1
torchaudio>=2.0.2
torchvision>=0.15.0
transformers>=4.20.0
accelerate>=0.20.0
alias-free-torch==0.0.6
descript-audio-codec==1.0.0
vector-quantize-pytorch==1.9.14
einops==0.7.0
open-clip-torch>=2.20.0
huggingface_hub
safetensors
sentencepiece>=0.1.99
```

## 🚀 Installation

### Step 1: Install ComfyUI Custom Node

1. **Navigate to your ComfyUI custom nodes folder:**
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. **Clone this repository:**
   ```bash
   git clone https://github.com/ShmuelRonen/ComfyUI-ThinkSound_Wrapper.git
   cd ComfyUI-ThinkSound_Wrapper
   ```

3. **Your folder structure should look like:**
   ```
   ComfyUI-ThinkSound_Wrapper/
   ├── __init__.py
   ├── nodes.py
   ├── requirements.txt
   ├── thinksound/
   │   ├── data/
   │   ├── models/
   │   ├── inference/
   │   └── ...
   └── README.md
   ```

### Step 3: Install Dependencies

**Option A: Install all dependencies (recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Install minimal dependencies**
```bash
pip install torch torchaudio torchvision transformers accelerate
pip install alias-free-torch==0.0.6 descript-audio-codec==1.0.0 vector-quantize-pytorch==1.9.14
pip install einops open-clip-torch huggingface_hub safetensors sentencepiece
```

### Step 4: Download Models

1. **Download the models pack from Google Drive:**
   
   **🔗 [Download Models (Google Drive)](https://drive.google.com/file/d/13nqfPFRy2kQUx5WE0RjsmZaCYiH98dYz/view?usp=sharing)**

2. **Extract the downloaded file and place models in:**
   ```
   ComfyUI/models/thinksound/
   ├── thinksound_light.ckpt
   ├── vae.ckpt
   ├── synchformer_state_dict.pth
   └── (other model files)
   ```

3. **Create the thinksound models folder if it doesn't exist:**
   ```bash
   mkdir -p ComfyUI/models/thinksound
   ```

### Step 5: Restart ComfyUI

1. **Restart ComfyUI completely**
2. **Check the console for successful loading messages:**
   ```
   🎉 ThinkSound modules imported successfully!
   ✅ SUCCESS: Found FeaturesUtils in thinksound.data.v2a_utils.feature_utils_224
   ```

## 🎛️ Usage

### Available Nodes

After installation, you'll find these nodes in ComfyUI:

1. **ThinkSound Model Loader**
   - Loads the main ThinkSound diffusion model
   - Input: `thinksound_model` (select your .ckpt file)
   - Output: `thinksound_model`

2. **ThinkSound Feature Utils Loader** 
   - Loads VAE and Synchformer models
   - Inputs: `vae_model`, `synchformer_model`
   - Output: `feature_utils`

3. **ThinkSound Sampler**
   - Generates audio from text and/or video
   - Main generation node

### Basic Workflow

```
ThinkSound Model Loader ──┐
                         ├── ThinkSound Sampler ── Audio Output
ThinkSound Feature Utils ─┘
Loader
```

### Sampler Node Parameters

- **Duration**: Audio length in seconds (1.0 - 30.0)
- **Steps**: Denoising steps (30 recommended)
- **CFG Scale**: Guidance strength (5.0 recommended)  
- **Seed**: Random seed for reproducibility
- **Caption**: Short audio description
- **CoT Description**: Detailed Chain-of-Thought prompt
- **Video**: Optional video input for video-to-audio generation

## 🎵 Examples

### Text-to-Audio Examples

**Example 1: Simple Audio**
```
Caption: "Dog barking"
CoT Description: "Generate the sound of a medium-sized dog barking outdoors. The barking should be natural and energetic, with slight echo to suggest an open space. Include 3-4 distinct barks with realistic timing between them."
```

**Example 2: Complex Scene**
```
Caption: "Ocean waves at beach"
CoT Description: "Create gentle ocean waves lapping against the shore. Add subtle sounds of water receding over sand and pebbles. Include distant seagull calls and a light ocean breeze for natural ambiance."
```

**Example 3: Musical Content**
```
Caption: "Jazz piano"
CoT Description: "Generate a smooth jazz piano melody in a minor key. Include syncopated rhythms, bluesy chord progressions, and subtle improvisation. The tempo should be moderate and relaxing, perfect for a late-night cafe atmosphere."
```

### Video-to-Audio Generation

1. **Load a video** using ComfyUI's video loader nodes
2. **Connect the video** to the ThinkSound Sampler's video input
3. **Add descriptive text** to guide the audio generation
4. **Generate audio** that syncs with the video content

## ⚠️ Important Notes

### Model Precision
- **ThinkSound requires fp32 precision** for stable operation
- The nodes automatically use fp32 (no precision selection needed)
- Do not force fp16 as it may cause tensor dimension errors

### Memory Requirements
- **8GB VRAM minimum** for basic operation
- **12GB+ VRAM recommended** for longer audio generation
- **Enable "force_offload"** to save VRAM (enabled by default)

### Video Input Format
- **Supported**: MP4, AVI, MOV (any format ComfyUI can load)
- **Recommended**: 8-30 seconds duration
- **Processing**: Automatically handled by the node

## 🐛 Troubleshooting

### Common Issues

**Issue: "ThinkSound source code not installed"**
```
Solution: Ensure you've downloaded the ThinkSound repository to the 'thinksound' folder
```

**Issue: "ImportError: No module named 'alias_free_torch'"**
```
Solution: Install missing dependencies:
pip install alias-free-torch==0.0.6 descript-audio-codec==1.0.0 vector-quantize-pytorch==1.9.14
```

**Issue: "Input type (float) and bias type (struct c10::Half) should be the same"**
```
Solution: This is resolved automatically with fp32 precision. Restart ComfyUI if you see this error.
```

**Issue: "Tensors must have same number of dimensions"**
```
Solution: Update to the latest version of the nodes. This was fixed in recent updates.
```

**Issue: Models not loading**
```
Solution: 
1. Check that models are in ComfyUI/models/thinksound/
2. Verify model file names match the dropdown options
3. Check ComfyUI console for specific error messages
```

### Performance Tips

1. **Start with shorter durations** (8-10 seconds) for testing
2. **Use lower step counts** (12-16) for faster generation during testing
3. **Enable force_offload** to manage VRAM usage
4. **Close other GPU-intensive applications** while generating

## 📊 Expected Performance

### Generation Times (approximate)
- **8 seconds audio**: 30-60 seconds on RTX 3080
- **15 seconds audio**: 60-120 seconds on RTX 3080
- **Video analysis**: Additional 10-20 seconds

### Quality Settings
- **Steps 12-16**: Fast, good quality
- **Steps 24**: Recommended balance
- **Steps 32+**: High quality, slower

## 🔄 Updates

To update the project:
1. **Pull latest changes**: `git pull origin main`
2. **Update ThinkSound source**: `cd thinksound && git pull`
3. **Restart ComfyUI**

## 📄 License

This project is a wrapper implementation based on ThinkSound by FunAudioLLM. Please refer to the original [ThinkSound repository](https://github.com/FunAudioLLM/ThinkSound) for licensing information.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📞 Support

If you encounter issues:
1. **Check the troubleshooting section** above
2. **Review ComfyUI console output** for error messages
3. **Open an issue** on GitHub with detailed error information

## 🎉 Acknowledgments

- **ThinkSound Team** for the original model and research
- **ComfyUI Community** for the excellent framework
- **Contributors** who helped test and improve this wrapper implementation

---

**Enjoy creating amazing audio with ThinkSound!** 🎵✨
