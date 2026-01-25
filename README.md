# SDXL LoRA Trainer

Train LoRA models for SDXL to capture your likeness for use in image generation pipelines.

## Quick Start

1. **Prepare your training images** in Google Drive with matching caption files:
   ```
   /MyDrive/lora_training/images/
     photo001.jpg
     photo001.txt  # Caption: "ohwx person, professional headshot, neutral expression"
     photo002.png
     photo002.txt  # Caption: "ohwx person, casual photo outdoors, smiling"
     ...
   ```

2. **Open the notebook** in Google Colab:
   - Upload `sdxl-lora-trainer.ipynb` to Colab, or
   - Open directly from GitHub once pushed

3. **Run cells in order**:
   - Cell 1: Install dependencies
   - Cell 2: Mount Google Drive
   - Cell 3: Configure paths and parameters
   - Cell 4: Validate and prepare dataset
   - Cell 5: Train the LoRA
   - Cell 6: Test your results

## Training Data Guidelines

### Recommended Image Count
- **Minimum**: 10 images
- **Recommended**: 15-30 images
- **Maximum**: 50 images (diminishing returns beyond this)

### Image Quality
- High resolution (minimum 512px, ideally 1024px+)
- Clear, sharp focus on the subject
- Good lighting (avoid harsh shadows)
- No heavy filters or editing

### Variety
Include variety across:
- **Angles**: Front, 3/4 profile, side profile
- **Lighting**: Natural, studio, indoor, outdoor
- **Expressions**: Neutral, smiling, serious
- **Backgrounds**: Plain, environmental, varied
- **Clothing**: Different outfits and styles

### What to Avoid
- Blurry or low-resolution images
- Heavily filtered/edited photos
- Group photos (unless cropped)
- Sunglasses covering eyes (unless intentional)
- Very similar/duplicate poses

## Captioning

### Format
Each image needs a matching `.txt` file with the same base name:
```
image.jpg  ->  image.txt
photo_001.png  ->  photo_001.txt
```

### Caption Structure
```
[trigger_word] person, [subject description], [clothing], [setting], [lighting]
```

### Examples
```
ohwx person, professional headshot, wearing navy suit, studio background, soft lighting
ohwx person, casual portrait, gray t-shirt, outdoor park setting, natural daylight
ohwx person, close-up face, neutral expression, white background, ring light
ohwx person, full body shot, standing, urban street background, evening light
```

### Tips
- **Always start with trigger word** (e.g., "ohwx person")
- Be descriptive but concise
- Include relevant visual details
- Maintain consistent terminology across captions
- The trainer will shuffle caption tokens, so order within phrases doesn't matter

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NETWORK_DIM` | 32 | LoRA rank. Higher = more detail, more VRAM. 32-64 for likeness. |
| `NETWORK_ALPHA` | 16 | Scaling factor. Usually dim/2 or dim. |
| `MAX_TRAIN_EPOCHS` | 10 | Training iterations. 10-15 for likeness. |
| `REPEATS` | 10 | How many times each image is seen per epoch. |
| `LEARNING_RATE` | 1e-4 | Base learning rate. Use 1.0 with Prodigy optimizer. |
| `TRIGGER_WORD` | "ohwx" | Unique token to activate the LoRA. |

### Optimizer Choice

- **Prodigy** (recommended): Automatically adjusts learning rate. Set LR to 1.0.
- **AdamW8bit**: Classic choice, memory efficient. Use LR 1e-4 to 5e-5.
- **Lion**: Fast convergence, use lower LR (1e-5).

### Memory Optimization (T4)

For Colab Free Tier (T4 with ~15GB VRAM):
- `BATCH_SIZE`: 1
- `GRADIENT_ACCUMULATION`: 4-8
- `NETWORK_DIM`: 32 or lower
- `RESOLUTION`: 1024 (can try 768 if OOM)
- `CACHE_LATENTS`: True
- `GRADIENT_CHECKPOINTING`: True

## Using Your Trained LoRA

### In the Image Generator Notebook

Add LoRA loading to your existing pipeline:

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(...)
pipe.load_lora_weights("/path/to/my_likeness.safetensors")

# Adjust LoRA strength
pipe.fuse_lora(lora_scale=0.8)

# Generate with trigger word
image = pipe("ohwx person, portrait photo, professional lighting").images[0]
```

### Prompt Tips

- **Always include trigger word**: "ohwx person, ..."
- **LoRA scale 0.7-0.8**: Good balance of likeness and flexibility
- **LoRA scale 1.0+**: Stronger likeness, may reduce style adaptation
- **LoRA scale 0.5-**: Subtle influence, more stylistic freedom

### Combining with Styles

```
"ohwx person, anime style portrait, vibrant colors"
"ohwx person, oil painting, renaissance style"
"ohwx person, cyberpunk character, neon lighting"
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `NETWORK_DIM` to 16 or 8
- Reduce `RESOLUTION` to 768
- Set `BATCH_SIZE` to 1
- Enable all memory optimizations

### Training Loss Not Decreasing
- Check caption quality and consistency
- Try lower learning rate
- Increase number of epochs
- Verify images are high quality

### LoRA Has No Effect
- Ensure trigger word is in prompt
- Increase `lora_scale` when loading
- Check that LoRA file isn't corrupted
- Verify you're loading the correct checkpoint

### Artifacts in Generated Images
- Reduce LoRA scale (try 0.5-0.7)
- Train for fewer epochs (might be overfit)
- Use lower `NETWORK_DIM`
- Improve caption quality

## File Structure

```
image-trainer/
  sdxl-lora-trainer.ipynb   # Main training notebook
  README.md                  # This file
  sample-captions.txt        # Example caption formats
```

## References

- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) - Training framework
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers) - Inference pipeline
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) - Original paper
