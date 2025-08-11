# Image Editor Tool

Interactive tool for creating target images for the LifeImage simulation fitness evaluation.

## Usage

```bash
# Basic usage (64x64 grid)
python tools/image_editor.py

# Custom grid size
python tools/image_editor.py --size 32
```

## Controls

- **Left click/drag**: Draw white pixels
- **Right click/drag**: Erase (draw black pixels)
- **Clear**: Reset the entire image to black
- **Save**: Save the current image as PNG
- **Load**: Load an existing image (automatically resized and thresholded)
- **Random**: Generate a random sparse pattern

## Output

Images are saved as PNG files and can be loaded in the simulation configuration using the `target_image` parameter.

The editor converts the binary pattern (white/black pixels) into an RGB image where white pixels represent the target pattern that organisms should form.

## Integration

To use a created image in simulation:

1. Create your target image using this tool
2. Save it to `assets/target/your_image.png`
3. Update `config/default.yaml`:
   ```yaml
   target_image: assets/target/your_image.png
   ```
