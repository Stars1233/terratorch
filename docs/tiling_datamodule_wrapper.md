# DataModule Tiling Wrapper

## Overview

The `TilingDataModuleWrapper` provides a generic way to add tiling capability to any PyTorch Lightning DataModule. This addresses the padding compatibility issue where models trained with one padding style produce shifted predictions when run with different padding (issues #1079, #1091).

## The Problem

**Before**: Padding was handled at the model level in [pixel_wise_model.py](../terratorch/models/pixel_wise_model.py)
```python
# Model forward pass
if self.patch_size and self.padding:
    x = pad_images(x, self.patch_size, self.padding)  # Pad here
    # ... run model ...
    mask = center_crop(mask, original_size)  # Crop here
```

**Issue**: Models trained with one padding style (e.g., "bottom-right" in v1.1.x) produce shifted predictions when loaded and run with different padding (e.g., "symmetric" in v1.2).

**Solution**: Move tiling to the DataModule level so:
1. Tiling happens during data loading, not model forward pass
2. Tiles are cached on disk (no re-computation)
3. Models trained with tiled data won't need padding at inference
4. Consistent behavior across training and inference

## Architecture

```
User Dataset
    ↓
TilingDataModuleWrapper (wraps any DataModule)
    ↓
TiledDataset (wraps dataset with on-the-fly tiling + caching)
    ↓
Cached Tiles on Disk
    ↓
DataLoader
```

## Usage

### Basic Example

```python
from terratorch.datamodules import GenericNonGeoSegmentationDataModule, TilingDataModuleWrapper

# Create your base datamodule as usual
base_dm = GenericNonGeoSegmentationDataModule(
    root_dir="./data",
    num_classes=10,
    batch_size=8,
    # ... other args
)

# Wrap it with tiling
tiled_dm = TilingDataModuleWrapper(
    base_datamodule=base_dm,
    tile_size=(512, 512),    # Each tile will be 512x512
    overlap=64,               # 64 pixel overlap between tiles
    cache_dir="./tile_cache", # Where to save cached tiles
    apply_to_splits=["train", "val"],  # Which splits to tile
)

# Use normally in training
from lightning import Trainer
trainer = Trainer(max_epochs=10)
trainer.fit(model, tiled_dm)
```

### Configuration File Example

For use with Lightning CLI:

```yaml
# config.yaml
data:
  class_path: terratorch.datamodules.TilingDataModuleWrapper
  init_args:
    base_datamodule:
      class_path: terratorch.datamodules.GenericNonGeoSegmentationDataModule
      init_args:
        root_dir: ./data
        num_classes: 10
        batch_size: 8
    tile_size: [512, 512]
    overlap: 64
    cache_dir: ./tile_cache
    apply_to_splits: [train, val]
    rebuild_cache: false
```

### Advanced: With Patch Size and Padding

If you want tiling to respect model patch size:

```python
tiled_dm = TilingDataModuleWrapper(
    base_datamodule=base_dm,
    tile_size=(512, 512),
    overlap=64,
    patch_size=16,           # Model expects inputs divisible by 16
    padding="symmetric",     # How padding was done during pre-training
    cache_dir="./tile_cache",
)
```

## Parameters

### TilingDataModuleWrapper

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_datamodule` | `LightningDataModule` | **Required** | The DataModule to wrap |
| `tile_size` | `tuple[int, int]` | `(512, 512)` | (height, width) for each tile |
| `overlap` | `int` | `64` | Pixels to overlap between adjacent tiles |
| `cache_dir` | `str` | `"./tile_cache"` | Directory for caching tiles |
| `patch_size` | `int \| None` | `None` | Model patch size (for padding compatibility) |
| `padding` | `str \| None` | `None` | Padding mode ("symmetric", "bottom-right", etc.) |
| `apply_to_splits` | `list[str] \| None` | `["train", "val"]` | Which splits to tile |
| `rebuild_cache` | `bool` | `False` | Force rebuild cache even if exists |
| `keep_incomplete_tiles` | `bool` | `False` | Keep edge tiles smaller than tile_size (uses custom collate for variable sizes) |
| `min_size` | `tuple[int, int]` | `(1, 1)` | Minimum (height, width) to process an image |
| `batch_size` | `int \| None` | `None` | Override batch size (None = use base) |
| `num_workers` | `int \| None` | `None` | Override num_workers (None = use base) |

### TiledDataset

The underlying dataset wrapper. You typically don't instantiate this directly (the DataModule wrapper does it for you), but you can if needed:

```python
from terratorch.datasets import TiledDataset

tiled_dataset = TiledDataset(
    base_dataset=my_dataset,
    tile_size=(512, 512),
    overlap=64,
    cache_dir="./tiles",
)
```

## How Caching Works

1. **First Run**: 
   - Wrapper checks for cached tiles
   - If not found, loads original image/mask from base dataset
   - Tiles the image/mask (on-the-fly)
   - Saves tiles to `cache_dir/{split}/tile_{idx}_{x}_{y}_{h}_{w}.pt`
   - Returns the tile

2. **Subsequent Runs**:
   - Wrapper checks cache
   - Finds existing tile
   - Loads from disk (very fast)
   - Returns the tile

3. **Cache Invalidation**:
   - Cache is tied to configuration (tile_size, overlap, patch_size, padding)
   - Changing any parameter creates a new cache key
   - Set `rebuild_cache=True` to force rebuild

## Cache Structure

```
tile_cache/
├── train/
│   ├── tile_index_<hash>.json       # Index of all tiles
│   ├── tile_0_0_0_512_512.pt        # Image 0, position (0,0), size 512x512
│   ├── tile_0_448_0_512_512.pt      # Image 0, position (448,0), 64px overlap
│   └── ...
├── val/
│   ├── tile_index_<hash>.json
│   └── ...
└── test/
    └── ...
```

Each `.pt` file contains a dict:
```python
{
    "image": torch.Tensor,           # Tiled image [C, H, W]
    "mask": torch.Tensor,            # Tiled mask [H, W] or [C, H, W]
    "tile_coords": (x, y, h, w),     # Tile position and size
    "base_idx": int,                 # Original image index
    # ... other keys from original sample
}
```

## Performance Considerations

### Memory
- **Low**: Tiles are cached to disk, not held in memory
- First epoch may use more disk I/O
- Subsequent epochs are as fast as reading small files

### Speed
- **First Epoch**: Slower (tiling + caching)
- **Later Epochs**: Faster (direct cache reads, ~5ms per tile vs ~100ms for tiling)
- **Disk Space**: Depends on tile size and overlap
  - Example: 100 images of 1024x1024, tiled to 512x512 with 64px overlap
  - ~400 tiles, ~100MB cache

## Inference and Stitching

### Handling Variable Tile Sizes

By default, `keep_incomplete_tiles=False` discards edge tiles that are smaller than `tile_size`. This ensures all tiles in a batch have uniform dimensions, which is required by PyTorch's default collation (`torch.stack`).

If you set `keep_incomplete_tiles=True`, the wrapper automatically uses a **custom collate function** that pads variable-sized tiles to the batch's maximum dimensions. This allows you to process complete images without losing edge regions.

```python
# Enable incomplete tiles for full image coverage
tiled_dm = TilingDataModuleWrapper(
    base_datamodule=base_dm,
    tile_size=(512, 512),
    overlap=64,
    keep_incomplete_tiles=True,  # Edge tiles with size < 512 will be padded
    cache_dir="./tile_cache",
)
```

### Stitching Predictions

After running inference on tiles, you can reconstruct full-image predictions using the provided `stitch_predictions` method:

```python
from terratorch.datamodules import TilingDataModuleWrapper
import torch

# Run inference on tiled dataloader
predictions = []
tile_coords_list = []

for batch in tiled_dm.predict_dataloader():
    images = batch["image"]
    coords = batch["tile_coords"]  # List of (y1, x1, y2, x2) tuples
    
    # Run model
    with torch.no_grad():
        preds = model(images)  # Shape: [batch_size, num_classes, H, W]
    
    predictions.append(preds)
    tile_coords_list.extend(coords)

# Concatenate all predictions
all_predictions = torch.cat(predictions, dim=0)  # [N_tiles, num_classes, H, W]

# Stitch back into full image
stitched = TilingDataModuleWrapper.stitch_predictions(
    tile_predictions=all_predictions,
    tile_coords=tile_coords_list,
    original_size=(1024, 1024),  # Original image dimensions
    overlap=64,
    use_blending=True,  # Smooth blending in overlap regions
)
# Result: [num_classes, 1024, 1024]
```

### Blend Masks for Smooth Stitching

When tiles overlap, predictions in the overlap regions come from multiple tiles. The `stitch_predictions` method uses cosine-weighted blending to smoothly merge these regions:

```python
# Get blend mask for visualization or custom stitching
blend_mask = TilingDataModuleWrapper.get_blend_mask(
    tile_size=512,
    overlap=64,
)
# Result: [512, 512] with values 0.0 to 1.0
# - Center region: 1.0 (full weight)
# - Overlap regions: smooth ramp from 0.0 to 1.0 using cosine
```

The blend mask:
- Has value `1.0` in non-overlapping regions
- Smoothly transitions from `0.0` to `1.0` over the overlap width using a cosine ramp
- Ensures seamless predictions without visible tile boundaries

**Why blending matters**: Without blending, stitched predictions may show visible seams or artifacts at tile boundaries, especially when overlap is used. Blending ensures smooth transitions.

### When to Use Overlap

| Overlap | Use Case |
|---------|----------|
| `0` | Fastest, no redundancy, good for training |
| `32-128` | Smooth predictions at tile boundaries during inference |
| `>128` | Useful for very large receptive fields |

**Note**: Overlap is most important at inference time. For training, you can often use `overlap=0` to maximize speed.

## Compatibility

### Works With
✅ Any `LightningDataModule`  
✅ TorchGeo NonGeoDataModule and GeoDataModule  
✅ Custom DataModules  
✅ GenericNonGeoSegmentationDataModule  
✅ GenericNonGeoPixelwiseRegressionDataModule  
✅ All geobench datamodules  

### Limitations
⚠️ Currently optimized for pixel-wise tasks (segmentation, regression)  
⚠️ Object detection needs special handling (see `od_tiled_dataset_wrapper.py`)  
⚠️ Multi-dataloader setups only wrap the first loader (warning logged)  

## Migration from Model-Level Padding

### Old Way (Model handles padding)
```python
model = PixelWiseModel(
    encoder=encoder,
    decoder=decoder,
    patch_size=16,
    padding="symmetric",
    # ...
)
```

### New Way (DataModule handles tiling)
```python
# Model: remove padding
model = PixelWiseModel(
    encoder=encoder,
    decoder=decoder,
    patch_size=None,     # Remove this
    padding=None,        # Remove this
    # ...
)

# DataModule: add tiling
tiled_dm = TilingDataModuleWrapper(
    base_datamodule=dm,
    tile_size=(512, 512),
    overlap=64,
    patch_size=16,       # Moved here
    padding="symmetric", # Moved here
)
```

## FAQ

**Q: Do I need to change my model code?**  
A: No! The tiling is transparent to the model. Just wrap your DataModule and train as usual.

**Q: Can I use this with pre-trained models?**  
A: Yes, but ensure tile_size and overlap are appropriate for your model's receptive field.

**Q: What if my images are smaller than tile_size?**  
A: Small images are passed through unchanged (configurable with `min_size`).

**Q: Can I use this for inference only?**  
A: Yes! Set `apply_to_splits=["test"]` or `["predict"]`.

**Q: How do I clear the cache?**  
A: Delete the `cache_dir` or set `rebuild_cache=True`.

**Q: Does this work with data augmentation?**  
A: Yes! Augmentation in your base DataModule's transforms will be applied to tiles.

**Q: Why does `keep_incomplete_tiles` default to `False`?**  
A: PyTorch's default `DataLoader` collation uses `torch.stack`, which requires all tensors in a batch to have identical dimensions. Edge tiles smaller than `tile_size` would cause collation errors. When `keep_incomplete_tiles=True`, the wrapper automatically uses a custom collate function that pads variable-sized tiles, but this adds overhead. For most training scenarios, discarding small edge tiles is acceptable and more efficient.

**Q: Does `overlap` eliminate incomplete edge tiles?**  
A: No. Overlap increases the number of tiles by creating overlapping windows, but edge tiles at image boundaries will still be smaller than `tile_size` if the image dimensions aren't exact multiples of `(tile_size - overlap)`. Overlap is primarily for improving model predictions at tile boundaries, not for avoiding incomplete tiles.

## Testing

Run the integration tests:
```bash
pytest tests/test_tiled_datamodule_wrapper.py -v
```

## See Also

- [pixel_wise_model.py](../terratorch/models/pixel_wise_model.py) - Where padding used to happen
- [utils.py](../terratorch/models/utils.py) - Padding/cropping utilities
- [od_tiled_dataset_wrapper.py](../terratorch/datasets/od_tiled_dataset_wrapper.py) - Object detection version
- [tiled_inference.py](../terratorch/tasks/tiled_inference.py) - Task-level inference tiling (alternative approach)
- `TilingDataModuleWrapper.stitch_predictions()` - Static method for stitching tile predictions back into full images
