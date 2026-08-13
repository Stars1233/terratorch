# Copyright contributors to the Terratorch project

"""DataModule wrapper that adds tiling capability with inference stitching.

This wrapper intercepts the datasets from a base DataModule and wraps them
with TiledDataset, which tiles images/masks on-the-fly with disk caching.

**Problem Solved:**
    Models trained with one padding style (e.g., "bottom-right" in v1.1.x) produce
    shifted predictions when loaded and run with different padding (e.g., "symmetric" 
    in v1.2). This wrapper moves tiling to the data pipeline, ensuring consistent
    behavior across training and inference.

**Key Features:**
    - Generic: Works with any PyTorch Lightning DataModule
    - Efficient: Disk caching avoids repeated tiling operations
    - Flexible: Variable tile sizes with custom collation support
    - Complete: Includes prediction stitching with smooth blending

**Quick Start - Training:**
    >>> from terratorch.datamodules import TilingDataModuleWrapper
    >>> from lightning import Trainer
    >>> 
    >>> # Wrap your existing datamodule
    >>> tiled_dm = TilingDataModuleWrapper(
    ...     base_datamodule=your_datamodule,
    ...     tile_size=(512, 512),
    ...     overlap=64,
    ...     cache_dir="./tile_cache",
    ... )
    >>> 
    >>> # Train normally
    >>> trainer = Trainer(max_epochs=10)
    >>> trainer.fit(model, tiled_dm)

**Quick Start - Inference:**
    >>> # Enable keep_incomplete_tiles for full image coverage
    >>> tiled_dm = TilingDataModuleWrapper(
    ...     base_datamodule=your_datamodule,
    ...     tile_size=(256, 256),
    ...     overlap=64,
    ...     keep_incomplete_tiles=True,
    ...     apply_to_splits=["predict"],
    ... )
    >>> 
    >>> # Run inference
    >>> predictions, coords = [], []
    >>> for batch in tiled_dm.predict_dataloader():
    ...     preds = model(batch["image"])
    ...     predictions.append(preds)
    ...     coords.extend(batch["tile_coords"])
    >>> 
    >>> # Stitch back into full image
    >>> full_pred = TilingDataModuleWrapper.stitch_predictions(
    ...     tile_predictions=torch.cat(predictions),
    ...     tile_coords=coords,
    ...     original_size=(1024, 1024),
    ...     overlap=64,
    ...     use_blending=True,
    ... )

**Configuration File (Lightning CLI):**
    .. code-block:: yaml
    
        data:
          class_path: terratorch.datamodules.TilingDataModuleWrapper
          init_args:
            base_datamodule:
              class_path: terratorch.datamodules.GenericNonGeoSegmentationDataModule
              init_args:
                root_dir: ./data
                batch_size: 8
            tile_size: [512, 512]
            overlap: 64
            cache_dir: ./tile_cache

**See Also:**
    - TiledDataset: Underlying dataset wrapper
    - docs/tiling_datamodule_wrapper.md: Comprehensive documentation
    - examples/notebooks/tiling_datamodule_tutorial.ipynb: Usage examples
    - scripts/inference_stitching_demo.py: End-to-end demo
"""

from typing import Any
import logging

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from terratorch.datasets.tiled_dataset_wrapper import TiledDataset

logger = logging.getLogger(__name__)


def create_variable_tile_collate_fn():
    """Create a collate function that handles variable-sized tiles.
    
    This is needed when keep_incomplete_tiles=True, as edge tiles may have
    different dimensions than complete tiles. The function pads all tensors
    to the maximum dimensions in the batch.
    
    Returns:
        A collate function compatible with DataLoader
    """
    def collate_fn(batch):
        """Collate function that pads variable-sized tiles."""
        if len(batch) == 0:
            return {}
        
        # Collect all keys from samples
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            values = [sample[key] for sample in batch]
            
            # Handle tensor values (images, masks)
            if isinstance(values[0], torch.Tensor):
                if key in ("image", "mask"):
                    # Find max dimensions in this batch
                    ndims = values[0].ndim
                    
                    if ndims >= 2:
                        # Get max H and W dimensions
                        if ndims == 2:  # [H, W]
                            max_h = max(v.shape[0] for v in values)
                            max_w = max(v.shape[1] for v in values)
                            
                            # Pad each tensor to max dimensions
                            padded = []
                            for v in values:
                                pad_h = max_h - v.shape[0]
                                pad_w = max_w - v.shape[1]
                                # Pad format: (left, right, top, bottom)
                                padded_v = torch.nn.functional.pad(v, (0, pad_w, 0, pad_h), value=0).contiguous()
                                padded.append(padded_v)
                            collated[key] = torch.stack(padded)
                            
                        elif ndims == 3:  # [C, H, W]
                            max_h = max(v.shape[1] for v in values)
                            max_w = max(v.shape[2] for v in values)
                            
                            padded = []
                            for v in values:
                                pad_h = max_h - v.shape[1]
                                pad_w = max_w - v.shape[2]
                                padded_v = torch.nn.functional.pad(v, (0, pad_w, 0, pad_h), value=0).contiguous()
                                padded.append(padded_v)
                            collated[key] = torch.stack(padded)
                            
                        elif ndims == 4:  # [B, C, H, W]
                            max_h = max(v.shape[2] for v in values)
                            max_w = max(v.shape[3] for v in values)
                            
                            padded = []
                            for v in values:
                                pad_h = max_h - v.shape[2]
                                pad_w = max_w - v.shape[3]
                                padded_v = torch.nn.functional.pad(v, (0, pad_w, 0, pad_h), value=0).contiguous()
                                padded.append(padded_v)
                            collated[key] = torch.cat(padded, dim=0).contiguous()  # Concat along batch dim
                        else:
                            # For other tensor dims, try default stack
                            collated[key] = torch.stack(values)
                    else:
                        # 1D or scalar tensors
                        collated[key] = torch.stack(values)
                else:
                    # Non-image/mask tensors: try default stack
                    try:
                        collated[key] = torch.stack(values)
                    except RuntimeError:
                        # If stack fails, keep as list
                        collated[key] = values
            
            # Handle non-tensor values (metadata)
            elif isinstance(values[0], (tuple, list)):
                # Keep as list (e.g., tile_coords)
                collated[key] = values
            elif isinstance(values[0], (int, float)):
                # Convert scalar values to tensor
                collated[key] = torch.tensor(values)
            else:
                # Keep other types as list (e.g., filenames)
                collated[key] = values
        
        return collated
    
    return collate_fn


class TilingDataModuleWrapper(LightningDataModule):
    """Wraps any DataModule to add tiling with caching and inference stitching.
    
    This is a generic wrapper that works with any LightningDataModule. It proxies
    all methods to the base datamodule, but intercepts dataloaders to wrap their
    datasets with TiledDataset. This solves padding compatibility issues by moving
    tiling to the data pipeline instead of model forward pass.
    
    Key Features:
        - Automatic tiling of large images into manageable tiles
        - Disk caching for fast repeated access
        - Custom collate function for variable-sized tiles (inference)
        - Prediction stitching with smooth blending
        - Compatible with any PyTorch Lightning DataModule
    
    Recommended Usage:
        
        **Training (default settings):**
        
        >>> from terratorch.datamodules import TilingDataModuleWrapper
        >>> 
        >>> # Wrap your existing datamodule
        >>> tiled_dm = TilingDataModuleWrapper(
        ...     base_datamodule=your_datamodule,
        ...     tile_size=(512, 512),
        ...     overlap=64,
        ...     cache_dir="./tile_cache",
        ...     apply_to_splits=["train", "val"],
        ... )
        >>> 
        >>> # Use with Lightning Trainer
        >>> trainer = Trainer(max_epochs=10)
        >>> trainer.fit(model, tiled_dm)
        
        **Inference with stitching:**
        
        >>> # Enable keep_incomplete_tiles for full image coverage
        >>> tiled_dm = TilingDataModuleWrapper(
        ...     base_datamodule=your_datamodule,
        ...     tile_size=(256, 256),
        ...     overlap=64,
        ...     keep_incomplete_tiles=True,  # Enables custom collate
        ...     apply_to_splits=["predict"],
        ...     cache_dir="./tile_cache_inference",
        ... )
        >>> 
        >>> # Run inference and collect predictions
        >>> predictions, coords = [], []
        >>> for batch in tiled_dm.predict_dataloader():
        ...     with torch.no_grad():
        ...         preds = model(batch["image"])
        ...     predictions.append(preds)
        ...     coords.extend(batch["tile_coords"])
        >>> 
        >>> all_preds = torch.cat(predictions, dim=0)
        >>> 
        >>> # Stitch back into full image
        >>> stitched = TilingDataModuleWrapper.stitch_predictions(
        ...     tile_predictions=all_preds,
        ...     tile_coords=coords,
        ...     original_size=(1024, 1024),
        ...     overlap=64,
        ...     use_blending=True,
        ... )
        
        **With model patch size compatibility:**
        
        >>> tiled_dm = TilingDataModuleWrapper(
        ...     base_datamodule=your_datamodule,
        ...     tile_size=(512, 512),
        ...     patch_size=16,  # Model expects inputs divisible by 16
        ...     padding="symmetric",  # Match pre-training padding
        ...     overlap=64,
        ... )
        
        **Configuration file (Lightning CLI):**
        
        .. code-block:: yaml
        
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
    
    Args:
        base_datamodule: The DataModule to wrap
        tile_size: (height, width) for each tile. Default: (512, 512)
        overlap: Pixels to overlap between tiles. Use for smoother predictions
            at tile boundaries. Default: 64
        cache_dir: Directory for caching tiles. Default: "./tile_cache"
        patch_size: Model patch size for padding compatibility. If specified,
            tiles will be padded to be divisible by this value. Default: None
        padding: Padding mode (e.g., "symmetric", "bottom-right"). Should match
            model's pre-training if applicable. Default: None
        apply_to_splits: Which splits to tile ("train", "val", "test", "predict").
            Default: ["train", "val"]
        rebuild_cache: Force rebuild cache even if it exists. Default: False
        keep_incomplete_tiles: Keep edge tiles that are smaller than tile_size.
            When True, automatically uses custom collate function to pad variable
            sizes. Recommended: False for training (efficiency), True for inference
            (full coverage). Default: False
        min_size: Minimum (height, width) to process an image. Smaller images
            pass through unchanged. Default: (1, 1)
        batch_size: Override batch size (None = use base datamodule's). Default: None
        num_workers: Override num_workers (None = use base datamodule's). Default: None
    
    Note:
        When keep_incomplete_tiles=True, a custom collate function is automatically
        used to handle variable-sized tiles by padding to the batch's maximum
        dimensions. This is necessary because PyTorch's default collation requires
        uniform tensor shapes.
    
    See Also:
        - TiledDataset: The underlying dataset wrapper
        - stitch_predictions(): Static method for reconstructing full images
        - get_blend_mask(): Static method for creating smooth blend masks
    """
    
    def __init__(
        self,
        base_datamodule: LightningDataModule,
        tile_size: tuple[int, int] = (512, 512),
        overlap: int = 64,
        cache_dir: str = "./tile_cache",
        patch_size: int | None = None,
        padding: str | None = None,
        apply_to_splits: list[str] | None = None,
        rebuild_cache: bool = False,
        keep_incomplete_tiles: bool = False,
        min_size: tuple[int, int] = (1, 1),
        batch_size: int | None = None,
        num_workers: int | None = None,
    ):
        super().__init__()
        self.base_dm = base_datamodule
        self.tile_size = tile_size
        self.overlap = overlap
        self.cache_dir = cache_dir
        self.patch_size = patch_size
        self.padding = padding
        self.apply_to_splits = apply_to_splits or ["train", "val"]
        self.rebuild_cache = rebuild_cache
        self.keep_incomplete_tiles = keep_incomplete_tiles
        self.min_size = min_size
        self._batch_size = batch_size
        self._num_workers = num_workers
        
        logger.info(
            f"[TilingDataModuleWrapper] Created with tile_size={tile_size}, "
            f"overlap={overlap}, cache_dir={cache_dir}, splits={self.apply_to_splits}"
        )
    
    def prepare_data(self):
        """Delegate to base datamodule."""
        return self.base_dm.prepare_data()
    
    def setup(self, stage: str):
        """Delegate to base datamodule."""
        return self.base_dm.setup(stage)
    
    def _wrap_dataset(self, dataset: Dataset, split_name: str) -> Dataset:
        """Wrap a dataset with TiledDataset if applicable."""
        if split_name not in self.apply_to_splits:
            logger.debug(f"[TilingDataModuleWrapper] Skipping tiling for split '{split_name}'")
            return dataset
        
        if not hasattr(dataset, '__len__') or len(dataset) == 0:
            logger.warning(f"[TilingDataModuleWrapper] Dataset for '{split_name}' is empty, skipping tiling")
            return dataset
        
        cache_subdir = f"{self.cache_dir}/{split_name}"
        logger.info(f"[TilingDataModuleWrapper] Wrapping '{split_name}' dataset with tiling")
        
        return TiledDataset(
            base_dataset=dataset,
            cache_dir=cache_subdir,
            tile_size=self.tile_size,
            overlap=self.overlap,
            patch_size=self.patch_size,
            padding=self.padding,
            min_size=self.min_size,
            rebuild=self.rebuild_cache,
            keep_incomplete_tiles=self.keep_incomplete_tiles,
        )
    
    def _create_dataloader(self, dataset: Dataset, split_name: str) -> DataLoader:
        """Create a dataloader with tiled dataset."""
        tiled_dataset = self._wrap_dataset(dataset, split_name)
        
        # Determine batch_size and num_workers
        # Try to get from base datamodule if not overridden
        batch_size = self._batch_size
        num_workers = self._num_workers
        
        if batch_size is None:
            batch_size = getattr(self.base_dm, 'batch_size', 1)
        if num_workers is None:
            num_workers = getattr(self.base_dm, 'num_workers', 0)
        
        # Get collate_fn
        collate_fn = getattr(self.base_dm, 'collate_fn', None)
        
        # Use custom collate_fn if keep_incomplete_tiles is enabled
        # This handles variable-sized tiles at image edges
        if self.keep_incomplete_tiles:
            collate_fn = create_variable_tile_collate_fn()
            logger.debug(f"[TilingDataModuleWrapper] Using custom collate_fn for variable tile sizes")
        
        return DataLoader(
            tiled_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split_name == "train"),
            collate_fn=collate_fn,
        )
    
    def train_dataloader(self) -> DataLoader:
        """Return tiled train dataloader."""
        base_loader = self.base_dm.train_dataloader()
        
        if base_loader is None:
            return None
        
        # Handle single dataloader or list/dict of dataloaders
        if isinstance(base_loader, DataLoader):
            return self._create_dataloader(base_loader.dataset, "train")
        else:
            # For multiple dataloaders, wrap each
            logger.warning(
                "[TilingDataModuleWrapper] Multiple train dataloaders detected. "
                "Wrapping first one only."
            )
            if isinstance(base_loader, list):
                return self._create_dataloader(base_loader[0].dataset, "train")
            return base_loader
    
    def val_dataloader(self) -> DataLoader:
        """Return tiled validation dataloader."""
        base_loader = self.base_dm.val_dataloader()
        
        if base_loader is None:
            return None
            
        if isinstance(base_loader, DataLoader):
            return self._create_dataloader(base_loader.dataset, "val")
        else:
            logger.warning(
                "[TilingDataModuleWrapper] Multiple val dataloaders detected. "
                "Wrapping first one only."
            )
            if isinstance(base_loader, list):
                return self._create_dataloader(base_loader[0].dataset, "val")
            return base_loader
    
    def test_dataloader(self) -> DataLoader:
        """Return tiled test dataloader."""
        base_loader = self.base_dm.test_dataloader()
        
        if base_loader is None:
            return None
            
        if isinstance(base_loader, DataLoader):
            return self._create_dataloader(base_loader.dataset, "test")
        else:
            logger.warning(
                "[TilingDataModuleWrapper] Multiple test dataloaders detected. "
                "Wrapping first one only."
            )
            if isinstance(base_loader, list):
                return self._create_dataloader(base_loader[0].dataset, "test")
            return base_loader
    
    def predict_dataloader(self) -> DataLoader:
        """Return tiled predict dataloader."""
        base_loader = self.base_dm.predict_dataloader()
        
        if base_loader is None:
            return None
            
        if isinstance(base_loader, DataLoader):
            return self._create_dataloader(base_loader.dataset, "predict")
        else:
            logger.warning(
                "[TilingDataModuleWrapper] Multiple predict dataloaders detected. "
                "Wrapping first one only."
            )
            if isinstance(base_loader, list):
                return self._create_dataloader(base_loader[0].dataset, "predict")
            return base_loader
    
    @staticmethod
    def get_blend_mask(tile_size: int, overlap: int) -> torch.Tensor:
        """Create a 2D blend mask for smooth tile stitching with overlaps.
        
        Uses cosine ramps in overlap regions to smoothly blend between adjacent tiles.
        The mask has value 1.0 in the center and smoothly transitions to 0.0 at edges
        over the overlap width. This prevents visible seams in stitched predictions.
        
        The blend mask is created by:
        1. Generating 1D cosine ramp windows for overlap regions
        2. Combining via outer product to create 2D mask
        3. Center regions have weight 1.0, edges have smooth 0.0-1.0 transition
        
        Example:
            >>> # Create blend mask for 256x256 tiles with 64px overlap
            >>> mask = TilingDataModuleWrapper.get_blend_mask(tile_size=256, overlap=64)
            >>> mask.shape
            torch.Size([256, 256])
            >>> 
            >>> # Center has full weight
            >>> mask[128, 128]
            tensor(1.0000)
            >>> 
            >>> # Edges have smooth ramp
            >>> mask[0, 128]  # Top edge center
            tensor(0.0000)
            >>> mask[32, 128]  # Halfway through overlap
            tensor(0.5000)
            >>> 
            >>> # Use for custom stitching
            >>> weighted_tile = prediction * mask
        
        Args:
            tile_size: Size of the square tile (H and W)
            overlap: Overlap width in pixels. If 0, returns all-ones mask.
            
        Returns:
            2D tensor of shape [tile_size, tile_size] with blending weights in [0, 1]
            
        See Also:
            stitch_predictions: Uses this mask for automatic stitching
        """
        if overlap == 0:
            return torch.ones(tile_size, tile_size)
        
        # Create 1D cosine window for smooth blending
        # Window goes from 0 to 1 over the overlap region
        def cosine_window(length):
            return 0.5 * (1 - torch.cos(torch.pi * torch.arange(length) / (length - 1)))
        
        # Create horizontal blend: ramp up on left, constant in middle, ramp down on right
        h_blend = torch.ones(tile_size)
        if overlap > 0:
            ramp = cosine_window(overlap)
            h_blend[:overlap] = ramp  # Ramp up on left edge
            h_blend[-overlap:] = torch.flip(ramp, dims=[0])  # Ramp down on right edge
        
        # Create vertical blend (same structure)
        v_blend = torch.ones(tile_size)
        if overlap > 0:
            ramp = cosine_window(overlap)
            v_blend[:overlap] = ramp  # Ramp up on top edge
            v_blend[-overlap:] = torch.flip(ramp, dims=[0])  # Ramp down on bottom edge
        
        # Combine via outer product to get 2D mask
        blend_mask = torch.outer(v_blend, h_blend)
        return blend_mask
    
    @staticmethod
    def stitch_predictions(
        tile_predictions: torch.Tensor,
        tile_coords: list[tuple[int, int, int, int]],
        original_size: tuple[int, int],
        overlap: int = 0,
        use_blending: bool = True,
    ) -> torch.Tensor:
        """Stitch tile predictions back into full image with optional blending.
        
        Reconstructs full-image predictions from tiles by accumulating weighted
        predictions. When tiles overlap, uses blend masks to smoothly merge
        predictions without visible seams.
        
        Algorithm:
        1. Initialize output canvas and weight accumulator
        2. For each tile:
           - Apply blend mask (if using blending)
           - Accumulate: output += prediction * mask
           - Track weights: weights += mask
        3. Normalize: output /= weights
        
        Recommended Usage:
            >>> # Standard inference workflow
            >>> from terratorch.datamodules import TilingDataModuleWrapper
            >>> 
            >>> # 1. Setup tiled datamodule
            >>> tiled_dm = TilingDataModuleWrapper(
            ...     base_datamodule=your_dm,
            ...     tile_size=(256, 256),
            ...     overlap=64,
            ...     keep_incomplete_tiles=True,  # For full coverage
            ... )
            >>> 
            >>> # 2. Run inference and collect predictions
            >>> all_predictions = []
            >>> all_coords = []
            >>> 
            >>> for batch in tiled_dm.predict_dataloader():
            ...     with torch.no_grad():
            ...         preds = model(batch["image"])  # [B, C, H, W]
            ...     all_predictions.append(preds)
            ...     all_coords.extend(batch["tile_coords"])
            >>> 
            >>> predictions = torch.cat(all_predictions, dim=0)  # [N, C, H, W]
            >>> 
            >>> # 3. Stitch back into full image
            >>> stitched = TilingDataModuleWrapper.stitch_predictions(
            ...     tile_predictions=predictions,
            ...     tile_coords=all_coords,
            ...     original_size=(1024, 1024),
            ...     overlap=64,
            ...     use_blending=True,  # Smooth blending in overlaps
            ... )
            >>> stitched.shape
            torch.Size([num_classes, 1024, 1024])
        
        Example with multiple images:
            >>> # Process multiple images by grouping tiles
            >>> all_preds = {}  # image_idx -> predictions
            >>> all_coords = {}  # image_idx -> coordinates
            >>> 
            >>> for batch in dataloader:
            ...     preds = model(batch["image"])
            ...     indices = batch["base_idx"]  # Original image index
            ...     
            ...     for i, idx in enumerate(indices):
            ...         if idx not in all_preds:
            ...             all_preds[idx] = []
            ...             all_coords[idx] = []
            ...         all_preds[idx].append(preds[i:i+1])
            ...         all_coords[idx].append(batch["tile_coords"][i])
            >>> 
            >>> # Stitch each image separately
            >>> stitched_images = {
            ...     idx: TilingDataModuleWrapper.stitch_predictions(
            ...         torch.cat(all_preds[idx]),
            ...         all_coords[idx],
            ...         original_size=(1024, 1024),
            ...         overlap=64,
            ...     )
            ...     for idx in all_preds.keys()
            ... }
        
        Args:
            tile_predictions: Tensor of shape [N, C, H, W] containing predictions
                for N tiles. Tiles may be padded (e.g., from custom collate).
            tile_coords: List of N tuples (y_start, x_start, y_end, x_end) for
                each tile. These define the bounding box of each tile in the
                original image coordinates.
            original_size: (H, W) of the original full image before tiling
            overlap: Overlap width in pixels used during tiling. Should match
                the value used when creating tiles. Default: 0
            use_blending: If True, uses cosine-weighted blending for smooth
                transitions in overlap regions. If False, uses simple averaging.
                Recommended: True for best visual quality. Default: True
            
        Returns:
            Stitched prediction tensor of shape [C, H_orig, W_orig]. Values are
            normalized by accumulated weights to handle overlapping regions.
            
        Raises:
            ValueError: If tile_predictions is empty or if number of predictions
                doesn't match number of coordinates.
                
        Note:
            - Automatically handles variable-sized edge tiles by cropping predictions
            - Works with any tile size and overlap configuration
            - Overlap regions are blended smoothly when use_blending=True
            - No NaN or Inf values in output (weights are clamped to avoid division by zero)
            
        See Also:
            get_blend_mask: Creates the blend masks used for stitching
        """
        if len(tile_predictions) == 0:
            raise ValueError("Cannot stitch empty tile_predictions")
        
        if len(tile_predictions) != len(tile_coords):
            raise ValueError(
                f"Mismatch: {len(tile_predictions)} predictions but {len(tile_coords)} coords"
            )
        
        # Initialize output canvas and weight accumulator
        _, C, tile_h, tile_w = tile_predictions.shape
        H_orig, W_orig = original_size
        
        output = torch.zeros(C, H_orig, W_orig, device=tile_predictions.device, dtype=tile_predictions.dtype)
        weights = torch.zeros(1, H_orig, W_orig, device=tile_predictions.device, dtype=tile_predictions.dtype)
        
        # Get blend mask if using blending
        if use_blending and overlap > 0:
            blend_mask = TilingDataModuleWrapper.get_blend_mask(max(tile_h, tile_w), overlap)
            blend_mask = blend_mask.to(tile_predictions.device)
        else:
            blend_mask = None
        
        # Place each tile
        for i, (y1, x1, y2, x2) in enumerate(tile_coords):
            pred = tile_predictions[i]  # [C, H, W]
            
            # Get actual tile dimensions (may differ from tile_size for edge tiles)
            actual_h = y2 - y1
            actual_w = x2 - x1
            
            # Crop prediction to actual size if needed (handles padding from collate_fn)
            pred = pred[:, :actual_h, :actual_w]
            
            # Get appropriate blend mask
            if blend_mask is not None:
                # Crop blend mask to actual tile size
                mask = blend_mask[:actual_h, :actual_w].unsqueeze(0)  # [1, H, W]
            else:
                mask = torch.ones(1, actual_h, actual_w, device=pred.device)
            
            # Accumulate weighted prediction
            output[:, y1:y2, x1:x2] += pred * mask
            weights[:, y1:y2, x1:x2] += mask
        
        # Normalize by accumulated weights
        weights = torch.clamp(weights, min=1e-8)  # Avoid division by zero
        output = output / weights
        
        return output
    
    def teardown(self, stage: str):
        """Delegate to base datamodule."""
        return self.base_dm.teardown(stage)
    
    def __getattr__(self, name: str) -> Any:
        """Forward any other attribute access to base datamodule."""
        # This is called only if the attribute is not found in self
        return getattr(self.base_dm, name)
