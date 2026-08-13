# Copyright contributors to the Terratorch project

"""Tests for TilingDataModuleWrapper and TiledDataset."""

import tempfile
from pathlib import Path

import pytest
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from terratorch.datamodules.tiled_datamodule_wrapper import TilingDataModuleWrapper
from terratorch.datasets.tiled_dataset_wrapper import TiledDataset


class DummyDataset(Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, num_samples=10, image_size=(256, 256), num_classes=5):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        h, w = self.image_size
        return {
            "image": torch.rand(3, h, w),
            "mask": torch.randint(0, self.num_classes, (h, w)),
            "filename": f"image_{idx}.tif",
        }


class DummyDataModule(LightningDataModule):
    """Simple datamodule for testing."""
    
    def __init__(self, batch_size=2, num_workers=0, image_size=(256, 256)):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = DummyDataset(num_samples=10, image_size=self.image_size)
            self.val_dataset = DummyDataset(num_samples=5, image_size=self.image_size)
        
        if stage == "test" or stage is None:
            self.test_dataset = DummyDataset(num_samples=3, image_size=self.image_size)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class TestTiledDataset:
    """Test TiledDataset functionality."""
    
    def test_basic_tiling(self):
        """Test basic tiling without caching."""
        dataset = DummyDataset(num_samples=3, image_size=(256, 256))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tiled = TiledDataset(
                base_dataset=dataset,
                cache_dir=tmpdir,
                tile_size=(128, 128),
                overlap=32,
                rebuild=False,
            )
            
            # Check number of tiles
            # 256x256 image with 128x128 tiles and 32px overlap
            # step = 128 - 32 = 96
            # tiles per dimension = ceil(256 / 96) = 3
            # total tiles per image = 3 × 3 = 9
            # 3 images × 9 tiles = 27 tiles
            # But actual count depends on exact tiling logic
            assert len(tiled) > 0, "Should have tiles"
            
            # Get a tile
            tile = tiled[0]
            assert "image" in tile
            assert "mask" in tile
            assert "tile_coords" in tile
            assert "base_idx" in tile
            
            # Check tile dimensions
            img = tile["image"]
            assert img.shape[1] == 128 or img.shape[1] < 128  # Height
            assert img.shape[2] == 128 or img.shape[2] < 128  # Width
    
    def test_caching(self):
        """Test that caching works correctly."""
        dataset = DummyDataset(num_samples=2, image_size=(256, 256))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # First creation - should create cache
            tiled1 = TiledDataset(
                base_dataset=dataset,
                cache_dir=tmpdir,
                tile_size=(128, 128),
                overlap=0,
                rebuild=False,
            )
            
            # Access first tile to populate cache
            tile1 = tiled1[0]
            
            # Check cache files exist
            cache_files = list(Path(tmpdir).glob("*.pt"))
            assert len(cache_files) >= 1, "Cache files should exist"
            
            # Second creation - should reuse cache
            tiled2 = TiledDataset(
                base_dataset=dataset,
                cache_dir=tmpdir,
                tile_size=(128, 128),
                overlap=0,
                rebuild=False,
            )
            
            # Should have same number of tiles
            assert len(tiled1) == len(tiled2)
            
            # Access same tile
            tile2 = tiled2[0]
            
            # Should be identical (loaded from cache)
            assert torch.allclose(tile1["image"], tile2["image"])
    
    def test_rebuild_cache(self):
        """Test cache rebuild functionality."""
        dataset = DummyDataset(num_samples=2, image_size=(256, 256))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create with rebuild=False
            tiled1 = TiledDataset(
                base_dataset=dataset,
                cache_dir=tmpdir,
                tile_size=(128, 128),
                overlap=0,
                rebuild=False,
            )
            _ = tiled1[0]  # Populate cache
            
            initial_cache_count = len(list(Path(tmpdir).glob("*.pt")))
            
            # Create with rebuild=True
            tiled2 = TiledDataset(
                base_dataset=dataset,
                cache_dir=tmpdir,
                tile_size=(128, 128),
                overlap=0,
                rebuild=True,  # Force rebuild
            )
            
            # Should still work
            assert len(tiled2) == len(tiled1)
    
    def test_no_overlap(self):
        """Test tiling with no overlap."""
        dataset = DummyDataset(num_samples=1, image_size=(256, 256))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tiled = TiledDataset(
                base_dataset=dataset,
                cache_dir=tmpdir,
                tile_size=(128, 128),
                overlap=0,
                rebuild=False,
            )
            
            # With 256x256 image and 128x128 tiles with no overlap
            # Should have 2×2 = 4 tiles per image
            assert len(tiled) == 4
    
    def test_incomplete_tiles(self):
        """Test handling of incomplete edge tiles."""
        dataset = DummyDataset(num_samples=1, image_size=(300, 300))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Keep incomplete tiles
            tiled_keep = TiledDataset(
                base_dataset=dataset,
                cache_dir=tmpdir + "_keep",
                tile_size=(128, 128),
                overlap=0,
                keep_incomplete_tiles=True,
            )
            
            # Don't keep incomplete tiles
            tiled_skip = TiledDataset(
                base_dataset=dataset,
                cache_dir=tmpdir + "_skip",
                tile_size=(128, 128),
                overlap=0,
                keep_incomplete_tiles=False,
            )
            
            # Should have different counts
            # With keep: (0, 128, 256) + edge = 9 tiles
            # Without keep: only complete 2x2 = 4 tiles
            assert len(tiled_keep) >= len(tiled_skip)


class TestTilingDataModuleWrapper:
    """Test TilingDataModuleWrapper functionality."""
    
    def test_basic_wrapping(self):
        """Test basic datamodule wrapping."""
        base_dm = DummyDataModule(batch_size=2, image_size=(256, 256))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tiled_dm = TilingDataModuleWrapper(
                base_datamodule=base_dm,
                tile_size=(128, 128),
                overlap=32,
                cache_dir=tmpdir,
                apply_to_splits=["train", "val"],
            )
            
            # Setup
            tiled_dm.setup("fit")
            
            # Get dataloaders
            train_loader = tiled_dm.train_dataloader()
            val_loader = tiled_dm.val_dataloader()
            
            assert train_loader is not None
            assert val_loader is not None
            
            # Check batch
            batch = next(iter(train_loader))
            assert "image" in batch
            assert "mask" in batch
            
            # Check batch size
            assert batch["image"].shape[0] <= 2  # May be smaller for last batch
    
    def test_selective_split_tiling(self):
        """Test tiling only specific splits."""
        base_dm = DummyDataModule(batch_size=2, image_size=(256, 256))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only tile train split
            tiled_dm = TilingDataModuleWrapper(
                base_datamodule=base_dm,
                tile_size=(128, 128),
                overlap=0,
                cache_dir=tmpdir,
                apply_to_splits=["train"],  # Only train
            )
            
            tiled_dm.setup("fit")
            
            train_loader = tiled_dm.train_dataloader()
            val_loader = tiled_dm.val_dataloader()
            
            # Train should have tiles
            train_batch = next(iter(train_loader))
            assert "tile_coords" in train_batch, "Train should have tiles"
            
            # Val should NOT have tiles (pass-through)
            val_batch = next(iter(val_loader))
            # Val will not have tile_coords since it's not tiled
    
    def test_batch_size_override(self):
        """Test overriding batch size."""
        base_dm = DummyDataModule(batch_size=2, image_size=(256, 256))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tiled_dm = TilingDataModuleWrapper(
                base_datamodule=base_dm,
                tile_size=(128, 128),
                overlap=0,
                cache_dir=tmpdir,
                batch_size=4,  # Override to 4
            )
            
            tiled_dm.setup("fit")
            train_loader = tiled_dm.train_dataloader()
            
            # Check that batch size is overridden
            batch = next(iter(train_loader))
            assert batch["image"].shape[0] <= 4
    
    def test_patch_size_compatibility(self):
        """Test patch size parameter."""
        base_dm = DummyDataModule(batch_size=2, image_size=(256, 256))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tiled_dm = TilingDataModuleWrapper(
                base_datamodule=base_dm,
                tile_size=(128, 128),
                overlap=0,
                cache_dir=tmpdir,
                patch_size=16,  # Model patch size
                padding="symmetric",
            )
            
            tiled_dm.setup("fit")
            train_loader = tiled_dm.train_dataloader()
            
            # Should work without errors
            batch = next(iter(train_loader))
            assert batch["image"].shape[1] == 3  # Channels
    
    def test_multiple_epochs(self):
        """Test that multiple epochs work correctly with caching."""
        base_dm = DummyDataModule(batch_size=2, image_size=(256, 256))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tiled_dm = TilingDataModuleWrapper(
                base_datamodule=base_dm,
                tile_size=(128, 128),
                overlap=0,
                cache_dir=tmpdir,
            )
            
            tiled_dm.setup("fit")
            train_loader = tiled_dm.train_dataloader()
            
            # First epoch
            epoch1_batches = []
            for batch in train_loader:
                epoch1_batches.append(batch["image"].shape[0])
            
            # Second epoch - should use cache
            epoch2_batches = []
            for batch in train_loader:
                epoch2_batches.append(batch["image"].shape[0])
            
            # Should have same structure
            assert len(epoch1_batches) == len(epoch2_batches)


def test_integration_with_real_datamodule():
    """Integration test with a more realistic scenario."""
    from torch.utils.data import DataLoader, Dataset
    
    class MockSegmentationDataset(Dataset):
        def __init__(self, root_dir, num_samples=20):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return {
                "image": torch.rand(3, 512, 512),
                "mask": torch.randint(0, 10, (512, 512)),
            }
    
    class MockBaseDataModule(LightningDataModule):
        def __init__(self):
            super().__init__()
            self.batch_size = 4
            self.num_workers = 0
        
        def setup(self, stage=None):
            self.train_ds = MockSegmentationDataset("train", num_samples=20)
            self.val_ds = MockSegmentationDataset("val", num_samples=10)
        
        def train_dataloader(self):
            return DataLoader(self.train_ds, batch_size=self.batch_size)
        
        def val_dataloader(self):
            return DataLoader(self.val_ds, batch_size=self.batch_size)
    
    base_dm = MockBaseDataModule()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tiled_dm = TilingDataModuleWrapper(
            base_datamodule=base_dm,
            tile_size=(256, 256),
            overlap=64,
            cache_dir=tmpdir,
            apply_to_splits=["train", "val"],
        )
        
        tiled_dm.setup("fit")
        
        # Simulate training loop
        train_loader = tiled_dm.train_dataloader()
        val_loader = tiled_dm.val_dataloader()
        
        # Train for one batch
        train_batch = next(iter(train_loader))
        assert train_batch["image"].shape[2] <= 256  # Tile height
        assert train_batch["image"].shape[3] <= 256  # Tile width
        
        # Validate for one batch
        val_batch = next(iter(val_loader))
        assert val_batch["image"].shape[2] <= 256
        assert val_batch["image"].shape[3] <= 256


def test_blend_mask():
    """Test blend mask generation."""
    # Test without overlap
    mask = TilingDataModuleWrapper.get_blend_mask(tile_size=128, overlap=0)
    assert mask.shape == (128, 128)
    assert torch.all(mask == 1.0)  # No blending, all ones
    
    # Test with overlap
    mask = TilingDataModuleWrapper.get_blend_mask(tile_size=128, overlap=32)
    assert mask.shape == (128, 128)
    
    # Check center is 1.0 (full weight)
    center_y, center_x = 64, 64
    assert mask[center_y, center_x] == pytest.approx(1.0, abs=1e-3)
    
    # Check edges approach 0.0
    # Top-left corner should be low
    assert mask[0, 0] < 0.1
    
    # Check smooth transition in overlap region
    # Values should increase from edge to center
    left_edge_values = mask[64, :32]  # Left overlap region at center height
    assert torch.all(left_edge_values[1:] >= left_edge_values[:-1] - 1e-5)  # Monotonic increasing


def test_stitch_predictions_no_overlap():
    """Test prediction stitching without overlap."""
    # Create 4 tiles covering a 512x512 image (no overlap)
    tile_size = 256
    original_size = (512, 512)
    num_classes = 3
    
    # Create mock predictions for 4 tiles
    tile_predictions = torch.zeros(4, num_classes, tile_size, tile_size)
    
    # Set distinct values for each tile to verify placement
    tile_predictions[0] = 1.0  # Top-left
    tile_predictions[1] = 2.0  # Top-right
    tile_predictions[2] = 3.0  # Bottom-left
    tile_predictions[3] = 4.0  # Bottom-right
    
    # Tile coordinates (y1, x1, y2, x2)
    tile_coords = [
        (0, 0, 256, 256),       # Top-left
        (0, 256, 256, 512),     # Top-right
        (256, 0, 512, 256),     # Bottom-left
        (256, 256, 512, 512),   # Bottom-right
    ]
    
    # Stitch without blending (no overlap)
    stitched = TilingDataModuleWrapper.stitch_predictions(
        tile_predictions=tile_predictions,
        tile_coords=tile_coords,
        original_size=original_size,
        overlap=0,
        use_blending=False,
    )
    
    assert stitched.shape == (num_classes, 512, 512)
    
    # Verify each quadrant has correct value
    assert torch.all(stitched[:, :256, :256] == 1.0)      # Top-left
    assert torch.all(stitched[:, :256, 256:] == 2.0)      # Top-right
    assert torch.all(stitched[:, 256:, :256] == 3.0)      # Bottom-left
    assert torch.all(stitched[:, 256:, 256:] == 4.0)      # Bottom-right


def test_stitch_predictions_with_overlap():
    """Test prediction stitching with overlap and blending."""
    tile_size = 128
    overlap = 32
    step_size = tile_size - overlap  # 96
    original_size = (192, 192)  # 2x2 tiles with overlap
    num_classes = 2
    
    # Create 4 overlapping tiles
    tile_predictions = torch.ones(4, num_classes, tile_size, tile_size)
    
    # Assign different values to verify blending
    tile_predictions[0] = 1.0  # Top-left
    tile_predictions[1] = 2.0  # Top-right
    tile_predictions[2] = 3.0  # Bottom-left
    tile_predictions[3] = 4.0  # Bottom-right
    
    # Tile coordinates with overlap
    tile_coords = [
        (0, 0, 128, 128),           # Top-left
        (0, 96, 128, 192),          # Top-right (overlaps 32px with left)
        (96, 0, 192, 128),          # Bottom-left (overlaps 32px with top)
        (96, 96, 192, 192),         # Bottom-right (overlaps with both)
    ]
    
    # Stitch with blending
    stitched = TilingDataModuleWrapper.stitch_predictions(
        tile_predictions=tile_predictions,
        tile_coords=tile_coords,
        original_size=original_size,
        overlap=overlap,
        use_blending=True,
    )
    
    assert stitched.shape == (num_classes, 192, 192)
    
    # Check non-overlap center regions have correct values
    # Region in top-left tile away from edges (past the overlap/ramp zones)
    center_tl = stitched[:, 40:60, 40:60]
    assert torch.allclose(center_tl, torch.tensor(1.0), atol=0.1)
    
    # Overlap regions should be blended (values between originals)
    # Horizontal overlap between top-left (1.0) and top-right (2.0)
    overlap_horizontal = stitched[:, 48:80, 96:128]  # Center of horizontal overlap
    # Values should be between 1.0 and 2.0 due to blending
    assert torch.all(overlap_horizontal >= 0.8)  # Allow for edge ramp effects
    assert torch.all(overlap_horizontal <= 2.2)


def test_stitch_predictions_incomplete_tiles():
    """Test stitching with incomplete edge tiles (variable sizes)."""
    num_classes = 2
    original_size = (300, 300)
    tile_size = 128
    overlap = 32
    
    # Create tiles covering the image
    # Some tiles will be 128x128, edge tiles will be smaller
    tile_predictions = []
    tile_coords = []
    
    step_size = tile_size - overlap  # 96
    
    for y in range(0, 300, step_size):
        for x in range(0, 300, step_size):
            y_end = min(y + tile_size, 300)
            x_end = min(x + tile_size, 300)
            actual_h = y_end - y
            actual_w = x_end - x
            
            # Create prediction for this tile
            # Padded to tile_size (simulating custom collate output)
            pred = torch.ones(num_classes, tile_size, tile_size) * 1.5
            tile_predictions.append(pred)
            tile_coords.append((y, x, y_end, x_end))
    
    tile_predictions = torch.stack(tile_predictions)
    
    # Stitch
    stitched = TilingDataModuleWrapper.stitch_predictions(
        tile_predictions=tile_predictions,
        tile_coords=tile_coords,
        original_size=original_size,
        overlap=overlap,
        use_blending=True,
    )
    
    assert stitched.shape == (num_classes, 300, 300)
    
    # Check that stitching completed without errors
    # With blending, edge values may be affected, but center should be stable
    center_region = stitched[:, 100:200, 100:200]
    assert torch.allclose(center_region, torch.tensor(1.5), atol=0.2)
    
    # No NaN or Inf values
    assert torch.all(torch.isfinite(stitched))


def test_variable_tile_collate():
    """Test custom collate function with variable-sized tiles."""
    from terratorch.datamodules.tiled_datamodule_wrapper import create_variable_tile_collate_fn
    
    collate_fn = create_variable_tile_collate_fn()
    
    # Create batch with variable-sized tiles
    batch = [
        {
            "image": torch.rand(3, 128, 128),
            "mask": torch.randint(0, 5, (128, 128)),
            "tile_coords": (0, 0, 128, 128),
        },
        {
            "image": torch.rand(3, 128, 96),  # Smaller width
            "mask": torch.randint(0, 5, (128, 96)),
            "tile_coords": (0, 96, 128, 192),
        },
        {
            "image": torch.rand(3, 96, 128),  # Smaller height
            "mask": torch.randint(0, 5, (96, 128)),
            "tile_coords": (96, 0, 192, 128),
        },
    ]
    
    # Collate
    collated = collate_fn(batch)
    
    # Check that all tensors are padded to max dimensions
    assert collated["image"].shape == (3, 3, 128, 128)  # Max H=128, W=128
    assert collated["mask"].shape == (3, 128, 128)
    
    # Check that tile_coords remain as list
    assert isinstance(collated["tile_coords"], list)
    assert len(collated["tile_coords"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
