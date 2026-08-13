#!/usr/bin/env python3
"""Quick test to verify GenericNonGeoObjectDetectionDataset can be instantiated."""

import sys
import tempfile
from pathlib import Path
import json
import numpy as np
import rasterio
import torch
import albumentations as A

# Add terratorch to path
sys.path.insert(0, '/dccstor/terratorch/users/aemam/xview-project/terratorch')

from terratorch.datasets.generic_nongeo_od_dataset import GenericNonGeoObjectDetectionDataset
from terratorch.datamodules.generic_nongeo_od_data_module import GenericNonGeoObjectDetectionDataModule


def create_dummy_data(tmpdir):
    """Create dummy images and labels for testing."""
    # Create dummy image
    img_dir = Path(tmpdir) / "images"
    img_dir.mkdir(exist_ok=True)
    
    img_array = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
    img_path = img_dir / "test_0_0.tif"
    
    with rasterio.open(
        img_path,
        'w',
        driver='GTiff',
        height=224,
        width=224,
        count=3,
        dtype=img_array.dtype
    ) as dst:
        for band_idx in range(3):
            dst.write(img_array[band_idx], band_idx + 1)
    
    # Create dummy labels
    label_dir = Path(tmpdir) / "labels"
    label_dir.mkdir(exist_ok=True)
    
    label_data = {
        "boxes": [[10, 10, 100, 100], [150, 150, 200, 200]],
        "labels": [1, 2]
    }
    label_path = label_dir / "test_0_0.json"
    with open(label_path, 'w') as f:
        json.dump(label_data, f)
    
    return img_dir, label_dir


def test_dataset():
    """Test that the dataset can be instantiated and loads data."""
    print("Creating temporary test data...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, label_dir = create_dummy_data(tmpdir)
        
        print(f"Test data created in {tmpdir}")
        print(f"Images: {list(img_dir.glob('*.tif'))}")
        print(f"Labels: {list(label_dir.glob('*.json'))}")
        
        # Create transform
        transform = A.Compose(
            [A.pytorch.transforms.ToTensorV2()],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )
        
        # Test dataset instantiation
        print("\nInstantiating dataset...")
        dataset = GenericNonGeoObjectDetectionDataset(
            data_root=img_dir,
            label_data_root=label_dir,
            num_classes=60,
            image_grep="*.tif",
            label_grep="*.json",
            transform=transform,
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        
        # Test loading a sample
        print("\nLoading first sample...")
        sample = dataset[0]
        
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Boxes shape: {sample['boxes'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        print(f"Filename: {sample['filename']}")
        
        assert sample['image'].shape == (3, 224, 224), f"Expected image shape (3, 224, 224), got {sample['image'].shape}"
        assert sample['boxes'].shape[0] == 2, f"Expected 2 boxes, got {sample['boxes'].shape[0]}"
        assert sample['labels'].shape[0] == 2, f"Expected 2 labels, got {sample['labels'].shape[0]}"
        
        print("\nDataset test PASSED!")
        return True


def test_datamodule():
    """Test that the datamodule can be instantiated."""
    print("\n" + "="*50)
    print("Testing DataModule...")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create train and val data
        for split in ["train", "val"]:
            split_dir = Path(tmpdir) / split
            split_dir.mkdir()
            
            img_dir = split_dir / "images"
            label_dir = split_dir / "labels"
            img_dir.mkdir()
            label_dir.mkdir()
            
            # Create one dummy sample
            img_array = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
            img_path = img_dir / f"test_0_0.tif"
            
            with rasterio.open(
                img_path,
                'w',
                driver='GTiff',
                height=224,
                width=224,
                count=3,
                dtype=img_array.dtype
            ) as dst:
                for band_idx in range(3):
                    dst.write(img_array[band_idx], band_idx + 1)
            
            # Create split file
            split_file = split_dir / f"{split}.txt"
            split_file.write_text("test_0_0\n")
            
            # Create label
            label_data = {"boxes": [[10, 10, 100, 100]], "labels": [1]}
            label_path = label_dir / "test_0_0.json"
            with open(label_path, 'w') as f:
                json.dump(label_data, f)
        
        # Create transform
        transform = A.Compose(
            [A.pytorch.transforms.ToTensorV2()],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )
        
        print("\nInstantiating datamodule...")
        dm = GenericNonGeoObjectDetectionDataModule(
            batch_size=2,
            num_workers=0,
            num_classes=60,
            train_data_root=Path(tmpdir) / "train" / "images",
            train_label_data_root=Path(tmpdir) / "train" / "labels",
            train_split=Path(tmpdir) / "train" / "train.txt",
            val_data_root=Path(tmpdir) / "val" / "images",
            val_label_data_root=Path(tmpdir) / "val" / "labels",
            val_split=Path(tmpdir) / "val" / "val.txt",
            train_transform=transform,
            val_transform=transform,
        )
        
        print("DataModule created successfully")
        
        print("\nCalling setup...")
        dm.setup("fit")
        
        print(f"Train dataset size: {len(dm.train_dataset)}")
        print(f"Val dataset size: {len(dm.val_dataset)}")
        
        print("\nGetting train dataloader...")
        train_loader = dm.train_dataloader()
        
        print("Iterating one batch...")
        for batch in train_loader:
            print(f"Batch keys: {batch.keys()}")
            print(f"Batch image shape: {batch['image'].shape}")
            print(f"Batch boxes: {batch['boxes']}")
            print(f"Batch labels: {batch['labels']}")
            print(f"Batch filenames: {batch['filename']}")
            break
        
        print("\nDataModule test PASSED!")
        return True


if __name__ == "__main__":
    try:
        test_dataset()
        test_datamodule()
        print("\n" + "="*50)
        print("ALL TESTS PASSED!")
        print("="*50)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
