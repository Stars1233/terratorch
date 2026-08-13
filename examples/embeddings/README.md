## Embedding Workflows

This folder contains examples for generating and working with embeddings in TerraTorch.
In standard TerraTorch workflows, downstream tasks are trained end-to-end on raw image data using an encoder–decoder pipeline. Embedding workflows decouple this process: you can extract EO embeddings from any supported backbone, store and analyze them, and train lightweight decoder-only models directly on these precomputed embeddings.

**Georeferencing.**  
The embedding pipeline supports georeferenced outputs. If your inputs are georeferenced GeoTIFFs, enable `return_georeference=True` in `GenericNonGeoSegmentationDataModule`. 
When exporting to GeoTIFF or Parquet, the outputs keep this georeferencing. For other input formats (e.g., Zarr), the pipeline attempts to infer georeferencing from metadata such as latitude/longitude; if this is not possible, it writes plain (non-georeferenced) files. 
Any remaining metadata is still saved (in GeoTIFF tags or Parquet columns). This only applies to metadata returned by the dataloader. If you encounter issues, please open an issue.

<sub>
  Note: Embedding workflows in TerraTorch are supported by the
  <strong>Embed2Scale project</strong>
  (EU Horizon Europe, Grant No. 101131841; SERI; UKRI).
</sub>

### Examples
This folder includes:

- **Embedding Generation**:
    A demo notebook and sample YAML for easy embedding generation in TerraTorch: [`embedding_generation_burnscars.ipynb`](embedding_generation_burnscars.ipynb)

- **Embedding-Based Downstream Task**:
    A demo notebook and sample YAML for a segmentation task using precomputed embeddings: [`downstream_segmentation_burnscars.ipynb`](downstream_segmentation_burnscars.ipynb)  

- **Manual Embedding Extraction via Backbone Registry**:
    A short demo showing how to use the TerraTorch Backbone Registry to build a custom pipeline and extract embeddings manually: [`embedding_generation_manual_backbone_registry.ipynb`](embedding_generation_manual_backbone_registry.ipynb)  
