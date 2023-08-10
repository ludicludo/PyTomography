"""This module contains classes pertaining to metadata in PyTomography. Metadata classes contain required information for interpretting data; for example, metadata corresponding to an object (with object data stored in a ``torch.Tensor``) contains the voxel spacing and voxel dimensions."""
from .metadata import ObjectMeta, ImageMeta
from .SPECT import SPECTObjectMeta, SPECTImageMeta, SPECTPSFMeta