import os
import matplotlib.pyplot as plt
from pathlib import Path
from pytomography.io.SPECT import dicom
from pytomography.transforms import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.projectors import SPECTSystemMatrix
from pytomography.algorithms import OSEM
import torch

import pydicom as dcm


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# nm_file = "images_testing/disco_new.dcm"
# nm_dcm_files = ["images_testing/jaszczak_disco_c1_j0_fov1.dcm"]
# ct_dir = "images_testing/ct"
nm_dcm_files = ["images_testing/jaszczak_intevo_c1_j0_fov1.dcm"]
ct_dir = "images_testing/ct_siemens"
# nm_dcm_files = ["images_testing/AMNRT1.nan.74f5.fr.nantes-ico.P1.6103513.1_00000.DCM"]
# ct_dir = "images_testing/pat-177Lu"
# nm_file = "images_testing/AMNRT1.nan.d5f4.fr.nantes-ico.P1.6103328.1_00000.DCM"
# nm_dcm_files = [
#     "images_testing/AMNRT1.nan.74f5.fr.nantes-ico.P1.6103513.1_00000.DCM",
#     "images_testing/AMNRT1.nan.d5f4.fr.nantes-ico.P1.6103328.1_00000.DCM",
# ]
# nm_dcm_files = [
#     "images_testing/177Lu-Patient-pytomo/nm/AMNRT1.nan.5d1d.fr.nantes-ico.P1.6057642.1_00000.DCM",
#     "images_testing/177Lu-Patient-pytomo/nm/AMNRT1.nan.3040.fr.nantes-ico.P1.6057640.1_00000.DCM",
#     "images_testing/177Lu-Patient-pytomo/nm/AMNRT1.nan.4090.fr.nantes-ico.P1.6057639.1_00000.DCM",
# ]
# ct_dir = "images_testing/177Lu-Patient-pytomo/ct"


list_recon = []
list_ds = []

for nm_file in nm_dcm_files:
    obj_meta, prj_meta = dicom.get_metadata(nm_file, index_peak=1)
    photopeak = dicom.get_projections(nm_file, index_peak=0)
    scatter = dicom.get_scatter_from_TEW(
        nm_file, index_peak=0, index_lower=1, index_upper=2
    )

    ct_files = [f for f in Path(ct_dir).glob("*.dcm")]
    att_transform = SPECTAttenuationTransform(filepath=ct_files)
    att_transform.configure(obj_meta, prj_meta)

    att_mumap = att_transform.attenuation_map

    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms=[att_transform],
        proj2proj_transforms=[],
        object_meta=obj_meta,
        proj_meta=prj_meta,
    )
    reconstruction_algorithm = OSEM(
        projections=photopeak, system_matrix=system_matrix, scatter=scatter
    )
    reconstructed_object = reconstruction_algorithm(n_iters=4, n_subsets=8)
    list_recon.append(reconstructed_object)


t_recons = torch.cat(list_recon, dim=0)
stiched_bed = dicom.stitch_multibed(t_recons, nm_dcm_files)


mip = stiched_bed.max(axis=2)
plt.imshow(mip.values.cpu()[0].rot90())
plt.show()


_, ax = plt.subplots(2, 2)
ax[0, 0].imshow(photopeak.cpu()[0, 0].rot90())
ax[0, 1].imshow(att_mumap.cpu()[0].sum(axis=1).rot90())
ax[1, 0].imshow(att_mumap.cpu()[0][:, :, obj_meta.shape[2] // 2].T)
ax[1, 1].imshow(
    reconstructed_object.cpu()[0][:, :, reconstructed_object.shape[3] // 2].T
)
plt.tight_layout()
plt.show()

dicom.save_dcm("recon_siemens", stiched_bed, nm_dcm_files[0], "")
