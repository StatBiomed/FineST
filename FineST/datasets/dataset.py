from copy import copy
from ._util_dataset import AMetadata


_NPC = AMetadata(
    name="NPC",
    doc_header="Nasopharyngeal carcinoma (NPC) dataset from `Gong et al <https://doi.org/10.1038/s41467-023-37614-6>`__.",
    shape=(1331, 36601),
    url="https://figshare.com/ndownloader/files/48619396", 
)

_BRCA_FineST_allspot = AMetadata(
    name="BRCA_FineST_allspot",
    doc_header="Breast cancer (BRCA) dataset (all spots) from `Janesick et al <https://doi/10.1038/s41467-023-43458-x`__.",
    shape=(19732, 863),
    url="https://figshare.com/ndownloader/files/49070155",
)

_BRCA_Original_whole = AMetadata(
    name="BRCA_Original_whole",
    doc_header="Breast cancer (BRCA) dataset (all spots) from `Janesick et al <https://doi/10.1038/s41467-023-43458-x`__.",
    shape=(4992, 18085),
    url="https://figshare.com/ndownloader/files/49286560",
)

_BRCA_FineST_subspot = AMetadata(
    name="BRCA_FineST_subspot",
    doc_header="Breast cancer (BRCA) dataset from `Janesick et al <https://doi/10.1038/s41467-023-43458-x`__.",
    shape=(4992, 18085),
    url="https://ndownloader.figshare.com/files/40178041",
)

_CRC16um = AMetadata(
    name="CRC16um",
    doc_header="Original colorectal cancer (CRC) dataset from Oliveira, et al. <https://www.biorxiv.org/content/10.1101/2024.06.04.597233v1.full>`__.",
    shape=(137051, 18085),  
    url="https://figshare.com/ndownloader/files/49633644", 
)

_CRC08um = AMetadata(
    name="CRC08um",
    doc_header="Original colorectal cancer (CRC) dataset from Oliveira, et al. <https://www.biorxiv.org/content/10.1101/2024.06.04.597233v1.full>`__.",
    shape=(545913, 18085),  
    url="https://figshare.com/ndownloader/files/50571282", 
)

_CRC_Original = AMetadata(
    name="CRC_Original",
    doc_header="Original colorectal cancer (CRC) dataset from Oliveira, et al. <https://www.biorxiv.org/content/10.1101/2024.06.04.597233v1.full>`__.",
    shape=(136954, 862), #(137051, 18085),  
    url="https://figshare.com/ndownloader/files/48724144", 
)

_CRC_FineST = AMetadata(
    name="CRC_FineST",
    doc_header="FineST colorectal cancer (CRC) dataset  from Oliveira, et al. <https://www.biorxiv.org/content/10.1101/2024.06.04.597233v1.full>`__.",
    shape=(136954, 862),
    url="https://figshare.com/ndownloader/files/48724447",
)

_A3 = AMetadata(
    name="A3",
    doc_header="12-PCW Fetus colon single rep from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(1080, 33538),
    url="https://figshare.com/ndownloader/files/40178311",
)

_A4 = AMetadata(
    name="A4",
    doc_header="19-PCW Fetus colon single rep from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(1242, 33538),
    url="https://figshare.com/ndownloader/files/40178314",
)

_A6 = AMetadata(
    name="A6",
    doc_header="12-PCW TI rep 1 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(346, 33538),
    url="https://figshare.com/ndownloader/files/40178017",
)

_A7 = AMetadata(
    name="A7",
    doc_header="12-PCW TI rep 2 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(344, 33538),
    url="https://figshare.com/ndownloader/files/40178014",
)

_A8 = AMetadata(
    name="A8",
    doc_header="12-PCW colon rep 1 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(709, 33538),
    url="https://figshare.com/ndownloader/files/40178011",
)

_A9 = AMetadata(
    name="A9",
    doc_header="12-PCW colon rep 2 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(644, 33538),
    url="https://figshare.com/ndownloader/files/40178308",
)


for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())


__all__ = [  # noqa: F822
    "NPC", "BRCA_FineST_allspot", "BRCA_Original_whole", "BRCA_FineST_subspot", "CRC16um", "CRC08um", "CRC_Original", "CRC_FineST", "A3","A4","A6","A7","A8","A9"]
