from copy import copy
from ._util_dataset import AMetadata


_NPC = AMetadata(
    name="NPC",
    doc_header="Nasopharyngeal carcinoma (NPC) dataset from `Gong et al <https://doi.org/10.1038/s41467-023-37614-6>`__.",
    shape=(1331, 36601),
    url="https://figshare.com/ndownloader/files/48619396", 
)

_BRCA = AMetadata(
    name="BRCA",
    doc_header="Breast cancer (BRCA) dataset (all spots) from `Janesick et al <https://doi/10.1038/s41467-023-43458-x`__.",
    shape=(4992, 18085),
    url="https://figshare.com/ndownloader/files/49286560",
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

_HCCP1T = AMetadata(
    name="HCCP1T",
    doc_header="Hepatocellular carcinoma (HCCP1T) 10xVisium dataset (receiving anti-PD-1 treatment with non-responders) from `Liu et al <https://doi.org/10.1016/j.jhep.2023.01.011`__.",
    shape=(3354, 36601),
    url="https://figshare.com/ndownloader/files/63263551",
)

_HCCP1Tanno = AMetadata(
    name="HCCP1Tanno",
    doc_header="Hepatocellular carcinoma (HCCP1T) annotated dataset (receiving anti-PD-1 treatment with non-responders) from `Liu et al <https://doi.org/10.1016/j.jhep.2023.01.011`__.",
    shape=(3348, 36601),
    url="https://figshare.com/ndownloader/files/63263548",
)

_HCCP7T = AMetadata(
    name="HCCP7T",
    doc_header="Hepatocellular carcinoma (HCCP7T) 10xVisium dataset (receiving anti-PD-1 treatment with responders) from `Liu et al <https://doi.org/10.1016/j.jhep.2023.01.011`__.",
    shape=(4106, 36601),
    url="https://figshare.com/ndownloader/files/63263554",
)

_HCCP7Tanno = AMetadata(
    name="HCCP7Tanno",
    doc_header="Hepatocellular carcinoma (HCCP7T) annotated dataset (receiving anti-PD-1 treatment with responders) from `Liu et al <https://doi.org/10.1016/j.jhep.2023.01.011`__.",
    shape=(4106, 20693),
    url="https://figshare.com/ndownloader/files/63263557",
)

for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())


__all__ = [  # noqa: F822
    "NPC", "BRCA", "CRC16um", "CRC08um", "HCCP1T", "HCCP1Tanno", "HCCP7T", "HCCP7Tanno"]
