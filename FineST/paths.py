## 2026.08.07 add this for run 'run_NPC_tutorial_HIPT.sh' and 'run_CRC_VisiumHD_HIPT.sh'

"""
Shared path presets for FineST tutorials and CLI entry points.

Mirrors the path block in ``Tutorial/NPC_Train_Impute_demo_*.ipynb`` so terminal
workflows can reuse the same layout without repeating long path strings.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .image_feature_extraction import default_output_paths, hist_model_tile_size


def normalize_data_root(data_root: str) -> str:
    """Return dataset root without trailing slash (e.g. ``FineST_tutorial_data``)."""
    return str(data_root).rstrip('/\\')


def infer_data_root(image_embed_path: str) -> str:
    """Infer dataset root from an image-embedding directory path."""
    normalized = str(image_embed_path).rstrip('/').replace('\\', '/')
    parts = [p for p in normalized.split('/') if p]
    for marker in ('FineST_tutorial_data_VisiumHD', 'FineST_tutorial_data', 'CRC16um'):
        if marker in parts:
            return '/'.join(parts[: parts.index(marker) + 1])
    if 'ImgEmbeddings' in parts:
        return '/'.join(parts[: parts.index('ImgEmbeddings')]) or '.'
    if 'HIPT' in parts:
        return os.path.dirname(os.path.dirname(normalized)) or '.'
    return os.path.dirname(os.path.dirname(normalized)) or '.'


def _default_patch_sizes(hist_model: str) -> tuple[int, int]:
    """Return (within/between patch_size, single-cell patch_size) for a hist model."""
    if str(hist_model).strip() == 'Virchow2':
        return 112, hist_model_tile_size(hist_model)
    return 64, hist_model_tile_size(hist_model)


def tutorial_path_presets(
    data_root: str = 'FineST_tutorial_data',
    hist_model: str = 'HIPT',
    patch_size: Optional[int] = None,
    sc_patch_size: Optional[int] = None,
    nuclei_save_folder: str = 'NPC_allspot_p075',
) -> Dict[str, str]:
    """Build NPC tutorial path presets (relative to ``system_path`` when used in CLI).

    Parameters
    ----------
    data_root : str
        Dataset root folder, e.g. ``FineST_tutorial_data``.
    hist_model : str
        ``HIPT`` or ``Virchow2``.
    patch_size : int, optional
        Within / between HE patch size (default 64 for HIPT, 112 for Virchow2).
    sc_patch_size : int, optional
        Single-cell / nuclei patch size (default 16 for HIPT, 14 for Virchow2).
    nuclei_save_folder : str
        Subfolder under ``NucleiSegments/``.

    Returns
    -------
    dict
        Keys include ``visium_path``, ``embed_dir_within``, ``save_adata_*``, etc.
    """
    root = normalize_data_root(data_root)
    hist_model = str(hist_model).strip()
    tile = hist_model_tile_size(hist_model)
    if patch_size is None or sc_patch_size is None:
        default_patch, default_sc = _default_patch_sizes(hist_model)
        patch_size = patch_size if patch_size is not None else default_patch
        sc_patch_size = sc_patch_size if sc_patch_size is not None else default_sc

    _, embed_within = default_output_paths(root, hist_model, patch_size)
    _, embed_between = default_output_paths(
        root, hist_model, patch_size, output_name=f'NEW_pth_{patch_size}_{tile}'
    )
    _, embed_sc = default_output_paths(
        root, hist_model, sc_patch_size, output_name=f'sc_pth_{sc_patch_size}_{tile}'
    )

    order_dir = f'{root}/OrderData'
    save_dir = f'{root}/SaveData'
    fig_dir = f'{root}/Figures'
    nuclei_dir = f'{root}/NucleiSegments'

    return {
        'data_root': root,
        'position_path': f'{root}/spatial/tissue_positions_list.csv',
        'position_path_add': f'{root}/spatial/tissue_positions_list_add.csv',
        'rawimage_path': f'{root}/20210809-C-AH4199551.tif',
        'STfactor_path': f'{root}/spatial/scalefactors_json.json',
        'embed_dir': f'{root}/ImgEmbeddings/{hist_model}/',
        'embed_dir_within': embed_within,
        'embed_dir_between': embed_between,
        'embed_dir_sc': embed_sc,
        'order_dir': order_dir,
        'fig_dir': fig_dir,
        'save_dir': save_dir,
        'position_order_all_path': f'{order_dir}/position_order_all.csv',
        'position_order_sc_path': f'{order_dir}/position_order_sc.csv',
        'spatial_pos_path': f'{order_dir}/position_order.csv',
        'reduced_mtx_path': f'{order_dir}/matrix_order.npy',
        'save_adata_count': f'{save_dir}/adata_count.h5ad',
        'save_adata_norml': f'{save_dir}/adata_norml.h5ad',
        'save_adata_infer': f'{save_dir}/adata_infer.h5ad',
        'save_adata_infer_spot': f'{save_dir}/adata_infer_spot.h5ad',
        'save_adata_imput': f'{save_dir}/adata_imput.h5ad',
        'save_adata_imput_spot': f'{save_dir}/adata_imput_spot.h5ad',
        'save_adata_imput_all_subspot': f'{save_dir}/adata_imput_all_subspot.h5ad',
        'save_adata_imput_all_spot': f'{save_dir}/adata_imput_all_spot.h5ad',
        'save_adata_imput_sc': f'{save_dir}/adata_imput_all_sc.h5ad',
        'nuclei_dir': nuclei_dir,
        'nuclei_coord_path': f'{nuclei_dir}/{nuclei_save_folder}/coordinates.csv',
        'nuclei_coord_csv': f'{nuclei_dir}/{nuclei_save_folder}/position_all_tissue_sc.csv',
        # CLI aliases (relative to system_path)
        'visium_path': f'{root}/spatial/tissue_positions_list.csv',
        'image_embed_path': embed_within,
        'imag_within_path': embed_within,
        'imag_betwen_path': embed_between,
        'image_embed_path_sc': embed_sc,
        'figure_save_path': f'{fig_dir}/',
        'save_data_path': f'{save_dir}/',
        'adata_all_supr_path': f'{save_dir}/adata_imput_all_subspot.h5ad',
        'adata_all_spot_path': f'{save_dir}/adata_imput_all_spot.h5ad',
        'adata_super_path_sc': f'{save_dir}/adata_imput_all_sc.h5ad',
    }


def visiumhd_path_presets(
    data_root: str = 'FineST_tutorial_data_VisiumHD',
    hist_model: str = 'HIPT',
    patch_size: Optional[int] = None,
) -> Dict[str, str]:
    """Build Visium HD CRC tutorial path presets (``CRC16_Train_Impute_count_HIPT.ipynb``).

    Replaces ``FineST_local/Dataset/CRC16um/`` and ``ContrastCRC16geneLR/`` with a single
    ``FineST_tutorial_data_VisiumHD/`` layout.
    """
    root = normalize_data_root(data_root)
    hist_model = str(hist_model).strip()
    tile = hist_model_tile_size(hist_model)
    if patch_size is None:
        patch_size = 32 if hist_model == 'HIPT' else 28

    square_dir = f'{root}/square_016um'
    embed_dir = f'{root}/HIPT/' if hist_model == 'HIPT' else f'{root}/ImgEmbeddings/{hist_model}/'
    embed_within = f'{embed_dir}HD_CRC_16um_pth_{patch_size}_{tile}/'
    embed_within_image = f'{embed_within.rstrip("/")}_image/'

    order_dir = f'{root}/OrderData'
    save_dir = f'{root}/SaveData'
    fig_dir = f'{root}/Figures'

    return {
        'data_root': root,
        'square_dir': square_dir,
        'position_path': f'{square_dir}/tissue_positions.parquet',
        'rawimage_path': f'{root}/Visium_HD_Human_Colon_Cancer_tissue_image.btf',
        'STfactor_path': f'{square_dir}/scalefactors_json.json',
        'embed_dir': embed_dir,
        'embed_dir_within': embed_within,
        'embed_dir_within_image': embed_within_image,
        'order_dir': order_dir,
        'fig_dir': fig_dir,
        'save_dir': save_dir,
        'spatial_pos_path': f'{order_dir}/position_order.csv',
        'reduced_mtx_path': f'{order_dir}/matrix_order.npy',
        'save_adata_count': f'{save_dir}/adata_count.h5ad',
        'save_adata_norml': f'{save_dir}/adata_norml.h5ad',
        'save_adata_infer': f'{save_dir}/adata_infer.h5ad',
        'save_adata_infer_spot': f'{save_dir}/adata_infer_spot.h5ad',
        'save_adata_imput': f'{save_dir}/adata_imput.h5ad',
        'save_adata_imput_spot': f'{save_dir}/adata_imput_spot.h5ad',
        'save_adata_imput_8um': f'{save_dir}/adata_imput_8um.h5ad',
        'save_adata_impt_align_8um': f'{save_dir}/adata_impt_align_8um.h5ad',
        # CLI aliases (relative to system_path)
        'visium_path': f'{square_dir}/tissue_positions.parquet',
        'image_embed_path': embed_within,
        'figure_save_path': f'{fig_dir}/',
        'save_data_path': f'{save_dir}/',
    }


def _path_presets_for_dataset(
    data_root: str,
    hist_model: str,
    dataset_class: Optional[str] = None,
    patch_size: Optional[int] = None,
) -> Dict[str, str]:
    if dataset_class == 'VisiumHD':
        return visiumhd_path_presets(data_root, hist_model=hist_model, patch_size=patch_size)
    return tutorial_path_presets(data_root, hist_model=hist_model, patch_size=patch_size)


def _fill_missing(args: Any, mapping: Dict[str, str]) -> Any:
    """Set attributes on ``args`` only when current value is None or a placeholder."""
    placeholders = {
        None,
        '',
        'figures',
        'spatial_pos.csv',
        'reduced_mtx.npy',
        'SaveData/',
    }
    for key, value in mapping.items():
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        if current is None or current in placeholders:
            setattr(args, key, value)
    return args


def apply_data_root_step1(args: Any, data_root: str, hist_model: Optional[str] = None) -> Any:
    """Fill Step1 CLI paths from path presets when not explicitly set."""
    hist = hist_model or getattr(args, 'hist_model', 'HIPT')
    presets = _path_presets_for_dataset(
        data_root,
        hist_model=hist,
        dataset_class=getattr(args, 'dataset_class', None),
        patch_size=getattr(args, 'patch_size', None),
    )
    return _fill_missing(args, {
        'visium_path': presets['visium_path'],
        'image_embed_path': presets['image_embed_path'],
        'spatial_pos_path': presets['spatial_pos_path'],
        'reduced_mtx_path': presets['reduced_mtx_path'],
        'figure_save_path': presets['figure_save_path'],
        'save_data_path': presets['save_data_path'],
    })


def apply_data_root_step2(args: Any, data_root: str, hist_model: str = 'HIPT') -> Any:
    """Fill Step2 CLI paths from ``tutorial_path_presets`` when not explicitly set."""
    presets = tutorial_path_presets(data_root, hist_model=hist_model)
    return _fill_missing(args, {
        'visium_path': presets['visium_path'],
        'imag_within_path': presets['imag_within_path'],
        'imag_betwen_path': presets['imag_betwen_path'],
        'image_embed_path_sc': presets['image_embed_path_sc'],
        'spatial_pos_path': presets['position_order_all_path'],
        'spatial_pos_path_sc': presets['position_order_sc_path'],
        'figure_save_path': presets['figure_save_path'],
        'adata_all_supr_path': presets['adata_all_supr_path'],
        'adata_all_spot_path': presets['adata_all_spot_path'],
        'adata_super_path_sc': presets['adata_super_path_sc'],
    })


def apply_data_root_nuclei(args: Any, data_root: str, hist_model: str = 'HIPT') -> Any:
    """Fill nuclei-segmentation CLI paths from ``tutorial_path_presets`` when not set."""
    presets = tutorial_path_presets(data_root, hist_model=hist_model)
    if getattr(args, 'out_dir', None) in (None, ''):
        args.out_dir = presets['nuclei_dir']
    if getattr(args, 'adata_path', None) in (None, ''):
        args.adata_path = presets['save_adata_imput_all_spot']
    if getattr(args, 'image_path', None) in (None, ''):
        args.image_path = presets['rawimage_path']
    return args
