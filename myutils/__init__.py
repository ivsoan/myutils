"""
-*- coding:utf-8 -*-
@Time      :2025/8/13 上午10:58
@Author    :Chen Junpeng

"""
import sys
import warnings


_import_structure = {
    'file_process': ['read_txt_to_list', 'merge_txt_files', 'get_unique_lines', 'save_data_to_json_file', 'save_array_to_npy_file',
                     'load_npy_file', 'load_json_file', 'save_list_to_txt_file', 'save_data_to_pkl_file', 'load_pkl_file',
                     'save_dict_to_npz_file', 'load_npz_file'],
    'logger': ['log_call', 'logging_init'],
    'model': ['FeedForwardNeuralNetwork', 'MyDataset', 'EarlyStop', 'RMSELoss', 'MyTrainer', 'normalize', 'denormalize',
              'save_pt_file', 'load_pt_file'],
    'molecules': ['clean_smiles_list', 'normalize_smiles', 'get_coordinates', 'generate_xyz_file', 'generate_xyz_file_worker',
                  'generate_gjf_file', 'generate_gjf_file_worker', 'calculate_fingerprint', 'draw_molecule_to_png',
                  'draw_molecule_to_png_worker', 'draw_molecule_to_svg', 'draw_molecule_to_svg_worker', 'draw_molecules_to_pdf',
                  'draw_chemical_space_plot', 'save_smiles_list_to_txt', 'calculate_molecular_properties', 'GroupCounter', 'group_count'],
    'plots': ['set_plot_config', 'draw_distribution_plot', 'draw_parity_plot', 'draw_violin_plot', 'draw_radar_plot', 'draw_line_plot', 'draw_multi_kde_plot'],
    'processes': ['run_in_processes'],
    'time_out': ['timeout'],
    'util': ['get_datetime_str'],
}

_DEPRECATED_MAP = {
    'log_execution': ('logger', 'log_call', 'v1.1.3'),
}

__all__ = []
for sub_modules in _import_structure.values():
    __all__.extend(sub_modules)
__all__.extend(_DEPRECATED_MAP.keys())


def __getattr__(name):
    for module_name, items in _import_structure.items():
        if name in items:
            module = __import__(f"myutils.{module_name}", fromlist=[name])
            return getattr(module, name)
    if name in _DEPRECATED_MAP:
        module_name, new_name, remove_version = _DEPRECATED_MAP[name]

        module = __import__(f"myutils.{module_name}", fromlist=[new_name])
        new_func = getattr(module, new_name)

        warnings.warn(
            f"{name} is deprecated and will be removed in version {remove_version}. Please use {new_name} instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        return new_func

    raise AttributeError(f"module 'myutils' has no attribute '{name}'")


def __dir__():
    return __all__