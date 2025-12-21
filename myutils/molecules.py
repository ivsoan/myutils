"""
-*- coding:utf-8 -*-
@Time      :2025/8/13 上午10:58
@Author    :Chen Junpeng

"""
import logging
import os
import sys
import warnings
from collections import Counter, OrderedDict
from typing import List, Union
import gc
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolToFile, MolDrawOptions, rdMolDraw2D
from rdkit import RDLogger
from rdkit.Chem import Crippen, Descriptors
from rdkit.Geometry import Point2D

try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    sys.path.append(os.path.join(os.environ['CONDA_PREFIX'], 'share', 'RDKit', 'Contrib'))
    from SA_Score import sascorer
from tqdm import tqdm
from .file_process import *
from .plots import *

RDLogger.DisableLog('rdApp.*')


def _check_smiles(smiles: str, sanitize: bool = True) -> Chem.Mol:
    """
    Check if the input SMILES string is valid.
    :param smiles: the input SMILES string.
    :param sanitize: sanitize the input SMILES string or not.
    :return:
    """
    if not isinstance(smiles, str) or not smiles:
        raise ValueError("Input must be a non-empty string.")

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)

        if mol is None:
            if sanitize:
                raise ValueError(
                    f"Could not parse SMILES string '{smiles}'. "
                    "Sanitization failed. Try setting sanitize=False first."
                )
            else:
                raise ValueError(
                    f"Could not parse SMILES string '{smiles}', check if input SMILES is valid."
                )

        return mol

    except Exception as e:
        raise ValueError(f"Invalid SMILES string '{smiles}'.") from e


def clean_smiles_list(smiles_list: List[str], sanitize: bool = True, num_processes: int = None, remove_error_results: bool = True, logger: logging.Logger = None) -> List[str]:
    """
    Clean the SMILES list by removing invalid SMILES strings.
    :param smiles_list: the input SMILES list.
    :param sanitize: sanitize the input SMILES string or not.
    :param num_processes: the number of processes to use.
    :param remove_error_results: remove the error results or not.
    :param logger: the logger to use.
    :return:
    """
    from .run_in_processes import run_in_processes

    parallel_decorator = run_in_processes(num_processes, remove_error_results=remove_error_results)
    parallel_normalize_smiles = parallel_decorator(normalize_smiles)
    results = parallel_normalize_smiles(smiles_list, sanitize=sanitize, logger=logger)

    return results


def normalize_smiles(smiles: str, sanitize: bool = True, **kwargs) -> str:
    """
    Normalize the input SMILES string.
    :param smiles: the input SMILES string.
    :param sanitize: sanitize the input SMILES string or not.
    :return: the normalized SMILES string.
    """
    try:
        mol = _check_smiles(smiles, sanitize=sanitize)
    except ValueError as e:
        raise e

    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return canonical_smiles


def get_coordinates(smiles: str, optimize: str = 'uff', sanitize: bool = True) -> (list, list, np.ndarray):
    """
    Get coordinates of molecules.
    :param sanitize: sanitize the input SMILES string or not.
    :param smiles: the SMILES string of the molecule.
    :param optimize: method to optimize the coordinates, 'uff' or 'mmff'.
    :return: the atomic numbers, atomic symbols, and coordinates of the molecule.
    """
    try:
        mol = _check_smiles(smiles, sanitize=sanitize)
    except ValueError as e:
        raise e

    mol = Chem.AddHs(mol)

    status = AllChem.EmbedMolecule(mol)
    if status == -1:
        raise ValueError(f"Could not generate 3D coordinates for SMILES: {smiles}")

    if optimize == 'uff':
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            raise ValueError(f"UFF optimization failed for SMILES: {smiles}")
    elif optimize == 'mmff':
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            raise ValueError(f"MMFF optimization failed for SMILES: {smiles}")
    else:
        pass

    # todo: openbabel

    atomic_numbers = []
    atomic_symbols = []

    for atom in mol.GetAtoms():
        atomic_numbers.append(atom.GetAtomicNum())
        atomic_symbols.append(atom.GetSymbol())

    try:
        conformer = mol.GetConformer()
        coordinates = conformer.GetPositions()
        coordinates = np.array(coordinates)
    except AttributeError:
        raise ValueError(f"Could not get coordinates for SMILES: {smiles}")

    assert len(coordinates) == len(atomic_numbers) == len(atomic_symbols), f"Coordinates length not match with atomic numbers and symbols, got {len(coordinates)}, {len(atomic_numbers)}and {len(atomic_symbols)}"

    return atomic_numbers, atomic_symbols, coordinates


def generate_xyz_file(smiles: str, save_path: str, name: str, optimize: str = 'uff', sanitize: bool = True, logger: logging.Logger = None) -> None:
    """
    Generate XYZ file for a given SMILES string.
    :param smiles: the input SMILES string.
    :param save_path: the path to save the XYZ file.
    :param name: the name of the XYZ file.
    :param optimize: the method to optimize the coordinates, 'uff' or 'mmff'.
    :param sanitize: sanitize the input SMILES string or not.
    :param logger: the logger to use.
    :return:
    """
    log = logger.info if logger else print

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        log(f"Created directory {save_path}")

    try:
        _, atomic_symbols, coordinates = get_coordinates(smiles, optimize, sanitize)
    except ValueError as e:
        raise e

    xyz_file = os.path.join(save_path, f"{name}.xyz")
    with open(xyz_file, 'w') as f:
        f.write(f"{len(coordinates)}\n")
        f.write(f"{smiles}\n")
        for i, (x, y, z) in enumerate(coordinates):
            f.write(f"{atomic_symbols[i]} {x:.6f} {y:.6f} {z:.6f}\n")
    log(f"Saved xyz file to {xyz_file}")


def generate_xyz_file_worker(task_info: tuple, save_path: str, optimize: str = 'uff', sanitize: bool = True, logger: logging.Logger = None):
    """
    Worker function for generating XYZ files in parallel, see usage in func `run_in_processes` for more details.
    :param task_info: the tuple (smiles, name).
    :param save_path: the path to save the XYZ file.
    :param optimize: the method to optimize the coordinates, 'uff' or 'mmff'.
    :param sanitize: sanitize the input SMILES string or not.
    :param logger: the logger to use.
    :return:
    """
    assert len(task_info) == 2, f"Task info should be a tuple of (smiles, name), got {task_info}"
    generate_xyz_file(task_info[0], save_path, task_info[1], optimize, sanitize, logger)


def generate_gjf_file(smiles: str, save_path: str, name: str, forward_lines: str = None, backward_lines: str = None, optimize: str = 'uff', sanitize: bool = True, logger: logging.Logger = None) -> None:
    """
    Generate GJF file for a given SMILES string.
    :param smiles: the input SMILES string.
    :param save_path: the path to save the GJF file.
    :param name: the name of the GJF file.
    :param forward_lines: the lines before the coordinates, such as "%nprocshared=4\n%mem=10GB\n#p opt b3lyp/6-31g(d,p)\n\ntest\n\n0 1\n"
    :param backward_lines: the lines after the coordinates"
    :param optimize: the method to optimize the coordinates, 'uff' or 'mmff'.
    :param sanitize: sanitize the input SMILES string or not.
    :param logger: the logger to use.
    :return:
    """
    from .utils_waring import UtilsWarning
    log = logger.info if logger else print

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"Created directory {save_path}")

    if forward_lines is None:
        warnings.warn(UtilsWarning("No forward lines specified. Check forward lines."))

    try:
        _, atomic_symbols, coordinates = get_coordinates(smiles, optimize, sanitize)
    except ValueError as e:
        raise e

    gjf_file = os.path.join(save_path, f"{name}.gjf")
    with open(gjf_file, 'w') as f:
        f.write(forward_lines if forward_lines else "")
        for i, (x, y, z) in enumerate(coordinates):
            f.write(f"{atomic_symbols[i]} {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")
        f.write(backward_lines if backward_lines else "\n")

    log(f"Saved gjf file to {gjf_file}")


def generate_gjf_file_worker(task_info: tuple, save_path: str, optimize: str = 'uff', sanitize: bool = True, logger: logging.Logger = None):
    """
    Worker function for generating GJF files in parallel, see usage in func `run_in_processes` for more details.
    :param task_info: the tuple (smiles, name, forward_lines, backward_lines).
    :param save_path: the path to save the GJF file.
    :param optimize: the method to optimize the coordinates, 'uff' or 'mmff'.
    :param sanitize: sanitize the input SMILES string or not.
    :param logger: the logger to use.
    :return:
    """
    assert len(task_info) == 4, f"Task info should be a tuple of (smiles, name, forward_lines, backward_lines), got {task_info}"
    generate_gjf_file(task_info[0], save_path, task_info[1], task_info[2], task_info[3], optimize, sanitize, logger)


def calculate_fingerprint(smiles: str, fp_type: str = 'ecfp', radius: int = 2, nbits: int = 2048, sanitize: bool = True, **kwargs) -> np.ndarray:
    """
    Calculate the fingerprint of a given SMILES string.
    :param smiles: the input SMILES string.
    :param fp_type: the type of fingerprint. 'ecfp' for ECFP, 'maccs' for MACCS, 'fcfp' for FCFP.
    :param radius: the radius of the fingerprint.
    :param nbits: the number of bits of the fingerprint.
    :param sanitize: sanitize the input SMILES string or not.
    :return: the fingerprint of the molecule. shape: (nbits,)
    """
    try:
        mol = _check_smiles(smiles, sanitize=sanitize)
    except Exception as e:
        raise e

    if fp_type == 'ecfp':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    elif fp_type == 'maccs':
        fp = AllChem.GetMACCSKeysFingerprint(mol)
    elif fp_type == 'fcfp':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits, useFeatures=True)
    else:
        raise ValueError(f"Invalid fingerprint type: {fp_type}")

    return np.array(fp)


def draw_molecule_to_png(smiles: str, save_path: str, name: str, msg: str = None, sanitize: bool = True, show_atom_numbers: bool = False, size: (int, int) = (400, 400), logger: logging.Logger = None) -> None:
    """
    Draw a molecule to a PNG file.
    :param smiles: the input SMILES string.
    :param save_path: the path to save the PNG file.
    :param name: the name of the PNG file.
    :param msg: the message to be drawn at the bottom of the image.
    :param sanitize: sanitize the input SMILES string or not.
    :param show_atom_numbers: show the atom numbers or not.
    :param size: the size of the molecule.
    :param logger: the logger to use.
    :return:
    """
    log = logger.info if logger else print
    try:
        mol = _check_smiles(smiles, sanitize=sanitize)
    except Exception as e:
        raise e

    try:
        AllChem.Compute2DCoords(mol)
    except Exception as e:
        warnings.warn(UtilsWarning(f"Could not generate 2D coordinates for SMILES: {smiles}, use unoptimized coordinates."))

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        log(f"Created directory {save_path}")

    png_file = os.path.join(save_path, f"{name}.png")

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    options = drawer.drawOptions()

    if show_atom_numbers:
        options.addAtomIndices = True

    options.legendColour = (0, 0, 0)
    drawer.DrawMolecule(mol, legend=str(msg) or "")

    drawer.FinishDrawing()

    try:
        with open(png_file, 'wb') as f:
            f.write(drawer.GetDrawingText())
        log(f"Saved png file to {png_file}")
    except Exception as e:
        log(f"Could not save png file to {png_file}, {e}")


def draw_molecule_to_png_worker(task_info: tuple, save_path: str, sanitize: bool = True, show_atom_numbers: bool = False, size: (int, int) = (400, 400), logger: logging.Logger = None):
    """
    Worker function for drawing a molecule to a PNG file, see usage in func `run_in_processes` for more details.
    :param task_info: the tuple (smiles, name).
    :param save_path: the path to save the PNG file.
    :param sanitize: sanitize the input SMILES string or not.
    :param show_atom_numbers: show the atom numbers or not.
    :param size: the size of the molecule.
    :param logger: the logger to use.
    :return:
    """
    assert len(task_info) == 2, f"Task info should be a tuple of (smiles, name), got {task_info}"
    draw_molecule_to_png(task_info[0], save_path, task_info[1], sanitize, show_atom_numbers, size, logger)


def draw_molecule_to_svg(smiles: str, save_path: str, name: str, sanitize: bool = True, show_atom_numbers: bool = False, add_stereo_annotation: bool = True, size: (int, int) = (400, 400), highlight_atoms: list = None, highlight_bonds: list = None, logger: logging.Logger = None) -> None:
    """
    Draw a molecule to a SVG file.
    :param smiles: the input SMILES string.
    :param save_path: the path to save the SVG file.
    :param name: the name of the SVG file.
    :param sanitize: sanitize the input SMILES string or not.
    :param show_atom_numbers: show the atom numbers or not.
    :param add_stereo_annotation: add the stereo annotation to the molecule or not.
    :param size: the size of the molecule.
    :param highlight_atoms: the list of atoms to highlight.
    :param highlight_bonds: the list of bonds to highlight.
    :param logger: the logger to use.
    :return:
    """
    from .utils_waring import UtilsWarning
    log = logger.info if logger else print
    try:
        mol = _check_smiles(smiles, sanitize=sanitize)
    except Exception as e:
        raise e

    try:
        AllChem.Compute2DCoords(mol)
    except Exception as e:
        warnings.warn(
            UtilsWarning(f"Could not generate 2D coordinates for SMILES: {smiles}, use unoptimized coordinates."))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"Created directory {save_path}")

    svg_file = os.path.join(save_path, f"{name}.svg")

    drawer = rdMolDraw2D.MolDraw2DSVG(*size)

    options = drawer.drawOptions()
    options.addAtomIndices = show_atom_numbers
    options.addStereoAnnotation = add_stereo_annotation
    options.explicitMethyl = False
    options.annotationFontScale = 0.8
    options.bondLineWidth = 2

    drawer.DrawMolecule(mol,
                        highlightAtoms=highlight_atoms,
                        highlightBonds=highlight_bonds)

    drawer.FinishDrawing()
    svg_data = drawer.GetDrawingText()

    with open(svg_file, 'w') as f:
        f.write(svg_data)

    log(f"Saved svg file to {svg_file}")


def draw_molecule_to_svg_worker(task_info: tuple, save_path: str, sanitize: bool = True, show_atom_numbers: bool = False, add_stereo_annotation: bool = True, size: (int, int) = (400, 400), logger: logging.Logger = None):
    """
    Worker function for drawing a molecule to a SVG file, see usage in func `run_in_processes` for more details.
    :param task_info: the tuple (smiles, name, highlight_atoms, highlight_bonds).
    :param save_path: the path to save the SVG file.
    :param sanitize: sanitize the input SMILES string or not.
    :param show_atom_numbers: show the atom numbers or not.
    :param add_stereo_annotation: add the stereo annotation to the molecule or not.
    :param size: the size of the molecule.
    :param logger: the logger to use.
    :return:
    """
    assert len(task_info) == 4, f"Task info should be a tuple of (smiles, name, highlight_atoms, highlight_bonds), got {task_info}"
    draw_molecule_to_svg(task_info[0], save_path, task_info[1], sanitize, show_atom_numbers, add_stereo_annotation, size, task_info[2], task_info[3], logger)


def draw_molecules_to_pdf(smiles_list: list,
                          save_path: str,
                          name: str,
                          clean_input_smiles_list: bool = False,
                          num_processes: int = None,
                          text_list: list = None,
                          sanitize: bool = True,
                          show_atom_numbers: bool = False,
                          add_stereo_annotation: bool = True,
                          size: (int, int) = (400, 400),
                          font_size: int = 12,
                          mols_per_page_rows: int = 4,
                          mols_per_page_cols: int = 5,
                          logger: logging.Logger = None) -> None:
    """
    Draw a list of molecules to a PDF file.
    :param smiles_list: the input SMILES list.
    :param save_path: the path to save the PDF file.
    :param name: the name of the PDF file.
    :param clean_input_smiles_list: clean the input SMILES list or not.
    :param num_processes: the number of processes to use.
    :param text_list: the list of text strings.
    :param sanitize: sanitize the input SMILES string or not.
    :param show_atom_numbers: show the atom numbers or not.
    :param add_stereo_annotation: add the stereo annotation to the molecule or not.
    :param size: the size of the molecule.
    :param font_size: the font size of the molecule.
    :param mols_per_page_rows: number of molecules per page row.
    :param mols_per_page_cols: number of molecules per page column.
    :param logger: the logger to use.
    :return:
    """
    import io
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from .utils_waring import UtilsWarning
    try:
        import cairosvg
    except ImportError:
        raise ImportError("Please install cairosvg to use draw_molecules_to_pdf function. bash```pip install cairosvg```")

    log = logger.info if logger else print

    A4_WIDTH_INCHES = 8.27
    A4_HEIGHT_INCHES = 11.69

    if clean_input_smiles_list:
        smiles_list = clean_smiles_list(smiles_list, sanitize=sanitize, num_processes=num_processes, logger=logger, remove_error_results=False)
        if text_list:
            text_list = [text for text, smiles in zip(text_list, smiles_list) if smiles is not None]
        smiles_list = [smiles for smiles in smiles_list if smiles is not None]
    else:
        warnings.warn(UtilsWarning("Make sure the input SMILES are all valid SMILES strings, or set `clean_input_smiles_groups` to True."))

    def _create_subplot(ax, mol_smiles, text_content):
        ax.axis('off')
        try:
            mol = _check_smiles(mol_smiles, sanitize=sanitize)
        except Exception as e:
            log(f"Could not parse SMILES string: {mol_smiles}, {e}")
            return

        d = rdMolDraw2D.MolDraw2DSVG(*size)
        options = d.drawOptions()
        options.addAtomIndices = show_atom_numbers
        options.addStereoAnnotation = add_stereo_annotation
        options.explicitMethyl = False
        options.annotationFontScale = 0.8
        options.bondLineWidth = 2
        d.DrawMolecule(mol)
        d.FinishDrawing()
        svg_text = d.GetDrawingText()

        img_extent = (0.1, 0.9, 0.4, 1.0)  # (left, right, bottom, top)

        try:
            png_bytes = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'), scale=4)
            img = plt.imread(io.BytesIO(png_bytes), format='png')
            ax.imshow(img, extent=img_extent, aspect='equal')
        except Exception as e:
            log(f"Could not generate image for SMILES: {mol_smiles}, {e}")
            return

        if text_content:
            ax.text(0.5, 0.35, text_content, fontsize=font_size,
                    ha='center', va='top', wrap=True)

    if text_list is not None:
        assert len(smiles_list) == len(text_list), f"The length of smiles_list and text_list should be the same, got {len(smiles_list)} and {len(text_list)}."
    else:
        text_list = [None] * len(smiles_list)

    num_molecules = len(smiles_list)
    mols_per_page = mols_per_page_rows * mols_per_page_cols
    num_pages = (num_molecules + mols_per_page - 1) // mols_per_page

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"Created directory {save_path}")

    pdf_path = os.path.join(save_path, f"{name}.pdf")

    with PdfPages(pdf_path) as pdf:
        for page_num in tqdm(range(num_pages), desc="generating pdf pages..."):
            fig, axes = plt.subplots(
                mols_per_page_cols, mols_per_page_rows,
                figsize=(A4_WIDTH_INCHES, A4_HEIGHT_INCHES)
            )
            axes_flat = axes.flatten()
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.97, wspace=0.1, hspace=0.1)

            for i in range(mols_per_page):
                mol_idx = page_num * mols_per_page + i
                if mol_idx < num_molecules:
                    _create_subplot(axes_flat[i], smiles_list[mol_idx], text_list[mol_idx])
                else:
                    axes_flat[i].axis('off')

            pdf.savefig(fig, dpi=1200)
            plt.close(fig)
    log(f"Saved pdf file to {pdf_path}")


def draw_chemical_space_plot(smiles_groups: List[Union[str, List[str]]],
                             save_path: str,
                             name: str,
                             group_names: list = None,
                             method: str = 'umap',
                             cuml: bool = True,
                             fp_type: str = 'ecfp',
                             radius: int = 2,
                             nbits: int = 2048,
                             sanitize: bool = True,
                             perplexity: int = 30,
                             n_neighbors: int = 100,
                             random_state: int = 42,
                             num_processes: int = None,
                             alpha_list: list = None,
                             s_list: list = None,
                             marker_list: list = None,
                             cmap: str | list = 'tab20',
                             show_legend: bool = False,
                             clean_input_smiles_groups: bool = False,
                             res_pkl_path: str = None,
                             save_res: bool = True,
                             logger: logging.Logger = None):
    """
    Draw a chemical space plot using t-SNE.
    :param n_neighbors: param for t-sne method in cuml
    :param cuml: use cuml or not
    :param method: umap or tsne
    :param smiles_groups: groups of smiles strings, each group is a list of smiles strings.
    :param save_path: the path to save the figure.
    :param name: the name of the figure.
    :param group_names: the list of group names.
    :param fp_type: the fingerprint type, can be 'ecfp', 'fcfp' or 'maccs'.
    :param radius: the radius of the fingerprint.
    :param nbits: the number of bits of the fingerprint.
    :param sanitize: sanitize the smiles strings or not.
    :param perplexity: the perplexity of the TSNE algorithm.
    :param random_state: the random state of the TSNE algorithm.
    :param num_processes: the number of processes to use.
    :param alpha_list: a list of alpha values for each group.
    :param s_list: a list of size values for each group.
    :param marker_list: a list of marker values for each group.
    :param cmap: the color map to use.
    :param show_legend: show the legend on the figure or not.
    :param clean_input_smiles_groups: clean the smiles groups before drawing the plot or not.
    :param res_pkl_path: the path to save the results as a pickle file.
    :param save_res: save the results or not.
    :param logger: the logger to use.
    :return:
    """
    from .run_in_processes import run_in_processes
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from .utils_waring import UtilsWarning
    from .plots import _check_plot_config
    from .util import get_datetime_str
    from .file_process import load_pkl_file, save_data_to_pkl_file


    _check_plot_config()
    log = logger.info if logger else print

    if res_pkl_path:
        res = load_pkl_file(res_pkl_path, logger=logger)
        save_res = False
        if res is not None:
            log(f"Successfully loaded results from {res_pkl_path}, skip saving results.")
        else:
            raise ValueError(f"Failed to load results from {res_pkl_path}, please check the file path.")
    else:
        res = None

    if save_res:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            log(f"Created directory {save_path}")


    if isinstance(smiles_groups[0], list):
        if alpha_list:
            assert len(alpha_list) == len(smiles_groups), f"The length of alpha_list should be the same as the number of groups, got {len(alpha_list)} and {len(smiles_groups)}."

        if clean_input_smiles_groups is True:
            smiles_groups = [clean_smiles_list(group, sanitize=sanitize, num_processes=num_processes, logger=logger) for group in smiles_groups]
        else:
            warnings.warn(UtilsWarning("Make sure the input SMILES are all valid SMILES strings, or set `clean_input_smiles_groups` to True."))

        if group_names:
            assert len(group_names) == len(smiles_groups), f"The length of group_names should be the same as the number of groups, got {len(group_names)} and {len(smiles_groups)}."
        else:
            group_names = [f"Group {i + 1}" for i in range(len(smiles_groups))]

    else:
        warnings.warn(UtilsWarning("Only one group of SMILES is provided."))
        if clean_input_smiles_groups is True:
            smiles_groups = clean_smiles_list(smiles_groups, sanitize=sanitize, num_processes=num_processes, logger=logger)
        else:
            warnings.warn(UtilsWarning("Make sure the input SMILES are all valid SMILES strings, or set `clean_input_smiles_groups` to True."))

    output_file = os.path.join(save_path, f"{name}.png")

    if cuml:
        try:
            import cuml.manifold as c
        except ImportError:
            warnings.warn(UtilsWarning('cuml is not installed. Please install it using `pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com`.'))
            c = None
            cuml = False

    if cuml and method == 'tsne':
        func = c.TSNE(n_components=2, random_state=random_state, perplexity=perplexity, n_neighbors=n_neighbors)
    elif cuml and method == 'umap':
        func = c.UMAP(n_components=2, random_state=random_state, n_neighbors=perplexity, min_dist=0.3, metric='cosine')
    elif method == 'umap':
        try:
            import umap
        except ImportError:
            warnings.warn(UtilsWarning('umap is not installed. Please install it using `pip install umap-learn`. Using t-sne algorithm.'))
            method = 'tsne'
        func = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=perplexity, min_dist=0.3, metric='cosine')
    elif method == 'tsne':
        func = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    else:
        raise ValueError(f"Unsupported method: {method}, please choose among ['tsne', 'umap']")

    if isinstance(smiles_groups[0], list):
        group_labels, smiles_list, fps = [], [], []

        for i, group in enumerate(smiles_groups):
            smiles_list.extend(group)
            group_labels.extend([i] * len(group))

        group_labels = np.array(group_labels)

        if res is not None:
            assert len(res) == len(smiles_list), f"The length of res should be the same as the number of smiles, got {len(res)} and {len(smiles_list)}, please check the input res_pkl_path."

        if res is None:
            parallel_decorator = run_in_processes(num_processes=num_processes)
            fp_calculator = parallel_decorator(calculate_fingerprint)

            fps = fp_calculator(smiles_list, fp_type=fp_type, radius=radius, nbits=nbits, sanitize=sanitize)

            fps = np.array(fps)
            res = func.fit_transform(fps)

            del func
            gc.collect()

            if save_res:
                save_data_to_pkl_file(res, save_path, f"{name}_tsne_res_{get_datetime_str()}", logger=logger)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 8))

        if isinstance(cmap, str):
            colors = plt.cm.get_cmap(cmap, len(group_names))
            indices = np.linspace(0, 1, len(group_names))
            colors = [colors(i) for i in indices]
        else:
            colors = cmap

        for i, name in enumerate(group_names):
            indices = np.where(group_labels == i)
            ax.scatter(res[indices, 0], res[indices, 1],
                       color=colors[i], label=name, alpha=alpha_list[i] if alpha_list else 0.7, s=s_list[i] if s_list else 20, marker=marker_list[i] if marker_list else '.')

        method_str = 't-SNE' if method == 'tsne' else 'UMAP'
        ax.set_title(f'Chemical Space Visualization using {method_str}')
        ax.set_xlabel(f'{method_str} Dimension 1')
        ax.set_ylabel(f'{method_str} Dimension 2')
        if show_legend:
            ax.legend(markerscale=1.5)
        ax.grid(False)

        plt.savefig(output_file, dpi=1200, bbox_inches='tight')
        log(f"Saved image to {output_file}")

    else:
        if res is not None:
            assert len(res) == len(smiles_groups), f'The length of res should be the same as the number of smiles, got {len(res)} and {len(smiles_groups)}, please check the input res_pkl_path.'
        else:
            parallel_decorator = run_in_processes(num_processes=num_processes)
            fp_calculator = parallel_decorator(calculate_fingerprint)

            fps = fp_calculator(smiles_groups, fp_type=fp_type, radius=radius, nbits=nbits, sanitize=sanitize)

            fps = np.array(fps)
            res = func.fit_transform(fps)

            del func
            gc.collect()

            if save_res:
                save_data_to_pkl_file(res, save_path, f"{name}_tsne_res_{get_datetime_str()}", logger=logger)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 10))

        ax.scatter(res[:, 0], res[:, 1], color='#007ACC', alpha=0.7, s=50)

        method_str = 't-SNE' if method == 'tsne' else 'UMAP'
        ax.set_title(f'Chemical Space Visualization using {method_str}')
        ax.set_xlabel(f'{method_str} Dimension 1')
        ax.set_ylabel(f'{method_str} Dimension 2')
        if show_legend:
            ax.legend(markerscale=1.5)
        ax.grid(False)

        plt.savefig(output_file, dpi=1200, bbox_inches='tight')
        log(f"Saved image to {output_file}")


def save_smiles_list_to_txt(smiles_list: List[str],
                            save_path: str,
                            name: str,
                            clean_input_smiles_list: bool = False,
                            sanitize: bool = True,
                            num_processes: int = None,
                            logger: logging.Logger = None):
    from .utils_waring import UtilsWarning
    if clean_input_smiles_list:
        smiles_list = clean_smiles_list(smiles_list, sanitize=sanitize, num_processes=num_processes, logger=logger, remove_error_results=True)
    else:
        warnings.warn(UtilsWarning("Make sure the input SMILES are all valid SMILES strings, or set `clean_input_smiles_list` to True."))

    log = logger.info if logger else print

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"Created directory {save_path}")

    output_file = os.path.join(save_path, f"{name}.txt")

    with open(output_file, 'w') as f:
        for smiles in smiles_list:
            if smiles is not None and smiles != '':
                f.write(smiles + '\n')

    log(f"Saved SMILES list to {output_file}, length: {len(smiles_list)}")


def calculate_molecular_properties(smiles: str, sanitize: bool = True, prop_list: list = None, logger: logging.Logger = None, quiet: bool = False):
    default_props_list = ['molwt', 'logp', 'hba', 'hbd', 'tpsa', 'sascore']
    default_props_set = set(default_props_list)
    if prop_list is None:
        prop_list = default_props_list

    _prop_set = set(prop_list)
    if not _prop_set.issubset(default_props_set):
        raise ValueError(f"Invalid property name, valid property names are: {default_props_list}, but got invalid property: {_prop_set - default_props_set}")

    log = logger.info if logger else print

    mol = _check_smiles(smiles, sanitize=sanitize)
    res = []
    for prop in prop_list:
        if prop == 'molwt':
            res.append(Descriptors.MolWt(mol))
        elif prop == 'logp':
            res.append(Crippen.MolLogP(mol))
        elif prop == 'hba':
            res.append(Descriptors.NumHAcceptors(mol))
        elif prop == 'hbd':
            res.append(Descriptors.NumHDonors(mol))
        elif prop == 'tpsa':
            res.append(Descriptors.TPSA(mol))
        elif prop =='sascore':
            res.append(sascorer.calculateScore(mol))

    if not quiet:
        log(f"Calculated molecular properties for {smiles}")
        for prop, value in zip(prop_list, res):
            log(f"{prop}: {value}")

    return res


class GroupCounter:
    def __init__(self, smiles, scaffold_smiles=None):
        self.smiles = smiles
        self.scaffold_smiles = scaffold_smiles
        self.scaffold_atom_indices = set()
        try:
            self.mol = _check_smiles(smiles, sanitize=True)
        except Exception as e:
            raise ValueError(f"Error processing SMILES string '{smiles}': {e}")

        if scaffold_smiles is not None:
            try:
                self.scaffold_mol = _check_smiles(scaffold_smiles, sanitize=True)
                match_indices_tuple = self.mol.GetSubstructMatches(self.scaffold_mol)
                if match_indices_tuple:
                    for item in match_indices_tuple:
                        self.scaffold_atom_indices.update(item)
            except Exception as e:
                raise ValueError(f"Error processing scaffold SMILES string '{scaffold_smiles}': {e}")
        else:
            raise ValueError("Scaffold SMILES string is required for group counting.")

        if not self.mol.HasSubstructMatch(self.scaffold_mol):
            raise ValueError(
                f"Molecular SMILES {self.smiles} does not contain scaffold {self.scaffold_smiles}.")


        self.FUNCTIONAL_GROUPS_PRIORITIZED = OrderedDict([
            ('Carboxylic Acid Group', '[CX3](=O)[OX2H1]'),  # Matches C, =O, OH from -COOH
            ('Acid Anhydride Core', '[CX3](=O)[OX2][CX3](=O)'),  # Matches C(=O)OC(=O) core
            ('Ester Linkage Core', '[CX3](=O)[OX2;!H1]'),  # Matches C, =O, -O- from ester
            ('Amide Core', '[CX3](=O)[N]'),  # Matches C, =O, N from amide (general N)
            ('Lactone Core', '[C&R](=O)[O&X2&R;!H1;!$(O[C&X3]=O)]'),
            ('Lactam Core', '[C&R](=O)[N&R]'),  # Lactam C=O-N in a ring
            ('Aldehyde Group', '[CH1X3]=O'),  # Matches C, H, =O from -CHO (CH1X3 ensures it's a terminal formyl C)
            ('Ketone Carbonyl', '[#6][C;X3;!$(C(=O)[O,N]);!$([CH1X3]=O)](=O)[#6]'),

            ('Indole Ring', 'c1ccc2c(c1)[nH]cc2'),
            ('Quinoline Ring', 'c1ccc2c(c1)cccn2'),
            ('Phenyl Ring', 'c1ccccc1'),
            ('Pyridine Ring', 'n1ccccc1'),
            ('Pyrrole Ring', 'n1cccc1'),
            ('Furan Ring', 'o1cccc1'),
            ('Thiophene Ring', 's1cccc1'),
            ('Imidazole Ring', 'n1cncc1'),  # Common SMARTS
            ('Pyrazole Ring', 'n1nccc1'),  # Common SMARTS
            ('Pyrimidine Ring', 'n1cnccc1'),  # Common SMARTS for pyrimidine
            ('Triazoles Ring', 'c1nnnc1'),

            # ('Phenolic Hydroxyl', '[c]-[OH1]'),  # Matches C-O-H on an aromatic carbon
            # ('Alcoholic Hydroxyl', '[C;!$(C=O);!a]-[OH1]'),  # Matches C-O-H on a non-carbonyl, non-aromatic carbon
            ('Enol Hydroxyl', '[C!a]=[C!a]-[OH1]'),  # Matches C=C-O-H
            # ('Ether Linkage', '[#6][OD2][#6]'),  # Matches C-O-C (general)
            ('Epoxide', '[O;X2;r3]1[C;X4;r3][C;X4;r3]1'),
            ('Nitro Group', '[N+](=O)[O-]'),
            ('Nitrile Group', 'C#N'),  # C attached to CN

            # ('Primary Amine', '[NH2;X3;!$(NC=O)]'),  # NH2 (3 connections), not part of amide
            # ('Secondary Amine', '[NH1;X3;!$(NC=O)]([#6])[#6]'),  # NH (3 connections), not part of amide
            # ('Tertiary Amine', '[NH0;X3;!$(NC=O)]([#6])([#6])[#6]'),  # N (3 connections), not part of amide
            ('Thiol Group', '[#6][SH1]'),  # -SH attached to a carbon
            ('Sulfide Linkage', '[#6][SD2][#6]'),  # C-S-C

            ('Trifluoromethyl', 'C(F)(F)F'),
            ('Trichloromethyl', 'C(Cl)(Cl)Cl'),
            ('Halogen (F)', '[F]'),
            ('Halogen (Cl)', '[Cl]'),
            ('Halogen (Br)', '[Br]'),
            ('Halogen (I)', '[I]'),
            ('Alkene (C=C)', '[C]=[C]'),
            ('Alkyne (C#C)', '[CX2]#[CX2]'),
            ('Nitrogen (N)', '[N]'),
            ('Oxygen (OH)', '[OH]'),
            ('Ether (O)', '[O]'),
            ('Methyl (CH3)', '[CH3]'),
            ('Methyl (CH2)', '[CH2]'),
            ('Methyl (CH)', '[CH]'),
        ])

    def count_functional_groups(self):
        counts = Counter()
        assigned_core_atoms = set()

        for group_name, smarts_pattern_str in self.FUNCTIONAL_GROUPS_PRIORITIZED.items():
            if not smarts_pattern_str:
                raise ValueError(f"SMARTS pattern for group '{group_name}' is empty.")
            pattern = Chem.MolFromSmarts(smarts_pattern_str)
            if pattern is None:
                raise ValueError(f"SMARTS pattern for group '{group_name}' is invalid.")

            matches = self.mol.GetSubstructMatches(pattern, useChirality=True, useQueryQueryMatches=True)

            for match_atoms_tuple in matches:
                current_match_atoms = set(match_atoms_tuple)

                if self.scaffold_atom_indices and current_match_atoms.issubset(self.scaffold_atom_indices):
                    continue

                is_blocked_by_prior_assignment = False
                for atom_idx in current_match_atoms:
                    if atom_idx in assigned_core_atoms:
                        is_blocked_by_prior_assignment = True
                        break

                if not is_blocked_by_prior_assignment:
                    counts[group_name] += 1
                    assigned_core_atoms.update(current_match_atoms)

        identified_specific_aromatic_ring_atom_sets = []
        for group_name, smarts_pattern_str in self.FUNCTIONAL_GROUPS_PRIORITIZED.items():
            if "Ring" in group_name and counts.get(group_name, 0) > 0:

                pattern = Chem.MolFromSmarts(smarts_pattern_str)
                if pattern:
                    matches = self.mol.GetSubstructMatches(pattern)
                    for match_atoms_tuple in matches:
                        identified_specific_aromatic_ring_atom_sets.append(set(match_atoms_tuple))

        unassigned_aromatic_sssr_rings_count = 0
        if hasattr(Chem, 'GetSymmSSSR'):
            for sssr_ring_atom_indices_tuple in Chem.GetSymmSSSR(self.mol):
                sssr_ring_atoms = set(sssr_ring_atom_indices_tuple)
                if not sssr_ring_atoms:
                    continue
                if self.scaffold_atom_indices and sssr_ring_atoms.issubset(self.scaffold_atom_indices):
                    continue

                is_valid_aromatic_sssr_ring = True
                try:
                    for atom_idx in sssr_ring_atoms:
                        atom = self.mol.GetAtomWithIdx(atom_idx)
                        if not atom.GetIsAromatic():
                            is_valid_aromatic_sssr_ring = False
                            break
                except RuntimeError:
                    is_valid_aromatic_sssr_ring = False

                if is_valid_aromatic_sssr_ring:
                    is_covered_by_specific_smarts = False
                    for specific_ring_set in identified_specific_aromatic_ring_atom_sets:
                        if sssr_ring_atoms == specific_ring_set:
                            is_covered_by_specific_smarts = True
                            break
                    if not is_covered_by_specific_smarts:
                        unassigned_aromatic_sssr_rings_count += 1

        if unassigned_aromatic_sssr_rings_count > 0:
            counts['Other Aromatic Rings (SSSR based, not specifically named)'] = unassigned_aromatic_sssr_rings_count

        return counts


def group_count(input, scaffold=None, save_path='./group_count/', name='group_count', save_fig=False, logger: logging.Logger = None):
    log = logger.info if logger else print
    if isinstance(input, str):
        cnt = GroupCounter(input, scaffold).count_functional_groups()
    elif isinstance(input, list):
        cnt = Counter()
        errors = []
        for smi in input:
            try:
                cnt.update(GroupCounter(smi, scaffold).count_functional_groups())
            except:
                errors.append(smi)

        log(f'error smiles: {errors}')
        log(f'number of error smiles: {len(errors)}')

    else:
        raise ValueError(f'Input must be a SMILES string or a list of SMILES strings, but got {type(input)}.')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cnt_dict = dict(cnt)
    save_data_to_json_file(cnt_dict, save_path, name, logger=logger)

    if save_fig:
        draw_distribution_plot(cnt_dict, save_path, name, xlabel='Group', ylabel='Count', figsize=(10, 8), logger=logger, title='Group Count Distribution')

