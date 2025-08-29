"""
-*- coding:utf-8 -*-
@Time      :2025/8/13 上午10:58
@Author    :Chen Junpeng

"""
import logging
import os
import sys
import warnings
from typing import List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolToFile, MolDrawOptions, rdMolDraw2D
from rdkit import RDLogger
from rdkit.Chem import Crippen, Descriptors
try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    sys.path.append(os.path.join(os.environ['CONDA_PREFIX'], 'share', 'RDKit', 'Contrib'))
    from SA_Score import sascorer
from tqdm import tqdm

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


def draw_molecule_to_png(smiles: str, save_path: str, name: str, sanitize: bool = True, show_atom_numbers: bool = False, size: (int, int) = (400, 400), logger: logging.Logger = None) -> None:
    """
    Draw a molecule to a PNG file.
    :param smiles: the input SMILES string.
    :param save_path: the path to save the PNG file.
    :param name: the name of the PNG file.
    :param sanitize: sanitize the input SMILES string or not.
    :param show_atom_numbers: show the atom numbers or not.
    :param size: the size of the molecule.
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
        warnings.warn(UtilsWarning(f"Could not generate 2D coordinates for SMILES: {smiles}, use unoptimized coordinates."))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"Created directory {save_path}")

    png_file = os.path.join(save_path, f"{name}.png")

    drawing_options = MolDrawOptions()
    if show_atom_numbers:
        drawing_options.addAtomIndices = True

    try:
        MolToFile(mol,
                  png_file,
                  size=size,
                  options=drawing_options)
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
            png_bytes = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'))
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

            pdf.savefig(fig, dpi=300)
            plt.close(fig)
    log(f"Saved pdf file to {pdf_path}")


def draw_chemical_space_plot(smiles_groups: List[Union[str, List[str]]],
                             save_path: str,
                             name: str,
                             group_names: list = None,
                             fp_type: str = 'ecfp',
                             radius: int = 2,
                             nbits: int = 2048,
                             sanitize: bool = True,
                             perplexity: int = 30,
                             random_state: int = 42,
                             num_processes: int = None,
                             alpha_list: list = None,
                             marker_list: list = None,
                             cmap: str | list = None,
                             show_legend: bool = False,
                             clean_input_smiles_groups: bool = False,
                             logger: logging.Logger = None):
    """
    Draw a chemical space plot using t-SNE.
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
    :param marker_list: a list of marker values for each group.
    :param cmap: the color map to use.
    :param show_legend: show the legend on the figure or not.
    :param clean_input_smiles_groups: clean the smiles groups before drawing the plot or not.
    :param logger: the logger to use.
    :return:
    """
    from .run_in_processes import run_in_processes
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from .utils_waring import UtilsWarning

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
        warnings.warn(UtilsWarning("Only one group of SMILES is provided, no need to visualize chemical space."))
        if clean_input_smiles_groups is True:
            smiles_groups = clean_smiles_list(smiles_groups, sanitize=sanitize, num_processes=num_processes, logger=logger)
        else:
            warnings.warn(UtilsWarning("Make sure the input SMILES are all valid SMILES strings, or set `clean_input_smiles_groups` to True."))

    log = logger.info if logger else print

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"Created directory {save_path}")

    output_file = os.path.join(save_path, f"{name}.png")

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)

    if isinstance(smiles_groups[0], list):
        group_labels, smiles_list, fps = [], [], []

        for i, group in enumerate(smiles_groups):
            smiles_list.extend(group)
            group_labels.extend([i] * len(group))

        group_labels = np.array(group_labels)

        parallel_decorator = run_in_processes(num_processes=num_processes)
        fp_calculator = parallel_decorator(calculate_fingerprint)

        fps = fp_calculator(smiles_list, fp_type=fp_type, radius=radius, nbits=nbits, sanitize=sanitize)

        fps = np.array(fps)
        res = tsne.fit_transform(fps)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 10))

        if isinstance(cmap, str):
            colors = plt.cm.get_cmap(cmap, len(group_names))
            indices = np.linspace(0, 1, len(group_names))
            colors = [colors(i) for i in indices]
        else:
            colors = cmap

        for i, name in enumerate(group_names):
            indices = np.where(group_labels == i)
            ax.scatter(res[indices, 0], res[indices, 1],
                       color=colors[i], label=name, alpha=alpha_list[i] if alpha_list else 0.7, s=20, marker=marker_list[i] if marker_list else '.')

        ax.set_title('Chemical Space Visualization using t-SNE', fontsize=20)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
        if show_legend:
            ax.legend(fontsize=12, markerscale=1.5)
        ax.grid(False)

        plt.savefig(output_file, dpi=1200, bbox_inches='tight')
        log(f"Saved image to {output_file}")

    else:
        parallel_decorator = run_in_processes(num_processes=num_processes)
        fp_calculator = parallel_decorator(calculate_fingerprint)

        fps = fp_calculator(smiles_groups, fp_type=fp_type, radius=radius, nbits=nbits, sanitize=sanitize)

        fps = np.array(fps)
        res = tsne.fit_transform(fps)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 10))

        ax.scatter(res[:, 0], res[:, 1], color='#007ACC', alpha=0.7, s=50)

        ax.set_title('Chemical Space Visualization using t-SNE', fontsize=20)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
        if show_legend:
            ax.legend(fontsize=12, markerscale=1.5)
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


def calculate_molecular_properties(smiles: str, sanitize: bool = True, prop_list: list = None, logger: logging.Logger = None):
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

    log(f"Calculated molecular properties for {smiles}")
    for prop, value in zip(prop_list, res):
        log(f"{prop}: {value}")

    return res




