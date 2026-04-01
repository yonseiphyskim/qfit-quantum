"""Common argparse setup and related CLI utility methods"""

import argparse
import logging
import os.path

from qfit.command_line.custom_argparsers import (
    ToggleActionFlag,
    CustomHelpFormatter,
    ValidateMapFileArgument,
    ValidateStructureFileArgument,
)
from qfit.solvers import available_qp_solvers, available_qubo_solvers
from qfit import MapScaler, XMap

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORMER = "qfit"


def get_base_argparser(description,
                       default_enable_external_clash=False,
                       default_transformer=DEFAULT_TRANSFORMER):
    """
    Define and return the base argparse configuration for QUBO-based qFit runs.

    """
    p = argparse.ArgumentParser(
        formatter_class=CustomHelpFormatter, description=description
    )

    # ==============================
    # Input: map and structure files
    # ==============================
    p.add_argument(
        "map",
        help="Density map (CCP4/MRC) or MTZ with reflections and phases.",
        type=str,
        action=ValidateMapFileArgument,
    )
    p.add_argument(
        "structure",
        help="PDB or mmCIF file containing structure.",
        type=str,
        action=ValidateStructureFileArgument,
    )

    # ==============================
    # Map input options
    # ==============================
    mo = p.add_argument_group("Map options")
    p.add_argument(
        "-l",
        "--label",
        default="2FOFCWT,PH2FOFCWT",
        metavar="<F,PHI>",
        help="MTZ column labels to build density",
    )
    p.add_argument(
        "-r",
        "--resolution",
        default=None,
        metavar="<float>",
        type=float,
        help="Map resolution (Å) (for CCP4 maps)",
    )
    p.add_argument(
        "-m",
        "--resolution-min",
        default=None,
        metavar="<float>",
        type=float,
        help="Lower resolution bound (Å)",
    )
    mo.add_argument(
        "-o",
        "--omit",
        action="store_true",
        help="Treat map file as an OMIT map in map scaling routines",
    )

    # ==============================
    # Map prep and scaling
    # ==============================
    mo.add_argument(
        "--scale",
        action=ToggleActionFlag,
        dest="scale",
        default=True,
        help="Scale density map (on/off)",
    )
    mo.add_argument(
        "-sv",
        "--scale-rmask",
        dest="scale_rmask",
        default=1.0,
        metavar="<float>",
        type=float,
        help="Scaling factor for soft-clash mask radius",
    )
    mo.add_argument(
        "-dc",
        "--density-cutoff",
        default=0.3,
        metavar="<float>",
        type=float,
        help="Density values below this are replaced by cutoff value",
    )
    mo.add_argument(
        "-dv",
        "--density-cutoff-value",
        default=-1,
        metavar="<float>",
        type=float,
        help="Replacement value for density below cutoff",
    )
    mo.add_argument(
        "--subtract",
        action=ToggleActionFlag,
        dest="subtract",
        default=True,
        help="Subtract neighboring residues' Fcalc during modeling",
    )
    mo.add_argument(
        "-pad",
        "--padding",
        default=8.0,
        metavar="<float>",
        type=float,
        help="Padding size for map creation",
    )

    p.add_argument(
        "--transformer",
        choices=["cctbx", "qfit"],
        default=default_transformer,
        dest="transformer",
        help="Map sampling algorithm (FFT engine)",
    )
    p.add_argument(
        "--transformer-map-coeffs",
        choices=["cctbx", "qfit"],
        default=None,
        dest="transformer_map_coeffs",
        help="Map coefficients FFT implementation (for testing gridding behavior)",
    )

    p.add_argument(
        "--expand-to-p1",
        action="store_true",
        dest="expand_to_p1",
        default=None,
        help="Force P1 expansion even if not required",
    )
    p.add_argument(
        "--no-expand-to-p1",
        action="store_false",
        dest="expand_to_p1",
        default=None,
        help="Disable P1 expansion when map is already complete",
    )

    # ==============================
    # Clash and density filtering
    # ==============================
    p.add_argument(
        "--waters-clash",
        action=ToggleActionFlag,
        dest="waters_clash",
        default=True,
        help="Include water molecules in clash detection",
    )
    p.add_argument(
        "--remove-conformers-below-cutoff",
        action="store_true",
        dest="remove_conformers_below_cutoff",
        help="Remove conformers with density below cutoff",
    )
    p.add_argument(
        "-cf",
        "--clash-scaling-factor",
        default=0.75,
        metavar="<float>",
        type=float,
        help="Set clash scaling factor",
    )

    if default_enable_external_clash:
        p.add_argument(
            "-ec",
            "--no-external-clash",
            action="store_false",
            dest="external_clash",
            help="Turn off external clash detection during sampling",
        )
    else:
        p.add_argument(
            "-ec",
            "--external-clash",
            action="store_true",
            dest="external_clash",
            help="Enable external clash detection during sampling",
        )

    # ==============================
    # General modeling parameters
    # ==============================
    p.add_argument(
        "-bs",
        "--bulk-solvent-level",
        default=0.3,
        metavar="<float>",
        type=float,
        help="Bulk solvent level in absolute values",
    )
    p.add_argument(
        "-c",
        "--cardinality",
        default=5,
        metavar="<int>",
        type=int,
        help="Cardinality constraint (max conformers for QUBO)",
    )
    p.add_argument(
        "-t",
        "--threshold",
        default=0.2,
        metavar="<float>",
        type=float,
        help="Threshold constraint for occupancy (QUBO)",
    )
    p.add_argument(
        "-hy",
        "--hydro",
        action="store_true",
        dest="hydro",
        help="Include hydrogens during calculations",
    )
    p.add_argument(
        "-rmsd",
        "--rmsd-cutoff",
        default=0.01,
        metavar="<float>",
        type=float,
        help="RMSD cutoff for removal of identical conformers",
    )

    # ==============================
    # Solver options (QUBO)
    # ==============================
    p.add_argument(
        "--qp-solver",
        dest="qp_solver",
        choices=available_qp_solvers.keys(),
        default=next(iter(available_qp_solvers.keys())),
        help="Select the QP solver",
    )
    p.add_argument(
        "--qubo-solver",
        dest="qubo_solver",
        choices=available_qubo_solvers.keys(),
        default=next(iter(available_qubo_solvers.keys())),
        help="Select the QUBO solver",
    )
    p.add_argument(
        "-p",
        "--nproc",
        type=int,
        default=1,
        metavar="<int>",
        help="Number of parallel threads",
    )

    # ==============================
    # Output options
    # ==============================
    og = p.add_argument_group("Output options")
    og.add_argument(
        "-d",
        "--directory",
        default=".",
        metavar="<dir>",
        type=os.path.abspath,
        help="Directory to store results",
    )
    og.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    og.add_argument(
        "--debug", action="store_true", help="Enable detailed debug logging"
    )
    og.add_argument(
        "--write_intermediate_conformers",
        action="store_true",
        help="Write intermediate structures (for debugging)",
    )
    og.add_argument("--pdb", help="Name of the input PDB file")
    return p


def load_and_scale_map(options, structure):
    """
    Load and scale experimental density map for QUBO workflow.
    """
    # Load map
    map_transformer = options.transformer_map_coeffs or options.transformer
    xmap = XMap.fromfile(
        options.map,
        resolution=options.resolution,
        label=options.label,
        transformer=map_transformer
    )

    expand_to_p1 = options.expand_to_p1
    if expand_to_p1 is None:
        expand_to_p1 = map_transformer == "qfit"
    xmap = xmap.canonical_unit_cell(expand_to_p1=expand_to_p1)

    # Apply model-based scaling
    if options.scale is True:
        # safe getattr() for em attribute
        em_flag = getattr(options, "em", False)
        scaler = MapScaler(xmap, em=em_flag, debug=options.debug)
        radius = 1.5
        reso = xmap.resolution.high or options.resolution
        if reso is not None:
            radius = 0.5 + reso / 3.0
        logger.info("Scaling with resolution=%.3f radius=%.3f", reso, radius)
        scaler.scale(
            structure,
            radius=options.scale_rmask * radius,
            transformer=options.transformer
        )
    return xmap

