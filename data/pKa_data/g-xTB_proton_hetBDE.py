from peregrine.moleculeset import MoleculeSet
from peregrine.molecule import Molecule
from peregrine.atom import Atom
import pandas as pd
from pathlib import Path
import os
import shutil
import subprocess
import re
import matplotlib.pyplot as plt
from scipy.stats import linregress

os.environ["OMP_NUM_THREADS"] = "1"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 400
plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["mathtext.fontset"] = "cm"

if __name__ == "__main__":
    
    pka_df = pd.read_csv(Path(__file__).parent / "aqueous_pka_data.csv")
    pka_df.set_index("Identifier", inplace=True)

    """protonated_ms = MoleculeSet.ReadSMILESList(
        SMILES_list=list(pka_df["ProtonatedCompound"]),
        Identifier_List=list(pka_df.index),
    )
    ms = MoleculeSet.ReadSMILESList(
        SMILES_list=list(pka_df["Compound"]),
        Identifier_List=list(pka_df.index), 
    )
    protonated_ms.OptimiseGeometry_xTB_bin()
    protonated_ms.WriteMolFileDirectory(
        Path(__file__).parent / "ProtonatedCompounds_g-xTB-Opt"
    )
    ms.OptimiseGeometry_xTB_bin()
    ms.WriteMolFileDirectory(
        Path(__file__).parent / "Compounds_g-xTB-Opt"
    )"""

    protonated_ms = MoleculeSet.ReadMolFileDirectory(
        Path(__file__).parent / "ProtonatedCompounds_g-xTB-Opt"
    )
    protonated_ms.WritePsi4Input(
        Path(__file__).parent / "ProtonatedCompounds-g-xTB-Opt_M06-2X-def2-SVP",
        method="m06-2x",
        basisset="def2-svp",
        restricted=True,
        CPU_count=6,
        job_scheduler_used=None,
        max_memory=4000,
        set_options={
            "ddx": True,
            "ddx_model": "pcm",        # pcm, cosmo or lpb
            "ddx_solvent": "water",
            "ddx_radii_set": "uff",    # or bondi
        }
    )
    new_protonated_ms = MoleculeSet.ReadPsi4Output(
        Path(__file__).parent / "ProtonatedCompounds-g-xTB-Opt_M06-2X-def2-SVP",
        Path(__file__).parent / "ProtonatedCompounds-g-xTB-Opt_M06-2X-def2-SVP_Psi4Output",
        template_moleculeset=protonated_ms,
    )
    ms = MoleculeSet.ReadMolFileDirectory(
        Path(__file__).parent / "Compounds_g-xTB-Opt"
    )
    ms.WritePsi4Input(
        Path(__file__).parent / "Compounds-g-xTB-Opt_M06-2X-def2-SVP",
        method="m06-2x",
        basisset="def2-svp",
        restricted=True,
        CPU_count=6,
        job_scheduler_used=None,
        max_memory=4000,
        set_options={
            "ddx": True,
            "ddx_model": "pcm",        # pcm, cosmo or lpb
            "ddx_solvent": "water",
            "ddx_radii_set": "uff",    # or bondi
        }
    )
    new_ms = MoleculeSet.ReadPsi4Output(
        Path(__file__).parent / "Compounds-g-xTB-Opt_M06-2X-def2-SVP",
        Path(__file__).parent / "Compounds-g-xTB-Opt_M06-2X-def2-SVP_Psi4Output",
        template_moleculeset=ms,
    )
    
    # === g-xtb analysis ===
    identifiers = [molObj.Identifier for molObj in protonated_ms.MoleculesDict.values()]
    prot_elec_ens = [protonated_ms.MoleculesDict[identifier].electronic_energy for identifier in identifiers]
    elec_ens = [ms.MoleculesDict[identifier].electronic_energy for identifier in identifiers]
    pka = [pka_df.loc[identifier, "pKa"] for identifier in identifiers]
    atom_centres = [pka_df.loc[identifier, "AtomCentre"] for identifier in identifiers]
    hybrid = [pka_df.loc[identifier, "FunctionalGroup"] for identifier in identifiers]
    charge = [ms.MoleculesDict[identifier].FormalCharge for identifier in identifiers]
    res_df = pd.DataFrame(
        {
            "Identifier": identifiers,
            "Atom Centres": atom_centres,
            "Functional Group":  hybrid,
            "Formal Charge": charge,
            "Aqueous pKa": pka,
            "Protonated Elec En (Eh)": prot_elec_ens,
            "Unprotonated Elec En (Eh)": elec_ens,
        }
    )
    res_df["Elec En Diff (Eh)"] = res_df["Protonated Elec En (Eh)"] - res_df["Unprotonated Elec En (Eh)"]
    res_df.to_csv(Path(__file__).parent / "g-xTB_calculated_protonation_energies.csv")

    #res_df = res_df[res_df["Functional Group"] != "Water"]
    res_df = res_df[res_df["Functional Group"] != "Halide"]
    res_df = res_df[res_df["Identifier"] != "a55"]

    fig, ax = plt.subplots()
    ax.scatter(
        res_df[res_df["Formal Charge"] == 0]["Aqueous pKa"],
        res_df[res_df["Formal Charge"] == 0]["Elec En Diff (Eh)"],
    )
    ax.scatter(
        res_df[res_df["Formal Charge"] == -1]["Aqueous pKa"],
        res_df[res_df["Formal Charge"] == -1]["Elec En Diff (Eh)"],
    )
    fig.savefig(Path(__file__).parent / "gxtb_pka_scatter.png", bbox_inches="tight")

    # === g-xtb/m06-2x analysis ===
    identifiers = [molObj.Identifier for molObj in new_protonated_ms.MoleculesDict.values()]
    prot_elec_ens = [new_protonated_ms.MoleculesDict[identifier].electronic_energy for identifier in identifiers]
    elec_ens = [new_ms.MoleculesDict[identifier].electronic_energy for identifier in identifiers]
    pka = [pka_df.loc[identifier, "pKa"] for identifier in identifiers]
    atom_centres = [pka_df.loc[identifier, "AtomCentre"] for identifier in identifiers]
    hybrid = [pka_df.loc[identifier, "FunctionalGroup"] for identifier in identifiers]
    charge = [new_ms.MoleculesDict[identifier].FormalCharge for identifier in identifiers]
    res_df = pd.DataFrame(
        {
            "Identifier": identifiers,
            "Atom Centres": atom_centres,
            "Functional Group":  hybrid,
            "Formal Charge": charge,
            "Aqueous pKa": pka,
            "Protonated Elec En (Eh)": prot_elec_ens,
            "Unprotonated Elec En (Eh)": elec_ens,
        }
    )
    res_df["Elec En Diff (Eh)"] = res_df["Protonated Elec En (Eh)"] - res_df["Unprotonated Elec En (Eh)"]
    res_df.to_csv(Path(__file__).parent / "g-xTB-m062x_calculated_protonation_energies.csv")

    res_df = res_df[res_df["Functional Group"] != "Water"]
    res_df = res_df[res_df["Functional Group"] != "Halide"]
    res_df = res_df[res_df["Identifier"] != "a55"]

    fig, ax = plt.subplots()
    ax.scatter(
        res_df[res_df["Formal Charge"] == 0]["Aqueous pKa"],
        res_df[res_df["Formal Charge"] == 0]["Elec En Diff (Eh)"],
        color="#D81B60",
        s=5,
    )
    ax.scatter(
        res_df[res_df["Formal Charge"] == -1]["Aqueous pKa"],
        res_df[res_df["Formal Charge"] == -1]["Elec En Diff (Eh)"],
        color="#1E88E5",
        s=5,
    )
    res = linregress(res_df["Aqueous pKa"], res_df["Elec En Diff (Eh)"])
    print(f"{res.slope}, {res.intercept}")
    print(res)
    ax.set_ylabel("g-xTB//M06-2X/def2-SVP/PCM(Water) (Eh)")
    ax.set_xlabel("pKa")
    fig.savefig(Path(__file__).parent / "m062x_pka_scatter.png", bbox_inches="tight")