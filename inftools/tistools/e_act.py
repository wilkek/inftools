from typing import Annotated
import typer
import numpy as np
import pathlib
import tomli
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


    

def activation_energy(
    kb: Annotated[float, typer.Option("-kb", help = "kB, in the correct energy unit of entries of energy.txt default: 8.617333262e-5 eV/K")] = 8.617333262*10**(-5),
    wham: Annotated[str, typer.Option("-wham", help="path to path_weights*.txt")] = "wham",
    load: Annotated[str, typer.Option("-load", help = "directory containing the paths")] = "load",
    toml: Annotated[str, typer.Option("-toml", help = "path to .toml file")] = "infretis.toml",
    Nbins: Annotated[int, typer.Option("-Nbins", help = "number of bin along OP")] = 100,
    out: Annotated[str, typer.Option("-out", help="path to store report")] = "wham",
    h5: Annotated[str, typer.Option("-h5", help="The h5 file if order.txt are stored in an h5 file. Requires the h5py package.")] = None,

):
    def save_axis(ax, basepath: Path, stem: str, formats=("pdf",)):
        # tight bbox in display units, expanded slightly for labels/ticks
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.05)
        # convert to inches for bbox_inches
        bbox_in = bbox.transformed(fig.dpi_scale_trans.inverted())
        for ext in formats:
            fig.savefig(basepath / f"{stem}.{ext}", bbox_inches=bbox_in, dpi=300)

    if h5 is not None:
        import h5py
        h5file = h5py.File(h5, 'r')
    else:
        load = pathlib.Path(load)
    wham = pathlib.Path(wham)
    out = pathlib.Path(out)
    pathw = np.loadtxt(wham / 'path_weights_plus.txt')
    pathw0 = np.loadtxt(wham / 'path_weights_minus.txt')
    min_op = []

    print('Find Minimum OP value across all paths to set binning range')
    for i, (path_nr, max_op, path_w) in enumerate(tqdm.tqdm(pathw0)):

        if h5 is not None:
            if str(int(path_nr)) in h5file:
                if "order.txt" in h5file[str(int(path_nr))]:
                    op = h5file[str(int(path_nr))]["order.txt"][:]
                else:
                    print(f"Skipping path {path_nr} due to missing files in h5.")
                    continue
            else:
                print(f"Skipping path {path_nr} due to missing files in h5.")
                continue
        else:
            path = load/str(int(path_nr))
            # skip if files are not found
            if path.is_dir():
                if not (path/"order.txt").exists() or not (path/"energy.txt").exists():
                    print(f"Skipping path {path_nr} due to missing files.")
                    continue
            else:
                continue

            op = np.loadtxt(path/"order.txt", usecols=[1])
        min_op.append(np.min(op))
    min_op = np.min(np.array(min_op))

    with open(toml, "rb") as rfile:
        toml = tomli.load(rfile)
    intfA = toml["simulation"]["interfaces"][0]
    intfB = toml["simulation"]["interfaces"][-1]
    temp = float(toml["engine"]["temperature"])
    beta = 1/(temp*kb)
    bins = np.linspace(min_op, np.max(pathw[:,1])+1e-5, Nbins)
    counts = np.zeros(Nbins)
    countsE = np.zeros(Nbins)
    # path energy for A to B paths (1 value each path)
    eAB = []
    wAB = []
    # energy of phase-points for all plus paths (1 value each phase point)
    eA = []
    wA = []
    successful_count = []
    # append pathw0 to pathw:
    pathw = np.vstack((pathw0, pathw))
    print('Processing paths...')    
    for i, (path_nr, max_op, path_w) in enumerate(tqdm.tqdm(pathw)):

        if h5 is not None:
            if str(int(path_nr)) in h5file:
                if "order.txt" in h5file[str(int(path_nr))] and "energy.txt" in h5file[str(int(path_nr))]:
                    op = h5file[str(int(path_nr))]["order.txt"][:,1]
                    en = h5file[str(int(path_nr))]["energy.txt"][:,1:3]
                else:
                    print(f"Skipping path {path_nr} due to missing files in h5.")
                    continue
            else:
                print(f"Skipping path {path_nr} due to missing files in h5.")
                continue
        else:
        # print(f"Processing path {str(int(path_nr))}")
            path = load/str(int(path_nr))

            # skip if files are not found
            if path.is_dir():
                if not (path/"order.txt").exists() or not (path/"energy.txt").exists():
                    print(f"Skipping path {path_nr} due to missing files.")
                    continue
            else:
                continue

            op = np.loadtxt(path/"order.txt", usecols=[1])
            print(op)
            en = np.loadtxt(path/"energy.txt", usecols = [1,2])

        # check that path_weights.txt and order.txt values match such that also energy.txt match
        round_diff = np.abs(np.max(op) - max_op)
        if  round_diff > 1e-5:
            print(f"[WARNING] Max order value in {path_nr}/order.txt does not match that in {path_w}, diff = {round_diff}")
            continue
        binidx = np.digitize(op, bins, right=False)
        np.add.at(counts, binidx, path_w)

        # total energy, ekin + vpot
        etot = en[:,0] + en[:,1]

        # use median here due to constraint artifacts in Amber with NVE
        # emed = np.median(etot[~np.isnan(etot)])
        emed = np.mean(etot[~np.isnan(etot)])
        np.add.at(countsE, binidx, path_w*emed)
        # A to B path
        if op[0]<=intfA and op[-1]>=intfB:
            eAB.append(emed)
            wAB.append(path_w)
        # overall state A path (also contains A to B paths)
        # path_weights.txt only contains plus paths
        eA += [emed]
        wA += [path_w*len(op)]
        successful_count.append(i)
    # only count paths that where actually succesfully loaded, some might miss energy.txt, or order.txt, etc.
    pathw = pathw[successful_count]
    eA = np.array(eA)
    eAB = np.array(eAB)
    EA = np.sum(wA*eA)/np.sum(wA)
    EAB = np.sum(eAB*wAB)/np.sum(wAB)
    Eact = (EAB - EA)*beta
    print(f"<EA> = {EA*beta}, <EAB> = {EAB*beta}, Eactivation = {(EAB - EA)*beta}")
    # now construct continuous activation energy curve
    # weights up to some max lambda
    idx = np.argsort(pathw[:,1])
    sorted_pathw = pathw[idx]
    sorted_eA = eA[idx]
    x = [] # order parameter
    y = [] # activation energy
    for i in range(len(sorted_pathw)):
        wAL = sorted_pathw[i:, 2]
        x.append(sorted_pathw[i,1])
        y.append(np.sum(sorted_eA[i:]*wAL)/np.sum(wAL) - EA)
    fe = np.loadtxt(wham / "histo_free_energy.txt")
    xfe = np.loadtxt(wham / "histo_xval.txt")

    with PdfPages(out / "E_report.pdf") as pdf:
        # All figures on one PDF page
        fig, axs = plt.subplots(2, 3, figsize=(12, 7))
        axs[0, 0].hist(eAB,bins=20, density=True, weights=wAB,label="reactive (A->B) path energies")
        axs[0, 0].hist(eA,bins=20,alpha=0.5, density=True, weights = wA, label="overall state A (including A->) path energies")
        axs[0, 0].legend()

        # axs[0, 1].plot(x,np.array(y),label="enthalpy",ls="--")
        # axs[0, 1].plot(xfe, fe, label="free-energy",ls="--")
        axs[0, 1].axhline(Eact,label=f"E_Act: {np.round(Eact, 3)} kbT",c="C1",ls=":")
        axs[0, 1].axvline(intfA,c="k")
        axs[0, 1].axvline(intfB,c="k")
        axs[0, 1].set(ylabel="energy in kBT units",xlabel="order parameter")
        axs[0, 1].axhline(0,c="k")
        axs[0, 1].legend()
        
        myfe = -np.log(counts)
        myfe = myfe - np.min(myfe)
        avE = countsE/counts
        print(avE)
        myE = (avE - np.min(avE[~np.isnan(avE)]))*beta
        axs[0, 1].plot(bins,myfe,label="FE")



        axs[0, 2].plot(bins,myE,label="U")
        axs[0, 2].plot(bins,-myE + myfe,label="-TS")
        axs[0, 2].axvline(intfA,c="k")
        axs[0, 2].axvline(intfB,c="k")
        axs[0, 2].axhline(Eact,label=f"E_Act: {np.round(Eact, 3)} kbT",c="C1",ls=":")
        axs[0, 2].set(ylabel="energy in kBT units",xlabel="order parameter")
        axs[0, 2].axhline(0,c="k")
        axs[0, 2].legend()

        
        plt.tight_layout()
        pdf.savefig(fig)
        
        fig.canvas.draw()

        names = [
            "E1_hist_E_tot",
            "E2_FE_EA",
            "E3_U-TS",
        ]

       

        # Map each axes to a name and export
        for ax, name in zip(axs.flat, names):
            save_axis(ax, out, name, formats=("pdf",))  # both vector + raster

        plt.close()
