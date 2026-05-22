from typing import Annotated as Atd

from inftools.tistools.simacc import sim_acc
from typer import Option as Opt
from pathlib import Path
import inftools.analysis.wham as wham_module
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from inftools.misc.infinit_helper import read_toml
from inftools.tistools.calc_simtime import calc_simtime
from inftools.misc.data_helper import data_reader
from inftools.tistools.simacc import sim_acc
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# mpl.use('TkAgg')


def report_output(
    folder: Atd[str, Opt("-folder", help="Output folder")] = "report",
    toml: Atd[str, Opt("-toml", help="The infretis .toml file")] = "infretis.toml",
    wham_toml: Atd[str, Opt("-wham_toml", help="The .toml file with the matching interfaces to data")] = "infretis.toml",
    data: Atd[str, Opt("-data", help="The infretis_data.txt file")] = "infretis_data.txt",
    nskip: Atd[int, Opt("-nskip", help="Number of lines to skip in infretis_data.txt")] = 100,
    lamres: Atd[float, Opt("-lamres", help="Resolution along the orderparameter, (intf1-intf0)/10)")] = None,
    nblock: Atd[int, Opt("-nblock", case_sensitive=False, help="Minimal number of blocks in the block-error analysis")] = 5,
    wham_folder: Atd[str, Opt("-wfolder", help="wham folder")] = "wham",
    load_folder: Atd[str, Opt("-lfolder", help="load folder")] = "load",
    simlog: Atd[str, Opt("-slog", help="simulation log file")] = "sim.log",
    fener: Atd[bool, Opt("-fener", help="If set, calculate the conditional free energy. See Wham_")] = False,
    nbx: Atd[int, Opt("-nbx", help="Number of bins in x-direction when calculating the free-energy")] = 100,
    nby: Atd[int, Opt("-nby", help="Same as -nbx but in y-direction")] = None,
    minx: Atd[float, Opt("-minx", help="Minimum orderparameter value in the x-direction when calculating FE")] = 0.0,
    maxx: Atd[float, Opt("-maxx", help="Maximum orderparameter value in the x-direction when calculating FE")] = 100.0,
    miny: Atd[float, Opt("-miny", help="Same as -minx but in y-direction")] = None,
    maxy: Atd[float, Opt("-maxy", help="Same as -maxx but in y-direction")] = None,
    xcol: Atd[int, Opt("-xcol", help="What column in order.txt to use as x-value when calculating FE")] = 1,
    ycol: Atd[int, Opt("-ycol", help="Same as -xcol but for y-value")] = None,
    h5: Atd[str, Opt("-h5", help="The h5 file if order.txt are stored in an h5 file. Requires the h5py package.")] = None,
    detailed: Atd[bool, Opt("-detailed", help="Only makes sense if no combo is used. If set, also creates the detailed report with the Pcross of each ensemble and the path length distribution.")] = False,
    recalc: Atd[bool, Opt("-recalc", help="If set, recalculates the wham analysis.")] = False,
    nVE: Atd[bool, Opt("-nVE", help="If set, calculates Activation Energy and Free energy.")] = False,
    kb: Atd[float, Opt("-kb", help = "kB, in the correct energy unit of entries of energy.txt default: 8.617333262e-5 eV/K")] = 8.617333262*10**(-5) ,
    load: Atd[ str, Opt("-load", help = "directory containing the paths")] = "load",
    Nbins: Atd[int, Opt("-Nbins", help = "number of bin along OP")] = 100,
    ):
    """Creates the figures from the report notebook and sums up the most important measures in txt format.

    Run this command in the root infretis folder containing the data and log files."""
    if nVE:
        fener = True
    if not Path(wham_folder).exists() or recalc:
        wham_module.wham(
            toml=wham_toml,
            data=data,
            nskip=nskip,
            lamres=lamres,
            nblock=nblock,
            folder=wham_folder,
            fener=fener,
            nbx=nbx, nby=nby, minx=minx, maxx=maxx, miny=miny, maxy=maxy, xcol=xcol, ycol=ycol, h5=h5
        )
    folder = Path(folder)
    folder.mkdir(exist_ok=True)


    
    # Helper: save a single Axes as its own PDF/PNG using its tight bbox
    def save_axis(ax, basepath: Path, stem: str, formats=("pdf",)):
        # tight bbox in display units, expanded slightly for labels/ticks
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.05)
        # convert to inches for bbox_inches
        bbox_in = bbox.transformed(fig.dpi_scale_trans.inverted())
        for ext in formats:
            fig.savefig(basepath / f"{stem}.{ext}", bbox_inches=bbox_in, dpi=300)
    

    # path to infretis data files/folders, to be changed if needed
    fpaths = {
        "toml": toml,
        "wham_toml": wham_toml,
        "data": data,
        "log": simlog,
        "wham": Path(wham_folder),
    }

    if h5 is not None:
        import h5py
        fpaths["h5"] = h5
    else:
        fpaths["load"] = Path(load_folder)

    
    # check that they exist
    valid = [os.path.exists(value) for key, value in fpaths.items()]
    assert np.all(valid)

    if h5 is not None:
        import h5py

    print("\nreport lies in ", os.getcwd())
    # Load the data
    print("\nSimulation data:")


    # print standard information
    toml = read_toml(fpaths["toml"])

    md_dt = toml["engine"]["timestep"]

    # guessing units: 
    ### Usually for the following engines:
    # Gromacs: md_dt = 0.002, units = "ps"
    # CP2K:    md_dt = 0.5,   units = "fs"
    if md_dt < 0.01:
        units = "ps"
    elif md_dt < 10:
        units = "fs"
    else:
        print("Time step is larger than 10, assuming internal units.")
        units = "internal"




    wham_toml = read_toml(fpaths["wham_toml"])

    intfs = wham_toml["simulation"]["interfaces"]
    print("Number of ensembles:", len(intfs), f", from 000 to {len(intfs)-1:03.0f}")
    subcycles = toml["engine"]["subcycles"]
    if units != "internal":
        if units == 'fs':
            md_dt *= 1e-15
        elif units == 'ps':
            md_dt *= 1e-12

        units = 's'
    unitconv = subcycles*md_dt
    

    data = data_reader(fpaths["data"])
    # count reactive
    rcnt = 0
    last_path = -1    
    for path in data:
        if int(path['pn']) > last_path:
            last_path = int(path['pn'])
        if float(path["max_op"]) > intfs[-1]:
            rcnt += 1
    print(f"Sampled paths: {len(data)}, reactive: {rcnt}")

    simtime, restarts = calc_simtime(fpaths["log"])
    print(f"Simulation wall time: {simtime:.04f} days, restarts: {restarts}\n")


    # Flux, Pcross, Rate from inft wham
    runav_flux = np.loadtxt(fpaths["wham"] / "runav_flux.txt")
    runav_pcro = np.loadtxt(fpaths["wham"] / "runav_Pcross.txt")
    runav_rate = np.loadtxt(fpaths["wham"] / "runav_rate.txt")
    pcro = np.loadtxt(fpaths["wham"] / "Pcross.txt")
    err_flux = np.loadtxt(fpaths["wham"] / "errFLUX.txt")
    err_pcro = np.loadtxt(fpaths["wham"] / "errPtot.txt")
    err_rate = np.loadtxt(fpaths["wham"] / "errRATE.txt")
    # Figure 4: Path length distribution
    plens = np.loadtxt(fpaths["wham"] / "pathlengths.txt")

    print(f"pcross {runav_pcro[-1, -1]:.04e} +- {err_pcro[-1, -1]*100:.4f}%")
    print(f"flux {runav_flux[-1, -1]:.04e} [1/{units}] +- {err_flux[-1, -1]*100:.4f}%", end=" ")
    print(f"time step of {md_dt} {units}")
    print(f"rate is estimated to be {runav_rate[-1, -1]:.4e} [1/{units}] +- {err_rate[-1, -1]*100:.4f}%")
    print("\nPath lengths per ensemble:")
    for i, plen in zip(plens[:, 0], plens[:, 1]):
        print(f"Ensemble {i}: {plen*subcycles*md_dt:.4e} {units}")
    
    
    # to be... is the error
    # Write summary to text file
    with open(folder / "summary.txt", "w") as f:
        f.write("measure\tvalue\terror (%)\n")
        f.write(f"pcross\t{runav_pcro[-1, -1]:.04e}\t{err_pcro[-1, -1]*100:.4f}\n")
        f.write(f"flux (1/{units})\t{runav_flux[-1, -1]:.04e}\t{err_flux[-1, -1]*100:.4f}\n")
        f.write(f"rate (1/{units})\t{runav_rate[-1, -1]:.4e}\t{err_rate[-1, -1]*100:.4f}\n")
        f.write(f"time step\t{md_dt}\t{units}\n")
        f.write(f"Sampled paths: {len(data)}\n")
        f.write(f"reactive: {rcnt}\n")
        f.write("-------------------------\n")
        f.write(f"Ensemble\t{units}\n")

        for i, plen in zip(plens[:, 0], plens[:, 1]):
            f.write(f"{int(i)}\t{plen*subcycles*md_dt:.4e}\n")

    with PdfPages(folder / "report.pdf") as pdf:
        # All figures on one PDF page
        fig, axs = plt.subplots(3, 3, figsize=(12, 11))
        fig.text(0.01, 0.99, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ha='left', va='top', fontsize=8)
        
        
        # fig.text(0.5, 0.98, f"pcross {runav_pcro[-1, -1]:.04e} +- {err_pcro[-1, -1]*100:.4f}% | flux {runav_flux[-1, -1]:.04e} [1/{units}] +- {err_flux[-1, -1]*100:.4f}% | rate {runav_rate[-1, -1]:.4e} [1/{units}] +- {err_rate[-1, -1]*100:.4f}%", ha='center', fontsize=9, va='top')
        

        # Figure 1a: Running Average Rate
        axs[0, 0].plot(runav_rate[:, 0], runav_rate[:, -1] / unitconv, label=f"Rate [1/{units}]", color="C0")
        axs[0, 0].set_yscale("log")
        axs[0, 0].set_title(r"Running Average Rate")
        axs[0, 0].set_xlabel("Accepted Paths")
        axs[0, 0].legend(fontsize=8)

        # Figure 1b: Running Average Pcross
        axs[0, 1].plot(runav_pcro[:, 0], runav_pcro[:, -1], label="Pcross", color="C1")
        axs[0, 1].set_yscale("log")
        axs[0, 1].set_title(r"Running Average Pcross")
        axs[0, 1].set_xlabel("Accepted Paths")
        axs[0, 1].legend(fontsize=8)

        # Figure 1c: Running Average Flux
        axs[0, 2].plot(runav_flux[:, 0], runav_flux[:, -1] / unitconv, label=f"Flux [1/{units}]", color="C2")
        axs[0, 2].set_yscale("log")
        axs[0, 2].set_title(r"Running Average Flux")
        axs[0, 2].set_xlabel("Accepted Paths")
        axs[0, 2].legend(fontsize=8)

        # Figure 2: Pcross
        for intf in intfs:
            axs[1, 0].axvline(intf, color="k", alpha=0.2, zorder=1)
        axs[1, 0].plot(pcro[:, 0], pcro[:, -1], zorder=2)
        axs[1, 0].set_yscale("log")
        axs[1, 0].set_title(r"Conditional Crossing Probability")
        axs[1, 0].set_xlabel("Order Parameter")

        # Figure 3: Block Error
        axs[1, 1].axhline(err_rate[-1, -1], alpha=0.2, color="C0")
        axs[1, 1].plot(err_rate[:, 0], err_rate[:, -1], label="Rate", color="C0")
        axs[1, 1].axhline(err_pcro[-1, -1], alpha=0.2, color="C1")
        axs[1, 1].plot(err_pcro[:, 0], err_pcro[:, -1], label="Pcross", color="C1")
        axs[1, 1].axhline(err_flux[-1, -1], alpha=0.2, color="C2")
        axs[1, 1].plot(err_flux[:, 0], err_flux[:, -1], label="Flux", color="C2")
        axs[1, 1].set_yscale("log")
        axs[1, 1].set_title(r"Running Block Error")
        axs[1, 1].set_xlabel("Accepted Paths")
        axs[1, 1].legend(fontsize=8)
        # Figure 4: Path length distribution
        axs[1, 2].scatter(plens[:, 0], plens[:, 1]*subcycles*md_dt)
        axs[1, 2].set_xlabel("Ensembles")
        axs[1, 2].set_xticks(plens[:, 0])

        axs[1, 2].set_ylabel(f"Path length [{units}]")
        if  h5 is not None:
            h5file = h5py.File(h5, 'r')
        for i in range(last_path-99, last_path+1):
            xo = None
            xe = None
            source_i = i
            if h5 is not None:
                while source_i >= 0:
                    key = str(source_i)
                    node = h5file.get(key)
                    if isinstance(node, h5py.Group) and "order.txt" in node and "energy.txt" in node:
                        xo = node["order.txt"][:]
                        xe = node["energy.txt"][:]
                        break
                    source_i -= 1
            else:
                while source_i >= 0:
                    order_file = fpaths["load"] / str(source_i) / "order.txt"
                    energy_file = fpaths["load"] / str(source_i) / "energy.txt"
                    if order_file.exists() and energy_file.exists():
                        xo = np.loadtxt(order_file)
                        xe = np.loadtxt(energy_file)
                        break
                    source_i -= 1
            if xo is None or xe is None:
                print(f"Skipping path {i}: no valid previous path found.")
                continue
            if source_i != i:
                print(f"Path {i} missing, using previous valid path {source_i}.")
            axs[2, 0].plot(xo[:, 0]*subcycles*md_dt, xo[:, 1])
            axs[2, 1].plot(xo[:, 0]*subcycles*md_dt, xe[:, 1])
            axs[2, 2].plot(xo[:, 0]*subcycles*md_dt, xe[:, 2])
        if h5 is not None:
            h5file.close()

        axs[2, 0].set_title("OP last 100")
        axs[2, 0].set_xlabel(f"Path length ({units})")
        axs[2, 0].set_ylabel("Order parameter")

        axs[2, 1].set_title("E_pot last 100")
        axs[2, 1].set_xlabel(f"Path length ({units})")
        axs[2, 1].set_ylabel("E_Pot (system units)")

        axs[2, 2].set_title("E_kin last 100")
        axs[2, 2].set_xlabel(f"Path length ({units})")
        axs[2, 2].set_ylabel("E_Pot (system units)")

        
        plt.tight_layout()
        pdf.savefig(fig)
        
        fig.canvas.draw()

        names = [
            "1a_running_average_rate",
            "1b_running_average_pcross",
            "1c_running_average_flux",
            "2_conditional_crossing_probability",
            "3_running_block_error",
            "4_path_length_distribution",
            "5_path_length_last100",
            "6_e_pot_last100",
            "7_e_kin_last100",
        ]

       

        # Map each axes to a name and export
        for ax, name in zip(axs.flat, names):
            save_axis(ax, folder, name, formats=("pdf",))  # both vector + raster

        plt.close()


    if nVE:
        from inftools.tistools.e_act import activation_energy
        activation_energy(
                kb = kb,
                wham = Path(wham_folder),
                load = Path(load_folder),
                toml = fpaths["toml"],
                Nbins = Nbins,
                out = folder,
                h5 = h5,
                       )





    if detailed:
        ploc_pcros = np.loadtxt(fpaths["wham"] / "ploc_WHAM.txt")
        ploc_runav = np.loadtxt(fpaths["wham"] / "runav_ploc.txt")
        ploc_err   = np.loadtxt(fpaths["wham"] / "errploc.txt")

        # for i, ens  in range(1, len(intfs)):
        for i, ens in enumerate([f"[{i}^+]" for i in range(len(intfs)-1)], start=1):
            fig, axs = plt.subplots(1, 3, figsize=(16, 3))
            axs[0].plot(ploc_pcros[:, 0], ploc_pcros[:, i]/np.max(ploc_pcros[:, i]), zorder=5)
            axs[0].set_ylim([0, 1])
            axs[0].axvline(intfs[0], color="k")
            axs[0].axvline(intfs[-1], color="k")
            axs[0].axvline(intfs[i], color="k", alpha=0.4, ls="--")
            axs[0].set_xlabel(r"Order parameter ($\lambda$)")
            axs[0].set_title(f"{ens} Crossing probability")

            axs[1].plot(ploc_runav[:, 0], ploc_runav[:, i], zorder=5)
            axs[1].axhline(ploc_runav[-1, i], color="k", alpha=0.4, ls="--")
            axs[1].set_xlabel(r"Uniquely sampled path")
            axs[1].set_title(f"{ens} Running estimate")
            
            axs[2].plot(ploc_err[:, 0], ploc_err[:, i], zorder=5)
            half = int(len(ploc_err[:, i])/2)
            axs[2].axhline(np.average(ploc_err[half:, i]), color="k", alpha=0.4, ls="--")
            axs[2].set_xlabel(r"Block length")
            axs[2].set_title(f"{ens} Estimated error")
            plt.savefig(folder / f"Pcross_ens{i:03.0f}.pdf")
            plt.close() 

        paths = data_reader(fpaths["data"])
        ens_cnt = [0 for _ in range(len(intfs))]
        print(ens_cnt)
        for path in paths:
            for key in path["cols"]:
                # ens_cnt[key] += 1
                ens_cnt[key] += 1
        plt.scatter(list(range(len(intfs))), ens_cnt)
        plt.ylim([0, None])
        plt.xlabel("Ensemble")
        plt.ylabel("Number of uniquely sampled paths")


        print(ploc_runav.shape[1])
        print(len(ploc_runav[-1, :]))
        plt.scatter(np.arange(ploc_runav.shape[1])[1:], ploc_runav[-1, 1:])
        plt.xlabel("Ensemble")
        plt.ylabel("Ensemble Pcross")
        plt.axhline(0, color="k")
        plt.axhline(1, color="k")
        plt.savefig(folder / f"Pcross_ens_scatter.pdf")
        plt.close()


        sim_acc(fpaths["log"], fpaths["toml"])
        plens = np.loadtxt(fpaths["wham"] / "pathlengths.txt")
        plt.scatter(plens[:, 0], plens[:, 1]*subcycles*md_dt)
        plt.xlabel("Ensembles")
        plt.ylabel(f"Path length [{units}]")
        plt.savefig(folder / f"pathlengths.pdf")
        plt.close()

