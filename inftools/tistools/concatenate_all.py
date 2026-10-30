from tkinter import TOP
from typing import Annotated

import typer
from inftools.tistools.concatenate import trjcat
from inftools.tistools.trajtxt_conv import trajtxt_conv
from infretis.asyncrunner import aiorunner, future_list
import os
import shutil


def trjcat_all(
    inp: Annotated[
        str, typer.Option("-out", help="input efolder name")
    ] = 'load',
    traj: Annotated[
        str, typer.Option("-traj", help="name of the traj.txt file")
    ] = 'traj.txt',
    restart: Annotated[
        str, typer.Option("-restart", help="restart file to not overwrite those files")
    ] = 'restart.toml',
    start: Annotated[
        int, typer.Option("-start", help="first path to merge")
    ] = 0,
    end: Annotated[
        int, typer.Option("-end", help="last path to merge")
    ] = -1,
    workers: Annotated[
        int, typer.Option("-workers", help="number of workers to use")
    ] = 1,

    h5: Annotated[
        str, typer.Option("-h5", help="if the traj.txt files are in h5 format")
    ] = '',
    remove: Annotated[
        bool, typer.Option("-remove", help="remove folders after merging")
    ] = False,

    selection: Annotated[
        str,
        typer.Option("-selection", help="atom selection to write to output"),
    ] = "all",
    engine: Annotated[
        str,
        typer.Option("-engine", help="what engine to use for concatenation"),
    ] = "mda",
    traj_format: Annotated[
        str, typer.Option("-format", help="trajectory format, can be inferred")
    ] = None,
    topology: Annotated[
        str, typer.Option("-topology", help="a structure file with atom info")
    ] = None,
    centersel: Annotated[
        str,
        typer.Option(
            "-centersel",
            help="center this atom selction in the middle of the box",
        ),
    ] = "",
):
    """
    Concatenate the trajectories from an infretis simulation to single files and remove folder if empty.
    """
    def read_active(restart_path):
        import toml
        with open(restart_path, "r") as f:
            restart = toml.load(f)
            active = restart['current']["active"]
        return active
    

    def find_trajtxt(inp, start, end, active, traj_name):
        subfolders = []
        for entry in os.scandir(inp):
            if entry.is_dir():
                subfolders.append(int(entry.name))
        
        subfolders = [s for s in subfolders if s not in active]
        subfolders = sorted(subfolders)

        if end == -1:
            end = subfolders[-1]
        subfolders = [s for s in subfolders if s >= start and s <= end]
        traj_folders = []
        inp_prefix = f"{inp}/"
        for i in subfolders:
            traj_path = f"{inp_prefix}{i}/{traj_name}"
            if os.path.isfile(traj_path):
                traj_folders.append(i)
            else:
                print(f"Warning: no traj.txt found for path {i}, skipping.")
        return traj_folders
    

    def read_h5(trajtxt_path, start, end, active, traj_name):
        import h5py
        with h5py.File(trajtxt_path, "r") as f:
            trajtxt = []
            for key in f.keys():
                try:
                    k = int(key)
                except ValueError:
                    continue

                obj = f[key]
                if hasattr(obj, "keys") and traj_name not in obj.keys():
                    continue
                trajtxt.append(k)

        trajtxt = [k for k in trajtxt if k not in active]
        trajtxt = sorted(trajtxt)
        if not trajtxt:
            return []
        if end == -1:
            end = trajtxt[-1]
        trajtxt = [s for s in trajtxt if s >= start and s <= end]
        print(f"Found trajtxt for paths: {trajtxt}")
        return trajtxt

    def write_h5():
        import h5py
        import numpy as np
        with h5py.File(h5, "r") as f:
            dataset = f[str(traj_no)][traj]
            data = dataset[()]
            dtype = dataset.dtype
        traj_lines = []
        for i, line in enumerate(data):
            trajfile = f'../../{traj_no}.{traj_format}'.encode()
            traj_lines.append((int(line[0]), trajfile, int(i), int(1)))

        with h5py.File(h5, "a") as f:
            if traj in f[str(traj_no)].keys():
                del f[str(traj_no)][traj]
            f[str(traj_no)].create_dataset(
                traj,
                data=np.array(traj_lines, dtype=dtype),
                dtype=dtype,
            )
        # print(traj_lines)
        return traj_lines


    def write_trajtxt():
        trajtxt_path = f"{inp}/{traj_no}/{traj}"
        with open(trajtxt_path, "r") as f:
            lines = f.readlines()

        with open(trajtxt_path, "w") as f:
            step = 0
            for line in lines:
                if line.startswith("#") or line.strip() == "":
                    f.write(line)
                else:
                    parts = line.strip().split()
                    parts[1] = f'../../{traj_no}.{traj_format}'
                    parts[2] = str(step)
                    parts[3] = '1.0'
                    f.write(" ".join(parts) + "\n")
                    step += 1



    active = read_active(restart)
    print(f"active paths: {active}")

    if h5 == '':
        traj_nos = find_trajtxt(inp, start, end, active, traj)
    else:
        traj_nos = read_h5(h5, start, end, active, traj)

    
    n_loops = len(traj_nos)
    for traj_no in traj_nos:
        out = f"{inp}/{traj_no}.{traj_format}"
        print(f"Processing {out}")
        try:
            trjcat(
                out=out,
                traj=f"{inp}/{traj_no}/{traj}",
                traj_format=traj_format,
                topology=topology,
                selection=selection,
                engine=engine,
                centersel=centersel,
                h5 = h5
            )
        except Exception as e:
            print(e)
        

        if h5 == '' and os.path.exists(out):
            print(f"Writing {traj} for path {traj_no}")
            write_trajtxt() 
        elif h5 != '' and os.path.exists(out):
            write_h5()

        if remove and os.path.exists(out):
            accepted_dir = f"{inp}/{traj_no}/accepted"
            traj_dir = f"{inp}/{traj_no}"
            if os.path.isdir(accepted_dir):
                for entry in os.scandir(accepted_dir):
                    if entry.is_file() or entry.is_symlink():
                        os.remove(entry.path)
                    elif entry.is_dir():
                        shutil.rmtree(entry.path)
                
                os.rmdir(accepted_dir)
            if os.path.isdir(traj_dir) and not os.listdir(traj_dir):
                os.rmdir(traj_dir)

            
    
