from os import path
from typing import Annotated, List
import typer
from ase.io import read, write





def calc_order_MD_ase(
    trajectories: Annotated[
        str,
        typer.Option(
            "-trajectories", "-trajectory", "-t",
            help="trajectories in ase readable format, e.g. .traj or .xyz",
        ),
    ] = "*.traj" ,
    toml: Annotated[str, typer.Option("-toml")] = "infretis.toml",
    out: Annotated[str, typer.Option("-out", help="The output of the analysis")] = "order_rec",
):
    """
    Recalculate the order parameter from a sampled infretis path
    using ASE .traj files instead of MDAnalysis-readable trajectories.
    """

    import os
    import glob
    import numpy as np
    import tomli
    from ase.io.trajectory import Trajectory
    from infretis.classes.orderparameter import create_orderparameter
    from infretis.classes.system import System

    # --- Load configuration ---
    with open(toml, "rb") as toml_file:
        toml_dict = tomli.load(toml_file)
    orderparameter = create_orderparameter(toml_dict)

    # --- Load trajectory index ---
    patterns = [trajectories] if isinstance(trajectories, str) else trajectories
    files = [filename for pattern in patterns for filename in glob.glob(pattern)]
    print(files)
    # --- Load ASE trajectories ---
    unis = {}
    for f in files:
        unis[f] = Trajectory(f)
        path = os.path.dirname(f)
        outf = out + "_" + os.path.basename(f).split('.')[0]
        # --- Prepare output file ---
        out_file = os.path.join(path, outf + ".txt")
        for i in range(1, 1000):
            if not os.path.isfile(out_file):
                break
            out_file = os.path.join(path, f"{outf}_{i}.txt")


        # --- Recalculate order parameters ---
        with open(out_file, "w") as writefile:
            writefile.write("# step\torder\n")
            for step, atoms in enumerate(read(f, index=":")):
                atoms.set_pbc(True)

                system = System()
                system.config = (f, step)
                system.pos = atoms.get_positions()
                system.box = atoms.get_cell().lengths()

                op = orderparameter.calculate(system)
                line = f"{step} " + "\t".join(map(str, op)) + "\n"
                writefile.write(line)

        print(f"[ INFO ] Orderparameter values written to {out_file}\n")



def get_lambda_A_from_MD(
    ordertxt: Annotated[
        str,
        typer.Option(
            "-ordertxt", "-order", "-ot",
            help="trajectories in ase readable format, e.g. .traj or .xyz",
        ),
    ] = "*.traj" ,
    toml: Annotated[str, typer.Option("-toml")] = "infretis.toml",
    out: Annotated[str, typer.Option("-out", help="The output of the analysis")] = "lambda_A",
    skip: Annotated[
        int,
        typer.Option(
            "-skip",
            help="Number of initial steps to skip when calculating lambda_A; 0 uses all steps",
        ),
    ] = 0,
):
    """
    Recalculate the order parameter from a sampled infretis path
    using ASE .traj files instead of MDAnalysis-readable trajectories.
    """

    import os
    import glob
    import numpy as np
    import tomli

    # --- Load configuration ---
    with open(toml, "rb") as toml_file:
        toml_dict = tomli.load(toml_file)
    interfaces = toml_dict["simulation"]["interfaces"]
    lambda_B = interfaces[-1]

    # --- Load trajectory index ---
    patterns = [ordertxt] if isinstance(ordertxt, str) else ordertxt
    files = [filename for pattern in patterns for filename in glob.glob(pattern)]
    print(files)
    # --- Load ASE trajectories ---
    unis = {}
    for f in files:
        path = os.path.dirname(f)
        outf = out + "_" + os.path.basename(f).split('.')[0]
        # --- Prepare output file ---
        out_file = os.path.join(path, outf + ".toml")
        plot_file = os.path.join(path, outf + ".png")
        steps = []
        values = []
        with open(f) as order_file:
            for line in order_file:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                columns = line.split()
                if len(columns) < 2:
                    continue
                try:
                    step = float(columns[0])
                    value = float(columns[1])
                except ValueError:
                    continue
                steps.append(step)
                values.append(value)

        if not values:
            raise ValueError(f"No order-parameter values found in {f}")

        selected_values = values[skip:] if skip > 0 else values
        if not selected_values:
            raise ValueError(f"No order-parameter values found in the first {skip} steps of {f}")

        lambda_A = round(float(np.percentile(selected_values, 80)), 2)
        with open(out_file, "w") as writefile:
            writefile.write("[simulation]\n")
            writefile.write("interfaces = [\n")
            writefile.write(f"    {lambda_A:.3f},\n")
            writefile.write(f"    {lambda_B},\n")
            writefile.write("]\n")

        import matplotlib.pyplot as plt
        plt.figure()
        plot_steps = steps[skip:] if skip > 0 else steps
        plot_values = values[skip:] if skip > 0 else values
        plt.plot(steps, values, label="order parameter")
        if plot_steps:
            plt.plot([plot_steps[0], plot_steps[-1]], [lambda_A, lambda_A], color="tab:orange", linestyle="--", label="lambda A")
        else:
            plt.axhline(lambda_A, color="tab:orange", linestyle="--", label="lambda A")
        plt.axhline(lambda_B, color="tab:red", linestyle="--", label="lambda B")
        plt.xlabel("step")
        plt.ylabel("order parameter")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()

        print(f"[ INFO ] lambda_A ({lambda_A}) written to {out_file}; plot written to {plot_file}\n")


def get_load_folder_from_MD(
    trajectories: Annotated[
        str,
        typer.Option(
            "-trajectories", "-trajectory", "-t",
            help="Trajectories in an ASE-readable format, e.g. .traj or .xyz",
        ),
    ] = "*.traj",
    toml: Annotated[str, typer.Option("-toml")] = "infretis.toml",
    skip: Annotated[
        int,
        typer.Option(
            "-skip",
            help="Number of initial steps to skip when calculating lambda_A",
        ),
    ] = 0,
    out_traj_formate : Annotated[
            str,
            typer.Option(
                "-out_traj_formate",
                help="ase trajectory file formate for ouput files",
            ),
        ] = "traj",
    out : Annotated[
            str,
            typer.Option(
                "-out",
                help="Prefix for generated output files",
            ),
        ] = ""
    ,
    no_traj_update: Annotated[
        bool,
        typer.Option(
            "-no-traj-update",
            help="Update text and TOML files without replacing accepted trajectories",
        ),
    ] = False,
):
    """Create the lambda-A input from ASE trajectories."""
    import glob
    import os
    import numpy as np
    import tomli
    import tomli_w
    import shutil


    def interfaces_have_samples(interfaces, plus_steps):
        """Return whether every interface interval contains a sampled value."""
        values = np.asarray(plus_steps)[:, 1]
        return all(
            np.any((values > left) & (values < right))
            for left, right in zip(interfaces[:-1], interfaces[1:])
        )

    def readjust_interfaces(interfaces, plus_steps, lamda_A, lamda_B):
        """Place internal interfaces between sampled order-parameter values."""
        values = np.unique(np.asarray(plus_steps)[:, 1])
        values = values[(values > lamda_A) & (values < lamda_B)]
        n_internal = len(interfaces) - 2
        if len(values) < n_internal + 1:
            return interfaces

        selected = values[
            np.linspace(0, len(values) - 1, n_internal + 1).astype(int)
        ]
        internal = np.round((selected[:-1] + selected[1:]) / 2, 2)
        return np.round(np.r_[lamda_A, internal, lamda_B], 2)

    def distribute_interface_steps(
        plus_steps,
        n_interfaces,
        lamda_A,
        lamda_B,
        interface_cap,
        growth_rate=1.7,
    ):
        """Return interface values and steps distributed by the second column."""
        plus_steps = np.asarray(plus_steps)
        if plus_steps.size == 0:
            raise ValueError("No plus steps available for interface distribution")
        elif plus_steps.ndim == 2 and plus_steps.shape[1] >= 2:
            order = np.argsort(plus_steps[:, 1])
            plus_steps = plus_steps[order]
            values = plus_steps[:, 1]
        else:
            raise ValueError("plus_steps must have two columns")

        plus_steps = plus_steps[values <= interface_cap]
        values = plus_steps[:, 1]
        if values.size == 0:
            raise ValueError("No plus steps available below interface_cap")

        highest_lambda_i = np.round(np.max(values)-0.001, 2)

        plus_steps = plus_steps[values != np.max(values)]
        values = plus_steps[:, 1]
        if values.size == 0:
            raise ValueError("No plus steps available below interface_cap")
        
        n_bins = n_interfaces - 2
        n_plus = len(values)
        if n_interfaces < 2:
            raise ValueError(f"n_interfaces = {n_interfaces} is too small")
        if n_plus < n_interfaces:
            raise ValueError(
                f"Not enough plus_steps ({n_plus}) for {n_interfaces} interfaces"
            )

        weights = growth_rate ** np.arange(n_bins)[::-1]
        weights /= weights.sum()
        bin_sizes = np.maximum(np.floor(weights * (n_plus - 1)).astype(int), 1)
        diff = n_plus - 1 - bin_sizes.sum()
        while diff:
            if diff > 0:
                bin_sizes[np.argmax(bin_sizes)] += 1
                diff -= 1
            else:
                idx = np.argmax(bin_sizes)
                if bin_sizes[idx] <= 1:
                    break
                bin_sizes[idx] -= 1
                diff += 1

        # Include both endpoints, then enforce the configured basin boundaries.
        indices = np.r_[0, np.cumsum(bin_sizes)[:-1], n_plus - 1]
        interfaces = np.round(values[indices.astype(int)], 2)
        interfaces[0] = lamda_A
        # interfaces[-1] = highest_lambda_i
        interfaces = np.append(interfaces, lamda_B)

        if not interfaces_have_samples(interfaces, plus_steps):
            interfaces = readjust_interfaces(
                interfaces, plus_steps, lamda_A, lamda_B
            )

        return interfaces

    

    # Generate the order-parameter input first when it was not supplied.

    patterns = [trajectories] if isinstance(trajectories, str) else trajectories
    files = [filename for pattern in patterns for filename in glob.glob(pattern)]

    for f in sorted(files):
        print(f"[ INFO ] Processing trajectory: {f}")
        if out is None or out == "":
            out = f"order_rec"
        ordertxt = f"{out}_{path.basename(f).split('.')[0]}.txt"
        if ordertxt is None or not glob.glob(ordertxt):
            calc_order_MD_ase(trajectories=trajectories, toml=toml, out=out)
        interface_toml = f"lambda_A_{out}_{path.basename(f).split('.')[0]}.toml"
        if interface_toml is None or not glob.glob(interface_toml):
            get_lambda_A_from_MD(
                ordertxt=ordertxt,
                toml=toml,
                skip=skip,
            )

        load_folder = f"load_{out}_{path.basename(f).split('.')[0]}" if out else "load_" + path.basename(f).split('.')[0]
        os.makedirs(load_folder, exist_ok=True)

        with open(interface_toml, "rb") as tf:
            old_interfaces = tomli.load(tf)["simulation"]["interfaces"]
            lambda_A, lambda_B = old_interfaces[0], old_interfaces[-1]
            
        with open(toml, "rb") as toml_file:
            toml_dict = tomli.load(toml_file)
            n_interfaces = 2 * int(toml_dict["runner"]["workers"])
            interface_cap = toml_dict["simulation"]["tis_set"]["interface_cap"]
        print(f"[ INFO ] n_interfaces = {n_interfaces}")
        order_data = np.loadtxt(ordertxt)
        print(lambda_A)
        plus_completer = order_data[np.where(order_data[:, 1] < lambda_A)[0][0], :]
        minus_completer = order_data[np.where(order_data[:, 1] > lambda_A)[0][0], :]

        # All rows where the second column is greater than lambda_A.
        minus_steps = order_data[order_data[:, 1] < lambda_A]
        minus_steps = minus_steps[-100:]

        plus_steps = order_data[order_data[:, 1] > lambda_A]

        print(order_data)
        print(plus_steps.shape)

        growth_rate = 1.7
        for _ in range(10):
            interfaces = distribute_interface_steps(
                plus_steps,
                n_interfaces,
                lambda_A,
                lambda_B,
                interface_cap,
                growth_rate=growth_rate,
            )
            if interfaces_have_samples(interfaces, plus_steps):
                break
            growth_rate = max(1.05, growth_rate - 0.1)
        else:
            interfaces = readjust_interfaces(
                interfaces, plus_steps, lambda_A, lambda_B
            )

        # Load the configuration with tomli and update its interfaces.
        with open(toml, "rb") as toml_file:
            updated_toml_dict = tomli.load(toml_file)
        updated_toml_dict["simulation"]["interfaces"] = [
            float(interface) for interface in interfaces
        ]
        updated_toml_dict["simulation"]["shooting_moves"] = [
            "sh",
            "sh",
            *["wf"] * (n_interfaces - 2),
        ]
        updated_toml_dict["runner"]["wmdrun"] = [

                    *["0"] * (n_interfaces),
                ]

        with open(os.path.join(load_folder, "infretis.toml_bkp"), "wb") as backup_file:
            tomli_w.dump(updated_toml_dict, backup_file)
        

        print(interfaces)
        def create_folder_content(steps, folder, traj_name, update_traj=True):
            """Write an order.txt file containing numbered values from steps,
            and write the corresponding trajectory frames using ase."""
            import os
            from ase.io import read as ase_read, write as ase_write

            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "order.txt"), "w") as order_file:
                order_file.write("# step\torder\n")
                for index, step in enumerate(steps):
                    order_file.write(f"{index}\t{np.round(step[1], 2)}\n")

            with open(os.path.join(folder, "traj.txt"), "w") as tfile:
                tfile.write("#\n")
                tfile.write("#     Step              Filename       index    vel\n")
                for index, step in enumerate(steps):
                    tfile.write(
                        f"\t\t{index}\t{str(traj_name).split('/')[-1]}\t{index}\t1\n"
                    )

            if update_traj:
                frames = [ase_read(f, index=int(step[0])) for step in steps]
                ase_write(os.path.join(load_folder, traj_name), frames)
    
        for i, interface in enumerate(interfaces):
            
            if i == 0:
                interface_folder = os.path.join(load_folder, f"{i}/accepted")
                os.makedirs(interface_folder, exist_ok=True)
                traj_name = f"{i}/accepted/minus_steps." + out_traj_formate
                steps = np.concatenate(
                    (
                        np.array([minus_completer]),
                        minus_steps,
                        np.array([minus_completer]),
                    )
                )
                create_folder_content(
                    steps,
                    os.path.join(load_folder, str(i)),
                    traj_name,
                    update_traj=not no_traj_update,
                )

                
            elif i == 1:
                steps = np.concatenate(
                    (
                        np.array([plus_completer]),
                        plus_steps,
                        np.array([plus_completer]),
                    )
                )
                interface_folder = os.path.join(load_folder, f"{i}/accepted")
                os.makedirs(interface_folder, exist_ok=True)
                traj_name = f"{i}/accepted/plus_steps." + out_traj_formate
                create_folder_content(
                    steps,
                    os.path.join(load_folder, str(i)),
                    traj_name,
                    update_traj=not no_traj_update,
                )
            else:
                if no_traj_update:
                    steps = np.concatenate(
                        (
                            np.array([plus_completer]),
                            plus_steps,
                            np.array([plus_completer]),
                        )
                    )
                    create_folder_content(
                        steps,
                        os.path.join(load_folder, str(i)),
                        f"{i}/accepted/plus_steps.{out_traj_formate}",
                        update_traj=False,
                    )
                else:
                    shutil.copytree(
                        os.path.join(load_folder, "1"),
                        os.path.join(load_folder, str(i)),
                        dirs_exist_ok=True,
                    )


    
