from typing import Annotated as Atd

from typer import Option as Opt


def run_health(
    folder: Atd[str, Opt("-folder", help="Simulation folder")] = ".",
    log: Atd[str, Opt("-log", help="sim.log, relative to -folder")] = "sim.log",
    toml: Atd[str, Opt("-toml", help="toml with interfaces/workers, relative to -folder")] = "restart.toml",
    gap: Atd[float, Opt("-gap", help="Minutes without log activity that separate two jobs")] = 30.0,
    stall: Atd[float, Opt("-stall", help="Minutes without a write in workerN/ after which a worker with a move counts as hung")] = 10.0,
    last: Atd[int, Opt("-last", help="Only show the last N jobs (0 = all)")] = 10,
    out: Atd[str, Opt("-out", help="Also write the report to this file")] = None,
    since: Atd[str, Opt("-since", help="Ignore sim.log before this time ('YYYY-MM-DD HH:MM:SS'), e.g. the job start")] = None,
    strict: Atd[bool, Opt("-strict", help="Exit with code 3 if a worker failed in the last job")] = False,
):
    """Compact health check of an infretis run, one line per job.

    Per job (jobs are split at gaps > -gap minutes in sim.log): moves/h,
    acceptance, workers that got moves, and problems: hung workers (in the
    running job, a worker with a move but no write in its workerN/ folder for
    -stall minutes; long paths keep writing and are fine), ensembles without
    an accepted move, OOM/tracebacks in slurm-*.out and idle-killed
    propagators.
    With -strict, every problem except ensembles without ACC counts as a
    worker failure (exit code 3), e.g. for a watchdog inside the job.
    """
    import glob
    import os
    import re
    from datetime import datetime, timedelta

    import typer

    from inftools.misc.infinit_helper import read_toml

    ts_re = re.compile(r"(\d{4}\.\d\d\.\d\d \d\d:\d\d:\d\d)")
    ens_re = re.compile(r"in ensembles: ([\d ]+?) with paths")
    logf = os.path.join(folder, log)
    conf = read_toml(os.path.join(folder, toml))
    nens = len(conf["simulation"]["interfaces"])
    nwork = conf.get("runner", {}).get("workers", 0)
    name = lambda e: "[0-]" if e == 0 else f"[{e - 1}+]"

    # parse sim.log into jobs: submits and finished moves with timestamps
    since = since and datetime.strptime(since, "%Y-%m-%d %H:%M:%S")
    jobs, t, done = [], None, None
    with open(logf) as f:
        for line in f:
            m = ts_re.search(line)
            if m:
                tn = datetime.strptime(m.group(1), "%Y.%m.%d %H:%M:%S")
                if since and tn < since:
                    continue
                if t is None or tn - t > timedelta(minutes=gap):
                    jobs.append({"start": tn, "subs": {}, "moves": [], "open": {}, "nens": nens})
                t = jobs[-1]["end"] = tn
            if not jobs:
                continue
            j = jobs[-1]
            if " xx |\t0 1 " in line:
                # ensemble table header: number of ensembles at that time
                j["nens"] = sum(x.isdigit() for x in line.split("|")[1].split())
            elif "shooting " in line and "and worker:" in line:
                w = int(line.split("and worker:")[1])
                ens = [int(e) for e in ens_re.search(line).group(1).split()]
                j["subs"][w] = j["subs"].get(w, 0) + 1
                j["open"][w] = (t, ens)
            elif "shooted " in line:
                done = [[int(e) for e in ens_re.search(line).group(1).split()], None]
            elif done and "with status:" in line:
                done[1] = line.split("with status:")[1].split()[0]
            elif done and "total time:" in line:
                w = int(line.split("worker:")[1].split()[0])
                j["moves"].append((done[0], done[1]))
                j["open"].pop(w, None)
                done = None

    now = datetime.now()
    live = jobs and now - datetime.fromtimestamp(os.path.getmtime(logf)) < timedelta(minutes=gap)
    slurms = glob.glob(os.path.join(folder, "slurm-*.out"))
    lines = [f"run_health {os.path.abspath(folder)}  ({nens} ensembles, {nwork} workers)",
             "job start        hours moves  mv/h  acc%  workers status"]
    problems, failed = [], 0
    shown = jobs[-last:] if last else jobs
    for i, j in enumerate(shown, len(jobs) - len(shown) + 1):
        is_last = j is jobs[-1]
        # with -since the job is the current one: measure up to now, also if
        # the whole simulation stopped writing to sim.log
        end = now if (is_last and (live or since)) else j["end"]
        hours = max((end - j["start"]).total_seconds() / 3600, 1e-9)
        nmv = len(j["moves"])
        nacc = sum(s == "ACC" for _, s in j["moves"])
        probs, fails = [], 0

        # workers: hung = has a move but no write in its workerN/ folder for
        # -stall minutes (a running propagation writes every few seconds).
        # The folders are recreated each job, so only the running job counts.
        # Once the cycles of the current iteration are reached, infretis hands
        # out no new moves and workers wait for infinit, so no check then.
        cur = conf.get("current", {})
        iter_done = cur.get("cstep", 0) >= conf["simulation"].get("steps", float("inf"))
        if is_last and (live or since) and not iter_done:
            for w, (ts, ens) in sorted(j["open"].items()):
                wdir = os.path.join(folder, f"worker{w}")
                if not os.path.isdir(wdir):
                    continue
                mt = max([os.path.getmtime(wdir)] + [e.stat().st_mtime for e in os.scandir(wdir)])
                quiet = (now - datetime.fromtimestamp(mt)).total_seconds() / 60
                if quiet > stall:
                    probs.append(f"worker {w} hung in {','.join(map(name, ens))}: no write in worker{w}/ for {quiet:.0f} min")
        if nwork and len(j["subs"]) < nwork:
            probs.append(f"only {len(j['subs'])}/{nwork} workers got moves")
        fails += len(probs)

        # ensembles without accepted move, flagged if >=5 moves were expected
        # ([0-] only moves together with [0+], so short jobs are not flagged)
        jn = j["nens"]
        acc = [0] * jn
        for ens, s in j["moves"]:
            for e in ens:
                if s == "ACC" and e < jn:
                    acc[e] += 1
        exp = sum(len(e) for e, _ in j["moves"]) / jn
        noacc = [name(e) for e in range(jn) if acc[e] == 0]
        if noacc and exp >= 5:
            probs.append(f"no ACC in {','.join(noacc)} (expected ~{exp:.0f} each)")

        # slurm output written during this job
        for sf in slurms:
            mt = datetime.fromtimestamp(os.path.getmtime(sf))
            if not j["start"] <= mt <= end + timedelta(minutes=gap):
                continue
            txt = open(sf, errors="replace").read()
            hits = {k: txt.count(p) for k, p in (("OOM", "CUDA out of memory"),
                    ("Traceback", "Traceback"), ("stopped", "did not complete"))}
            hits = ", ".join(f"{k} x{n}" for k, n in hits.items() if n)
            if hits:
                probs.append(f"{os.path.basename(sf)}: {hits}")
                fails += 1

        # propagator logs (appended across jobs) only describe the latest job
        if is_last:
            dead = [os.path.basename(p).split("_")[0]
                    for p in sorted(glob.glob(os.path.join(folder, "worker*_propagator.log")))
                    if datetime.fromtimestamp(os.path.getmtime(p)) >= j["start"]
                    and "Idle time exceeded" in open(p, errors="replace").read()[-300:]]
            if dead:
                probs.append(f"propagator idle-killed: {','.join(dead)}")
                fails += 1

        status = ("running " if is_last and live else "") + (f"!! {len(probs)}" if probs else "OK")
        lines.append(f"{i:>3} {j['start']:%m-%d %H:%M} {hours:5.1f} {nmv:5d} {nmv / hours:5.1f} "
                     f"{100 * nacc / max(nmv, 1):5.1f} {len(j['subs']):>3}/{nwork:<3} {status}")
        lines += [f"      !! {p}" for p in probs]
        if is_last:
            problems, failed = probs, fails

    lines.append(f"STATUS: {'PROBLEMS' if problems else 'OK'} (last job)")
    print("\n".join(lines))
    if out:
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
    if strict and failed:
        raise typer.Exit(code=3)


if __name__ == "__main__":
    import typer

    app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})
    app.command()(run_health)
    app()
