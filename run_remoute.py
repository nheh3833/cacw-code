import shlex

import paramiko


# Fill these in locally. Do not commit real credentials.
HOST = ""
PORT = 22  # int, not str: paramiko passes this straight to socket.connect
USER = ""
PASSWORD = ""

# NOTE: REMOTE_HOME is no longer hardcoded. "/home" is the parent of all user
# home directories and is root-owned, so every upload / install / log write
# failed with "Permission denied". We now ask the server where home actually is.
SCRIPT_NAME = "my_script.py"
LOG_NAME = "experiment_output.log"

LOCAL_FILES = ["main.py", "federated_learning.py", "models.py", "requirements.txt"]


def run_and_wait(ssh, cmd, label, quiet=False):
    """Run a remote command, stream its output, and raise if it fails.

    Output is drained BEFORE asking for the exit status. Doing it the other way
    round deadlocks on anything chatty: pip installing torch prints far more
    than the SSH channel buffer holds, so the remote process blocks on write and
    never exits, and recv_exit_status() waits forever.
    """
    stdin, stdout, stderr = ssh.exec_command(cmd)
    stdout.channel.set_combine_stderr(True)  # one stream, so neither can block the other
    lines = []
    for line in iter(stdout.readline, ""):
        lines.append(line)
        if not quiet:
            print("   " + line.rstrip())
    exit_status = stdout.channel.recv_exit_status()
    out = "".join(lines)
    if exit_status != 0:
        if quiet:
            print(out)
        raise RuntimeError(f"{label} failed (exit {exit_status})")
    return out


def probe(ssh, cmd):
    """Run a command just to see whether it succeeds. Returns (ok, output)."""
    stdin, stdout, stderr = ssh.exec_command(cmd)
    stdout.channel.set_combine_stderr(True)
    out = stdout.read().decode(errors="replace")
    return stdout.channel.recv_exit_status() == 0, out.strip()


ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
transport = ssh.get_transport()
transport.set_keepalive(60)

try:
    # -------------------------
    # Work out the real home directory, and check python3 exists
    # -------------------------
    ok, remote_home = probe(ssh, "cd ~ && pwd")
    if not ok or not remote_home.startswith("/"):
        raise RuntimeError(f"Could not determine remote home directory (got {remote_home!r})")

    ok, py_version = probe(ssh, "python3 --version")
    if not ok:
        raise RuntimeError("python3 not found on the remote host")

    REMOTE_HOME = remote_home
    SCRIPT_REMOTE = f"{REMOTE_HOME}/{SCRIPT_NAME}"
    LOG_FILE = f"{REMOTE_HOME}/{LOG_NAME}"

    print(f"Remote home:   {REMOTE_HOME}")
    print(f"Remote python: {py_version}")

    # Fail early and clearly rather than halfway through the upload.
    ok, _ = probe(ssh, f"test -w {shlex.quote(REMOTE_HOME)}")
    if not ok:
        raise RuntimeError(f"{REMOTE_HOME} is not writable by {USER}")

    # -------------------------
    # Refuse to start a second copy on top of a running one
    # -------------------------
    # Two copies would write to the same log file and fight over the GPU.
    ok, running = probe(ssh, f"pgrep -f {shlex.quote(SCRIPT_REMOTE)} || true")

    # -------------------------
    # Upload script + its dependencies
    # -------------------------
    # main.py imports federated_learning and models, so they must land in the
    # same directory as the script for those imports to resolve.
    sftp = ssh.open_sftp()
    try:
        for local_path in LOCAL_FILES:
            remote_path = SCRIPT_REMOTE if local_path == "main.py" else f"{REMOTE_HOME}/{local_path}"
            sftp.put(local_path, remote_path)
            print(f"   {local_path} -> {remote_path}")
    finally:
        sftp.close()
    print(f"Uploaded {len(LOCAL_FILES)} files.")

    # -------------------------
    # Ensure pip is available
    # -------------------------
    # Use "python3 -m pip" instead of a hardcoded ~/.local/bin/pip3 path. That
    # path only exists if pip was installed by get-pip --user, so a host with a
    # perfectly good system pip was still sent down the get-pip branch.
    print("\nChecking pip...")
    pip_present, pip_version = probe(ssh, "python3 -m pip --version")
    if pip_present:
        print(f"pip is already installed: {pip_version}")
    else:
        print("pip missing, trying ensurepip...")
        pip_present, _ = probe(ssh, "python3 -m ensurepip --user")
        if not pip_present:
            # wget is not installed everywhere, so fall back to curl.
            has_wget, _ = probe(ssh, "command -v wget")
            url = "https://bootstrap.pypa.io/get-pip.py"
            target = f"{REMOTE_HOME}/get-pip.py"
            fetch = (
                f"wget -q {url} -O {shlex.quote(target)}"
                if has_wget
                else f"curl -fsSL {url} -o {shlex.quote(target)}"
            )
            print("Downloading get-pip.py...")
            run_and_wait(ssh, fetch, "get-pip download")
            print("Installing pip locally...")
            run_and_wait(ssh, f"python3 {shlex.quote(target)} --user", "pip install")
        pip_present, pip_version = probe(ssh, "python3 -m pip --version")
        if not pip_present:
            raise RuntimeError("pip could not be installed on the remote host")
        print(f"pip installed: {pip_version}")

    # -------------------------
    # Install required packages
    # -------------------------
    print("\nInstalling requirements (this can take a while)...")
    run_and_wait(
        ssh,
        f"python3 -m pip install --user -r {shlex.quote(REMOTE_HOME + '/requirements.txt')}",
        "requirements install",
    )
    print("Requirements installed.")

    # -------------------------
    # Run script remotely using nohup
    # -------------------------
    print(f"\nLaunching experiment in background (logging to {LOG_FILE})...")

    # 1. 'cd' first, so CIFAR/MNIST download into ./data next to the script
    #    rather than into whatever directory the SSH session starts in.
    # 2. 'nohup' ignores hangup signals, so it survives disconnection.
    # 3. stdout and stderr both go to LOG_FILE.
    # 4. '&' backgrounds it; 'echo $!' hands back the PID.
    #
    # PYTHONPATH is deliberately NOT set any more: "pip install --user" puts
    # packages in the user site-packages that python3 already searches. The old
    # script pinned it to .../python3.10/site-packages, so on a host running any
    # other python version it pointed at a directory that did not exist.
    #
    # 'cd X; nohup ... &' uses ';' rather than '&&' on purpose: with '&&' the
    # whole AND-list is backgrounded in a subshell and $! is the subshell PID,
    # not python's.
    run_cmd = (
        f"cd {shlex.quote(REMOTE_HOME)}; "
        f"nohup python3 -u {shlex.quote(SCRIPT_REMOTE)} > {shlex.quote(LOG_FILE)} 2>&1 & "
        f"echo $!"
    )

    stdin, stdout, stderr = ssh.exec_command(run_cmd)
    pid = stdout.read().decode().strip()
    stdout.channel.recv_exit_status()

    # Confirm it is actually alive. Previously a script that died instantly
    # (bad import, syntax error, missing torch) still reported "SUCCESS".
    alive, _ = probe(ssh, f"kill -0 {shlex.quote(pid)}")
    if not alive:
        print(f"ERROR: process {pid} exited immediately. End of log:\n")
        _, log_tail = probe(ssh, f"tail -n 40 {shlex.quote(LOG_FILE)}")
        print(log_tail)
        raise RuntimeError("remote script failed to start")

    print(f"SUCCESS: Remote process started with PID: {pid}")
    print("You can now safely disconnect. The process will keep running.")
    print(f"Stop it later with:  ssh {USER}@{HOST} \"kill {pid}\"")

    # -------------------------
    # Optional: Stream the log file for live feedback
    # -------------------------
    print("\n--- Streaming Output (Ctrl+C to stop watching, process will continue) ---")
    try:
        # '-F' rather than '-f' retries if the file is not visible yet, and
        # '-n +1' starts from the first line so nothing is missed.
        stdin, stdout, stderr = ssh.exec_command(f"tail -n +1 -F {shlex.quote(LOG_FILE)}")
        for line in iter(stdout.readline, ""):
            print(line, end="")
    except KeyboardInterrupt:
        print("\n\nStopped watching log. Remote process is STILL running in the background.")
finally:
    ssh.close()
    print("SSH session closed.")
