# Viewer for the experiment launched by run_remoute.py.
# Connection settings and the log location are kept in sync with that script.
import os
import shlex

import paramiko

# Same environment variables as run_remoute.py, so both scripts stay in sync:
#   PowerShell:  $env:TEZA_HOST="1.2.3.4"; $env:TEZA_USER="me"; $env:TEZA_PASSWORD="secret"
#   bash:        export TEZA_HOST=1.2.3.4 TEZA_USER=me TEZA_PASSWORD=secret
HOST = os.environ.get("TEZA_HOST", "add")
PORT = int(os.environ.get("TEZA_PORT", 22))
USER = os.environ.get("TEZA_USER", "add")
PASSWORD = os.environ.get("TEZA_PASSWORD", "add")

if not HOST:
    raise SystemExit("TEZA_HOST is not set and HOST is empty. Set it before running.")

# Must match run_remoute.py. REMOTE_HOME is not hardcoded: "/home" is the
# root-owned parent of all user home directories, so the log actually lives in
# the real home directory, which we ask the server for.
SCRIPT_NAME = "my_script.py"
LOG_NAME = "experiment_output.log"


def probe(ssh, cmd):
    """Run a command just to see whether it succeeds. Returns (ok, output)."""
    stdin, stdout, stderr = ssh.exec_command(cmd)
    stdout.channel.set_combine_stderr(True)
    out = stdout.read().decode(errors="replace")
    return stdout.channel.recv_exit_status() == 0, out.strip()


# ==========================================
# CONNECT
# ==========================================
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
transport = ssh.get_transport()
transport.set_keepalive(60)

try:
    ok, remote_home = probe(ssh, "cd ~ && pwd")
    if not ok or not remote_home.startswith("/"):
        raise RuntimeError(f"Could not determine remote home directory (got {remote_home!r})")

    REMOTE_HOME = remote_home
    SCRIPT_REMOTE = f"{REMOTE_HOME}/{SCRIPT_NAME}"
    LOG_FILE = f"{REMOTE_HOME}/{LOG_NAME}"

    print(f"Remote home: {REMOTE_HOME}")
    print(f"Log file:    {LOG_FILE}")

    # Say up front whether the experiment is still running, so a log that stops
    # growing is not mistaken for a hang.
    ok, running = probe(ssh, f"pgrep -f {shlex.quote(SCRIPT_REMOTE)} || true")
    if running:
        print(f"Status:      RUNNING (PID(s): {' '.join(running.split())})")
    else:
        print(f"Status:      NOT running ({SCRIPT_NAME} is not alive on this host)")

    ok, _ = probe(ssh, f"test -f {shlex.quote(LOG_FILE)}")
    if not ok:
        print(f"\nNo log file at {LOG_FILE} yet. Has run_remoute.py been started?")
    else:
        print("\n--- Streaming log (Ctrl+C to stop watching) ---")
        # '-n +1' starts at line 1 so nothing is missed; '-F' keeps following
        # even if the file is rotated or recreated by a new run.
        stdin, stdout, stderr = ssh.exec_command(f"tail -n +1 -F {shlex.quote(LOG_FILE)}")
        try:
            for line in iter(stdout.readline, ""):
                print(line, end="")
        except KeyboardInterrupt:
            print("\nStopped viewing, but the remote process is still running.")
finally:
    ssh.close()
    print("SSH session closed.")
