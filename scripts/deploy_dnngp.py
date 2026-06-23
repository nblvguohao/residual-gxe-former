"""Upload updated DNNGP code and launch script to server, then execute."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import paramiko

SERVER = "100.66.246.20"
USER = "amax"
# Password is typically provided via environment or interactive input
PASSWORD = os.environ.get("AMAX_PASSWORD", "")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"

FILES_TO_UPLOAD = [
    (SRC / "residual_gxe" / "models" / "baselines.py",
     "/opt/data/lgh/gwas1/src/residual_gxe/models/baselines.py"),
    (SCRIPTS / "06_run_ablations.py",
     "/opt/data/lgh/gwas1/scripts/06_run_ablations.py"),
    (SCRIPTS / "launch_dnngp_only.sh",
     "/opt/data/lgh/gwas1/scripts/launch_dnngp_only.sh"),
]


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER, username=USER, password=PASSWORD, look_for_keys=True)
    return client


def upload_files(sftp: paramiko.SFTPClient):
    for local_path, remote_path in FILES_TO_UPLOAD:
        if not local_path.exists():
            print(f"  SKIP missing: {local_path}")
            continue
        print(f"  Upload: {local_path.name} -> {remote_path}")
        sftp.put(str(local_path), remote_path)
        # Verify
        remote_stat = sftp.stat(remote_path)
        local_size = local_path.stat().st_size
        if remote_stat.st_size != local_size:
            print(f"  WARNING: size mismatch! local={local_size} remote={remote_stat.st_size}")


def run_remote(client: paramiko.SSHClient):
    """Execute the DNNGP launch script on the server."""
    cmd = (
        "cd /opt/data/lgh/gwas1 && "
        "chmod +x scripts/launch_dnngp_only.sh && "
        "bash scripts/launch_dnngp_only.sh"
    )
    print(f"\nExecuting: {cmd}\n")
    stdin, stdout, stderr = client.exec_command(cmd)

    # Stream output line by line
    for line in iter(stdout.readline, ""):
        print(line, end="")
    # Print any stderr
    stderr_text = stderr.read().decode()
    if stderr_text:
        print(f"\n[STDERR]\n{stderr_text}")

    exit_code = stdout.channel.recv_exit_status()
    print(f"\nExit code: {exit_code}")
    return exit_code


def main():
    if not PASSWORD:
        print("ERROR: Set AMAX_PASSWORD environment variable")
        print("  $env:AMAX_PASSWORD = 'your_password'")
        sys.exit(1)

    print(f"Connecting to {USER}@{SERVER}...")
    client = connect()
    print("Connected.\n")

    print("Uploading files...")
    sftp = client.open_sftp()
    upload_files(sftp)
    sftp.close()
    print("Upload complete.\n")

    print("Running DNNGP launch script...")
    exit_code = run_remote(client)

    client.close()
    print(f"\nDone. Exit code: {exit_code}")


if __name__ == "__main__":
    main()
