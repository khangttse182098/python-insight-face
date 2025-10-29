Dockerfile and run instructions

This file explains how to build and run the container for the `python-face` FastAPI app on a VM.

Build (from repo root):

```bash
# build image (use -f Dockerfile if dockerfile name differs)
docker build -t python-face:latest .
```

Run (expose port 8000):

```bash
# run container
docker run -d --name python-face -p 8000:8000 python-face:latest
```

Notes:
- The Dockerfile filters out GPU-specific `nvidia-*` and `triton` entries from `requirements.txt` at build time so it can run on a CPU-only VM. If you need GPU support, you must:
  1. Install the NVIDIA drivers on the VM and the NVIDIA Container Toolkit.
  2. Use an appropriate base image (for example, NVIDIA CUDA or a compatible Python image) and remove the filtering step.
- The image will be large because of ML libraries (TensorFlow, PyTorch, insightface, etc.). Consider using a slimmed-down requirements list for production to reduce image size.
- The container runs Uvicorn on port 8000. If you prefer Gunicorn with Uvicorn workers, replace the `CMD` in the `Dockerfile` accordingly.

CI / GitHub Actions deployment
--------------------------------

This repository includes a GitHub Actions workflow at `.github/workflows/deploy.yml` that builds the Docker image, saves it to `image.tar`, copies the tarball to your VM over SCP, then SSHes into the VM, loads the image and restarts the container.

Required repository secrets (set these in GitHub Settings → Secrets and variables → Actions):

- `SSH_HOST` — VM hostname or IP address
- `SSH_USER` — SSH username on the VM
- `SSH_KEY` — Private SSH key (PEM) with access for `SSH_USER`; include newlines as in the key file
- `REMOTE_DIR` — Directory on the VM where `image.tar` will be placed (e.g. `/home/ubuntu/deploy`)
- `SSH_PORT` — Optional, default `22` if not set

Usage:

1. Add the secrets listed above to the repository.
2. Push to `main` (or manually trigger the workflow via Actions → Run workflow).

The workflow will:

- Build the Docker image on the runner
- Save and SCP `image.tar` to `${REMOTE_DIR}` on the VM
- SSH into the VM, run `docker load -i image.tar`, stop and remove any existing `python-face` container, then start a fresh container bound to port 8000.

Notes and troubleshooting:

- The VM must have Docker engine available and the `SSH_USER` must have permission to run Docker commands (or be in the `docker` group / use sudo). If sudo is required, the workflow needs a small modification to prepend `sudo` to Docker commands executed on the remote host.
- The Actions runner will create a temporary `ssh_key` file; keep your private key protected and rotate it if exposed.
- If your `REMOTE_DIR` isn't writable by the SSH user, create a deploy directory and give proper ownership beforehand.

