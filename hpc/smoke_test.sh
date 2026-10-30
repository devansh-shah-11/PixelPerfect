#!/bin/bash
# Smoke test for a Cloud Bursting GPU node (task S-06). Read-only; takes seconds.
# Run it inside a job (sbatch hpc/job_template.sbatch) or an interactive GPU
# session, then paste the whole output into the team chat / AGENT_LOG.md.
#
# It answers: which GPU and driver, is Singularity/Apptainer there, can the node
# render headless (EGL/GLX/Vulkan libraries, needed for the Unreal simulator),
# where is scratch and how full, and can the node download model weights.

section() { printf '\n=== %s ===\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

section "node"
echo "host: $(hostname)"
echo "date: $(date -Is 2>/dev/null || date)"
echo "slurm job: ${SLURM_JOB_ID:-none}  partition: ${SLURM_JOB_PARTITION:-none}  account: ${SLURM_JOB_ACCOUNT:-none}"
echo "os: $(uname -srm)"
[ -r /etc/os-release ] && grep -E '^PRETTY_NAME=' /etc/os-release

section "gpu"
if have nvidia-smi; then
    nvidia-smi --query-gpu=name,memory.total,driver_version,utilization.gpu --format=csv
    nvidia-smi | grep -m1 'CUDA Version' || true
else
    echo "nvidia-smi: not found (no GPU on this machine?)"
fi

section "containers"
for tool in singularity apptainer; do
    if have "$tool"; then echo "$tool: $($tool --version)"; else echo "$tool: not found"; fi
done

section "headless rendering libraries (Unreal simulator)"
if have ldconfig; then
    libs=$(ldconfig -p 2>/dev/null | grep -E 'libEGL_nvidia|libGLX_nvidia|libnvidia-glcore|libvulkan\.so' | awk '{print $1}' | sort -u)
    if [ -n "$libs" ]; then echo "$libs"; else echo "no NVIDIA EGL/GLX/Vulkan libraries found by ldconfig"; fi
else
    echo "ldconfig: not found"
fi
for d in /usr/share/vulkan/icd.d /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d; do
    [ -d "$d" ] && echo "$d: $(ls "$d" 2>/dev/null | tr '\n' ' ')"
done
if have vulkaninfo; then
    vulkaninfo --summary 2>&1 | grep -E 'deviceName|apiVersion|driverName' | head -6
else
    echo "vulkaninfo: not found (can check later inside a container)"
fi

section "storage"
echo "HOME=$HOME"
echo "SCRATCH=${SCRATCH:-not set}"
df -h "$HOME" ${SCRATCH:+"$SCRATCH"} 2>/dev/null
have myquota && myquota 2>/dev/null | head -20

section "software"
if have python3; then echo "python3: $(python3 --version 2>&1)"; else echo "python3: not found"; fi
if have conda; then echo "conda: $(conda --version 2>&1)"; else echo "conda: not found"; fi
have module && echo "module: available (run 'module avail' for the list)"

section "internet (for downloading weights)"
if have curl; then
    curl -sS -o /dev/null -m 10 -w 'huggingface.co: HTTP %{http_code}\n' https://huggingface.co || echo "huggingface.co: unreachable"
    curl -sS -o /dev/null -m 10 -w 'github.com: HTTP %{http_code}\n' https://github.com || echo "github.com: unreachable"
else
    echo "curl: not found"
fi

section "done"
echo "Paste everything above into the team chat."
