# =============================================================================
# Configure PythonCall.jl to use the project's own "biopert" conda env.
#
# Why: PythonCall needs a *shared* libpython (.so), not a static libpython.a.
# Cluster-provided Python modules sometimes ship only a static lib, which
# PythonCall cannot dynamically load. Conda-forge Python builds always ship
# a shared libpython, so we point PythonCall at the "biopert" conda env
# (built from environment.yml) instead of letting CondaPkg.jl manage its
# own separate Python environment.
#
# Usage: source scripts/setup_python_env.sh
# (must be sourced, not executed, so the exports reach your shell)
# =============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: this script must be sourced, e.g. 'source scripts/setup_python_env.sh'" >&2
    exit 1
fi

set -u -o pipefail

ENV_NAME="biopert"
BIOPERT_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' not found — creating it from environment.yml..."
    conda env create -f "$BIOPERT_ROOT/environment.yml" || return 1
fi

# Resolve the env prefix directly from `conda env list --json` rather than
# via `conda run`, which does not reliably prepend the target env's bin/
# ahead of other Python installs (e.g. pyenv shims) already on PATH.
ENV_PREFIX="$(conda env list --json | grep -o '"[^"]*/envs/'"$ENV_NAME"'"' | tr -d '"' | head -1)"
if [[ -z "$ENV_PREFIX" ]]; then
    echo "Error: could not resolve the prefix of conda env '$ENV_NAME'." >&2
    return 1
fi
PYTHON_EXE="$ENV_PREFIX/bin/python"

# Note: sysconfig's INSTSONAME/Py_ENABLE_SHARED are unreliable on conda-forge
# builds (they can report a static build even though a shared lib is also
# shipped), so glob LIBDIR for the real shared library instead of trusting them.
LIBPYTHON="$("$PYTHON_EXE" -c '
import glob, os, sysconfig
libdir = sysconfig.get_config_var("LIBDIR")
ver = sysconfig.get_config_var("py_version_short")
exact = os.path.join(libdir, f"libpython{ver}.so")
if os.path.exists(exact):
    print(exact)
else:
    candidates = sorted(glob.glob(os.path.join(libdir, f"libpython{ver}*.so*")))
    print(candidates[0] if candidates else "")
')"

if [[ ! -f "$LIBPYTHON" ]]; then
    echo "Error: could not find a shared libpython in '$ENV_NAME' (looked for $LIBPYTHON)." >&2
    echo "PythonCall requires a shared libpython (.so), not a static libpython.a." >&2
    return 1
fi

export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE="$PYTHON_EXE"
export JULIA_PYTHONCALL_LIB="$LIBPYTHON"

echo "PythonCall configured to use conda env '$ENV_NAME':"
echo "  JULIA_PYTHONCALL_EXE = $JULIA_PYTHONCALL_EXE"
echo "  JULIA_PYTHONCALL_LIB = $JULIA_PYTHONCALL_LIB"
