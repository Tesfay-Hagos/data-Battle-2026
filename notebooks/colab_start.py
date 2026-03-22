# %% [markdown]
# # DataBattle 2026 — Colab Starter
# Run this notebook once at the beginning of every Colab session.
# It clones / pulls the latest code from GitHub and mounts Google Drive.

# %% [markdown]
# ## Step 1 — Clone or update the repo

# %%
import os, subprocess, sys

REPO_URL  = "https://github.com/Tesfay-Hagos/data-Battle-2026.git"
REPO_DIR  = "/content/data-Battle-2026"

# For private repos: add your token to Colab Secrets (key icon in left sidebar)
# Secret name: GITHUB_TOKEN  — never paste tokens in code or chat
_auth_url = REPO_URL
try:
    from google.colab import userdata
    _token = userdata.get("GITHUB_TOKEN")   # raises SecretNotFoundError if missing
    if _token:
        _auth_url = REPO_URL.replace("https://", f"https://{_token}@")
        print("GitHub token loaded from Secrets.")
    else:
        print("WARNING: GITHUB_TOKEN secret is empty.")
except Exception as _e:
    print(f"WARNING: Could not load GITHUB_TOKEN from Colab Secrets ({_e}).\n"
          "  → Open the 🔑 key icon in the left sidebar.\n"
          "  → Check that 'Notebook access' is toggled ON for GITHUB_TOKEN.")

if os.path.exists(f"{REPO_DIR}/.git"):
    print("Repo already cloned — pulling latest …")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
else:
    print("Cloning repo …")
    result = subprocess.run(["git", "clone", _auth_url, REPO_DIR])
    if result.returncode != 0:
        raise RuntimeError(
            "git clone failed.\n"
            "Add your GitHub token to Colab Secrets:\n"
            "  1. Click the key icon (🔑) in the left sidebar\n"
            "  2. Add secret name: GITHUB_TOKEN\n"
            "  3. Value: your personal access token (repo scope)\n"
            "  4. Re-run this cell"
        )

os.chdir(REPO_DIR)
print(f"Working directory: {os.getcwd()}")

# %% [markdown]
# ## Step 2 — Install dependencies

# %%
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("Dependencies installed.")

# %% [markdown]
# ## Step 3 — Run environment setup (mounts Drive, defines all paths)
# Drive will prompt for permission once — click Allow.

# %%
exec(open("env_setup.py").read())

# %% [markdown]
# ## Step 4 — Ready. Run any notebook:
#
# ```python
# exec(open("notebooks/01_eda.py").read())
# ```
#
# Or open the .ipynb version:
# - File → Open → navigate to `/content/data-Battle-2026/notebooks/01_eda.ipynb`
