# %% [markdown]
# # DataBattle 2026 — Colab Starter
# Run this notebook once at the beginning of every Colab session.
# It clones / pulls the latest code from GitHub and mounts Google Drive.

# %% [markdown]
# ## Step 1 — Clone or update the repo

# %%
import os

REPO_URL  = "https://github.com/Tesfay-Hagos/data-Battle-2026.git"
REPO_DIR  = "/content/data-Battle-2026"

if os.path.exists(REPO_DIR):
    print("Repo already cloned — pulling latest …")
    os.system(f"git -C {REPO_DIR} pull")
else:
    print("Cloning repo …")
    os.system(f"git clone {REPO_URL} {REPO_DIR}")

os.chdir(REPO_DIR)
print(f"Working directory: {os.getcwd()}")

# %% [markdown]
# ## Step 2 — Install dependencies

# %%
os.system("pip install -q -r requirements.txt")
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
