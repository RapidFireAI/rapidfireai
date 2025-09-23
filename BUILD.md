# RapidFire
An open source ML tool that allows for efficient, optimized, and user-friendly model training experimentation.

## Building RapidFire AI

### Building pypi package
```bash
# install nodejs locally, either using nvm or installing latest node 22.x
# Option 1: NodeSource repository (official)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs

# Option 2: nvm (Node Version Manager) - works on all Linux distros
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 22
nvm use 22

# on mac
# brew install node@22

# from rapidfireai/frontend, create the production-optimized frontend build
cd rapidfireai/frontend
node ./yarn/releases/yarn-4.9.1.cjs install
node ./yarn/releases/yarn-4.9.1.cjs build

# go back to rapidfireai root
cd ../..

# Remove any distributions and build to /dist folder
rm -rf dist/ *.egg-info/ .eggs/ && python -m build

# Copy or rsync /dist folder over to remote instance
# Alternatively, git clone the repo
rsync -av dist/ user:~/rapidfire

# from directory where dist/ folder is
pip install rapidfireai-0.10.0-py3-none-any.whl

export PATH="$HOME/.local/bin:$PATH"

rapidfireai --version
# RapidFire AI 0.10.0

# install specific dependencies and initialize rapidfire
rapidfireai init

# start the rapidfire server
rapidfireai start
```

### Automated Deployment and Version Management

#### Version Bumping
Use the included `bump_version.sh` script to automatically increment version numbers:

```bash
# Bump patch version (0.9.5 → 0.9.6)
./bump_version.sh patch

# Bump minor version (0.9.5 → 0.10.0)
./bump_version.sh minor

# Bump major version (0.9.5 → 1.0.0)
./bump_version.sh major
```

The script will:
- Update the version in `pyproject.toml`
- Update the version comment in `requirements.txt`
- Commit the changes
- Create a git tag for the new version

#### Automated TestPyPI Deployment
The project includes a GitHub Action that automatically builds and deploys to TestPyPI when you push a version tag:

1. **Bump the version**: `./bump_version.sh patch`
2. **Push the tag**: `git push origin test0.9.6`
3. **GitHub Action triggers**: Automatically builds and uploads to TestPyPI

**Prerequisites for TestPyPI deployment:**
- Add `TESTPYPI_API_TOKEN` to your GitHub repository secrets
- Get the token from [TestPyPI account settings](https://test.pypi.org/manage/account/token/)

**Manual TestPyPI upload** (if needed):
```bash
python -m twine upload --repository testpypi dist/*
```