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
pip install rapidfireai-0.12.1-py3-none-any.whl

export PATH="$HOME/.local/bin:$PATH"

rapidfireai --version
# RapidFire AI 0.12.1

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
./setup/bump_version.sh patch

# Bump minor version (0.9.5 → 0.10.0)
./setup/bump_version.sh minor

# Bump major version (0.9.5 → 1.0.0)
./setup/bump_version.sh major

# Bump release candidate version (0.9.5 → 0.9.6rc1)
./setup/bump_version.sh rc

# Bump to a specific version
./setup/bump_version.sh 1.1.1rc1
```

The script will:
- Update the version in `pyproject.toml`
- Update the version comment in `requirements.txt`
- Update the version and version_info in `version.py`
- Update the Version in `constants.tsx`
- Update the version in wheel and comment in `BUILD.md`
- Commit the changes
- Create a git tag for the new version

### Manual Release Candidate Deployment
The project includes a GitHub Action that builds and deploys to PyPI when manually triggered:

1. Go to: [Manual Deploy Action](https://github.com/RapidFireAI/rapidfireai/actions/workflows/manual-deploy.yml)
2. Click grey `Run workflow` button
3. Enter desired version number starting with the letter `v` and ending with the letters `rc` i.e. v1.0.1rc  there should be no number
   after the rc letters, the release candidate number will be determined by existing rc versions if any.
4. Optionally check `Build only` to not upload to PyPI.
5. Click green `Run Workflow` button

The above actions will do the following:

1. Determine the next rc version based on existing tags on GitHub, if none will create rc1 if any other number will increment by 1
2. Create a new branch called release/vW.X.YrcZ (with W, X, Y deteremined by input and Z automatically calculated)
3. Run `bump_version.sh` to bump the version to the specified release candidate version
4. Tag the new branch with the version
5. Push code to the new branch
6. Build the frontend and PyPI artifacts
7. Publish to PyPI unless chosen to skip

***Note: Only after releasing final version, delete all rc tags on GitHub, delete all rc release branches, and delete rc builds on pypi.org


### Automated TestPyPI Deployment
The project includes a GitHub Action that automatically builds and deploys to TestPyPI when you push a version tag:

1. **Tag a branch**: Tag the desired branch with v
3. **GitHub Action triggers**: Automatically builds and uploads to TestPyPI

**Manual TestPyPI upload** (if needed):
```bash
python -m twine upload --repository testpypi dist/*
```

### Semi-Automated PyPI Release Deployment
The project includes a GitHub Action that semi automatically builds and deploys to PyPI when you create a release:

1. **Create a new branch**: Create a branch called `release/vX.Y.Z`, i.e. 
```bash
git fetch origin
git checkout main
git pull
git checkout -b release/vX.Y.Z
```
2. **Run bump_version.sh**: Run `bump_version.sh` with flag as specified above, i.e.
```bash
./setup/bump_version.sh minor
```
3. **Update CHANGELOG.md**: Update `CHANGELOG.md` with all information from PRs since last release
4. **Push code to GitHub**: Push all changes to GitHub, i.e.
```bash
git add .
git commit -m "Release vX.Y.Z"
git push --set-upstream origin release/vX.Y.Z
```
5. **Create/Approval for a PR**: Create and have approved a new PR
6. **Merge PR**: After approval merge PR
7. **Draft Release**: On [GitHub Releases](https://github.com/RapidFireAI/rapidfireai/releases) Draft a new release:
- **Create a new tag**: vX.Y.Z
- **Set Title** to vX.Y.Z
- **Generate release notes**
- **Publish Release**
8. **Remove all release candidates**: Remove all release candidate tags and branches, remove from PyPI 