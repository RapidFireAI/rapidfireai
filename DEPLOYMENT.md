# RapidFire AI Deployment Guide

This guide covers automated deployment and version management for RapidFire AI.

## Overview

The project includes several tools for automated deployment:
- **Version bumping script**: Automatically increments version numbers
- **GitHub Actions**: Automatically build and deploy to TestPyPI
- **Manual deployment**: Trigger builds manually when needed

## Quick Start

### 1. Bump Version and Deploy

```bash
# Bump patch version (0.9.5 → 0.9.6)
./bump_version.sh patch

# Push the tag to trigger automatic deployment
git push origin v0.9.6
```

That's it! The GitHub Action will automatically:
- Build the package
- Upload to TestPyPI
- Store build artifacts

### 2. Manual Deployment

If you need to deploy without creating a tag:

1. Go to GitHub Actions → "Manual Deploy to TestPyPI"
2. Click "Run workflow"
3. Enter the version number
4. Choose whether to upload or just build
5. Click "Run workflow"

## Detailed Usage

### Version Bumping Script

The `bump_version.sh` script handles semantic versioning:

```bash
# Patch version (bug fixes, small changes)
./bump_version.sh patch    # 0.9.5 → 0.9.6

# Minor version (new features, backward compatible)
./bump_version.sh minor    # 0.9.5 → 0.10.0

# Major version (breaking changes)
./bump_version.sh major    # 0.9.5 → 1.0.0
```

**What the script does:**
1. Checks for uncommitted changes
2. Reads current version from `pyproject.toml`
3. Calculates new version based on bump type
4. Updates `pyproject.toml`, `requirements.txt`, `version.py`
5. Commits changes with message "Bump version to X.Y.Z"
6. Creates git tag `vX.Y.Z`

### GitHub Actions

#### Automatic Deployment (Tag-based)
- **Trigger**: Push a version tag (e.g., `v0.9.6`)
- **Action**: Automatically builds and uploads to TestPyPI
- **File**: `.github/workflows/deploy-testpypi.yml`

#### Manual Deployment
- **Trigger**: Manual workflow dispatch
- **Action**: Build and optionally upload to TestPyPI
- **File**: `.github/workflows/manual-deploy.yml`
- **Use case**: Testing builds, deploying specific versions

## Configuration

### TestPyPI Setup

1. **Create TestPyPI account**: [https://test.pypi.org/account/register/](https://test.pypi.org/account/register/)

2. **Generate API token**: 
   - Go to [TestPyPI account settings](https://test.pypi.org/manage/account/token/)
   - Create a new token with "Entire account" scope

3. **Add to GitHub secrets**:
   - Go to your repository → Settings → Secrets and variables → Actions
   - Add new repository secret: `TESTPYPI_API_TOKEN`
   - Paste your TestPyPI token

### Local Configuration (Optional)

Create a `.pypirc` file for local uploads:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = your-pypi-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = your-testpypi-token-here
```

**Note**: `.pypirc` is already in `.gitignore` for security.

## Workflow Examples

### Standard Release Process

```bash
# 1. Make your changes and commit them
git add .
git commit -m "Add new feature X"

# 2. Bump version
./bump_version.sh minor

# 3. Push changes and tag
git push origin main
git push origin v0.10.0

# 4. GitHub Action automatically deploys to TestPyPI
```

### Hotfix Release

```bash
# 1. Fix the bug and commit
git add .
git commit -m "Fix critical bug Y"

# 2. Bump patch version
./bump_version.sh patch

# 3. Push tag to trigger deployment
git push origin v0.9.6
```

### Manual Build and Test

```bash
# 1. Build locally
rm -rf dist/ *.egg-info/ .eggs/
python -m build

# 2. Test installation
pip install dist/rapidfireai-*.whl

# 3. If satisfied, use manual deployment workflow
# or bump version and push tag
```

## Troubleshooting

### Common Issues

**Build fails in GitHub Action:**
- Check Python version compatibility
- Verify all dependencies are in `pyproject.toml`
- Check for syntax errors in Python code

**Upload fails:**
- Verify `TESTPYPI_API_TOKEN` secret is set
- Check token permissions on TestPyPI
- Ensure package name is unique

**Version bump fails:**
- Ensure you're in the project root directory
- Check for uncommitted changes
- Verify `pyproject.toml` has correct version format

### Getting Help

1. Check GitHub Actions logs for detailed error messages
2. Verify TestPyPI token permissions
3. Test builds locally before pushing tags
4. Check package name conflicts on TestPyPI

## Best Practices

1. **Always test locally** before pushing tags
2. **Use semantic versioning** consistently
3. **Keep dependencies updated** with minimum version constraints
4. **Monitor GitHub Actions** for build/deployment status
5. **Use TestPyPI** for testing before PyPI releases
6. **Document breaking changes** in release notes

## Next Steps

Once you're comfortable with TestPyPI deployment:

1. **Set up PyPI deployment** for production releases
2. **Add release notes** automation
3. **Implement changelog** generation
4. **Add deployment notifications** (Slack, Discord, etc.)
5. **Set up staging environments** for testing 