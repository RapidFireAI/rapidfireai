#!/bin/bash

# Version bumping script for RapidFire AI
# Usage: ./bump_version.sh [major|minor|patch]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository. Please run this script from the project root."
    exit 1
fi

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_warning "You have uncommitted changes. Please commit or stash them before bumping version."
    git status --short
    echo
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

if [ -z "$CURRENT_VERSION" ]; then
    print_error "Could not find version in pyproject.toml"
    exit 1
fi

print_info "Current version: $CURRENT_VERSION"

# Parse version components
IFS='.' read -ra VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR=${VERSION_PARTS[0]}
MINOR=${VERSION_PARTS[1]}
PATCH=${VERSION_PARTS[2]}

# Determine bump type
BUMP_TYPE=${1:-patch}

case $BUMP_TYPE in
    major)
        NEW_MAJOR=$((MAJOR + 1))
        NEW_MINOR=0
        NEW_PATCH=0
        print_info "Bumping major version: $MAJOR.$MINOR.$PATCH → $NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
        ;;
    minor)
        NEW_MAJOR=$MAJOR
        NEW_MINOR=$((MINOR + 1))
        NEW_PATCH=0
        print_info "Bumping minor version: $MAJOR.$MINOR.$PATCH → $NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
        ;;
    patch)
        NEW_MAJOR=$MAJOR
        NEW_MINOR=$MINOR
        NEW_PATCH=$((PATCH + 1))
        print_info "Bumping patch version: $MAJOR.$MINOR.$PATCH → $NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
        ;;
    *)
        print_error "Invalid bump type. Use: major, minor, or patch"
        exit 1
        ;;
esac

NEW_VERSION="$NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"

# Update pyproject.toml
print_info "Updating pyproject.toml..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
else
    # Linux
    sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
fi

# Update requirements.txt if it has a version comment
if grep -q "^# version " requirements.txt; then
    print_info "Updating requirements.txt version comment..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/^# version .*/# version $NEW_VERSION/" requirements.txt
    else
        sed -i "s/^# version .*/# version $NEW_VERSION/" requirements.txt
    fi
fi

# Update README.md if it has a version comment
if grep -q "^# RapidFire AI " README.md; then
    print_info "Updating README.md version comment and wheel version..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/^# RapidFire AI .*/# RapidFire AI $NEW_VERSION/" README.md
        sed -i '' "s/^pip install rapidfireai-.*/pip install rapidfireai-$NEW_VERSION-py3-none-any.whl/" README.md
    else
        sed -i "s/^# RapidFire AI .*/# RapidFire AI$NEW_VERSION/" README.md
        sed -i "s/^pip install rapidfireai-.*/pip install rapidfireai-$NEW_VERSION-py3-none-any.whl/" README.md
    fi
fi

# Update BUILD.md if it has a version comment
if grep -q "^# RapidFire AI" BUILD.md; then
    print_info "Updating BUILD.md version comment and wheel version..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/^# RapidFire AI .*/# RapidFire AI $NEW_VERSION/" BUILD.md
        sed -i '' "s/^pip install rapidfireai-.*/pip install rapidfireai-$NEW_VERSION-py3-none-any.whl/" BUILD.md
    else
        sed -i "s/^# RapidFire AI .*/# RapidFire AI $NEW_VERSION/" BUILD.md
        sed -i "s/^pip install rapidfireai-.*/pip install rapidfireai-$NEW_VERSION-py3-none-any.whl/" BUILD.md
    fi
fi

# Update the central version file
print_info "Updating rapidfireai/version.py..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/^__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" rapidfireai/version.py
    sed -i '' "s/^__version_info__ = (.*)/__version_info__ = ($NEW_MAJOR, $NEW_MINOR, $NEW_PATCH)/" rapidfireai/version.py
else
    # Linux
    sed -i "s/^__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" rapidfireai/version.py
    sed -i "s/^__version_info__ = (.*)/__version_info__ = ($NEW_MAJOR, $NEW_MINOR, $NEW_PATCH)/" rapidfireai/version.py
fi

# Verify the changes
UPDATED_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
if [ "$UPDATED_VERSION" != "$NEW_VERSION" ]; then
    print_error "Failed to update version in pyproject.toml"
    exit 1
fi

# Verify version.py was updated
UPDATED_VERSION_PY=$(grep '^__version__ = ' rapidfireai/version.py | sed 's/__version__ = "\(.*\)"/\1/')
if [ "$UPDATED_VERSION_PY" != "$NEW_VERSION" ]; then
    print_error "Failed to update version in rapidfireai/version.py"
    exit 1
fi

print_success "Version updated to $NEW_VERSION"

if [ "$2" == "test"]; then
    IS_TEST=test
else
    IS_TEST=""
fi

# Commit the changes
print_info "Committing version bump..."
git add pyproject.toml requirements.txt rapidfireai/version.py BUILD.md README.md
git commit -m "Bump $IS_TEST version to $NEW_VERSION"

# Create and push tag
if [ "$IS_TEST" == "test" ]; then
    print_info "Creating git tag test$NEW_VERSION..."
    git tag -a "test$NEW_VERSION" -m "Test Release version $NEW_VERSION"
else
    print_info "Creating git tag v$NEW_VERSION..."
    git tag -a "v$NEW_VERSION" -m "Release version $NEW_VERSION"
fi

print_success "Version $NEW_VERSION has been bumped and tagged!"
if [ "$IS_TEST" == "test" ]; then
    print_info "To deploy to TestPyPI, push the tag:"
    echo "  git push origin test$NEW_VERSION"
else
    print_info "To deploy to PyPI, push the tag and create a release:"
    echo "  git push origin v$NEW_VERSION"
fi
echo
print_info "Or push all tags:"
echo "  git push --tags" 