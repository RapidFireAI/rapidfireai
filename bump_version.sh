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

is_github_actions() {
    [[ "${GITHUB_ACTIONS:-false}" == "true" ]]
}

check_string_format() {
    local input_string="$1"
    if [[ "$input_string" =~ ^[0-9]+[.][0-9]+[.][0-9]+rc[0-9]+$ ]]; then
        echo "rc"
        return 0 # Success
    elif [[ "$input_string" =~ ^[0-9]+[.][0-9]+[.][0-9]+a[0-9]+$ ]]; then
        echo "alpha"
        return 0 # Success
    elif [[ "$input_string" =~ ^[0-9]+[.][0-9]+[.][0-9]+b[0-9]+$ ]]; then
        echo "beta"
        return 0 # Success
    elif [[ "$input_string" =~ ^[0-9]+[.][0-9]+[.][0-9]+$ ]]; then
        echo "number"
        return 0 # Success
    else
        echo "\"$input_string\" does not match a valid version pattern."
        return 1 # Failure
    fi
}

bump_number() {
    local input_string="$1"
    if [[ "$input_string" =~ ^[0-9]+rc[0-9]+$ ]]; then
        patch_version=${input_string%%rc*}
        rc_version=${input_string##*rc}
        echo "${patch_version}rc$((rc_version + 1))"
        return 0
    elif [[ "$input_string" =~ ^[0-9]+a[0-9]+$ ]]; then
        patch_version=${input_string%%a*}
        alpha_version=${input_string##*a}
        echo "${patch_version}a$((alpha_version + 1))"
        return 0
    elif [[ "$input_string" =~ ^[0-9]+b[0-9]+$ ]]; then
        patch_version=${input_string%%b*}
        beta_version=${input_string##*b}
        echo "${patch_version}b$((beta_version + 1))"
        return 0
    elif [[ "$input_string" =~ ^[0-9]+$ ]]; then
        echo $((input_string + 1))
        return 0
    else
        echo "\"$input_string\" does not match the number, rc alpha, or beta pattern."
        return 1
    fi
}

update_pyproject_toml_file() {
    local NEW_VERSION="$1"
    # Update pyproject.toml
    print_info "Updating pyproject.toml..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
    else
        # Linux
        sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
    fi
    # Verify the changes
    UPDATED_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
    if [ "$UPDATED_VERSION" != "$NEW_VERSION" ]; then
        print_error "Failed to update version in pyproject.toml"
        exit 1
    fi
}

update_build_md_file() {
    local NEW_VERSION="$1"
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
    # Verify the changes
    UPDATED_VERSION=$(grep '^pip install rapidfireai-' BUILD.md | sed 's/pip install rapidfireai-\(.*\)-py3-none-any.whl/\1/')
    if [ "$UPDATED_VERSION" != "$NEW_VERSION" ]; then
        print_error "Failed to update version in BUILD.md"
        exit 1
    fi
}

update_requirements_txt_file() {
    local NEW_VERSION="$1"
    # Update requirements.txt if it has a version comment
    if grep -q "^# version " requirements.txt; then
        print_info "Updating requirements.txt version comment..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/^# version .*/# version $NEW_VERSION/" requirements.txt
        else
            sed -i "s/^# version .*/# version $NEW_VERSION/" requirements.txt
        fi
    fi
    # Verify the changes
    UPDATED_VERSION=$(grep '^# version ' requirements.txt | sed 's/# version \(.*\)/\1/')
    if [ "$UPDATED_VERSION" != "$NEW_VERSION" ]; then
        print_error "Failed to update version in requirements.txt"
        exit 1
    fi
}

update_version_py_file() {
    local NEW_VERSION="$1"
    IFS='.' read -ra VERSION_PARTS <<< "$NEW_VERSION"
    local NEW_MAJOR=${VERSION_PARTS[0]}
    local NEW_MINOR=${VERSION_PARTS[1]}
    local NEW_PATCH=${VERSION_PARTS[2]}
    # Update the central version file
    print_info "Updating rapidfireai/version.py..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" rapidfireai/version.py
        if [[ "$NEW_PATCH" =~ ^[0-9]+$ ]]; then
            sed -i '' "s/^__version_info__ = (.*)/__version_info__ = ($NEW_MAJOR, $NEW_MINOR, $NEW_PATCH)/" rapidfireai/version.py
        else
            sed -i '' "s/^__version_info__ = (.*)/__version_info__ = ($NEW_MAJOR, $NEW_MINOR, \"$NEW_PATCH\")/" rapidfireai/version.py
        fi
    else
        # Linux
        sed -i "s/^__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" rapidfireai/version.py
        if [[ "$NEW_PATCH" =~ ^[0-9]+$ ]]; then
            sed -i "s/^__version_info__ = (.*)/__version_info__ = ($NEW_MAJOR, $NEW_MINOR, $NEW_PATCH)/" rapidfireai/version.py
        else
            sed -i "s/^__version_info__ = (.*)/__version_info__ = ($NEW_MAJOR, $NEW_MINOR, \"$NEW_PATCH\")/" rapidfireai/version.py
        fi
    fi
    # Verify version.py was updated
    UPDATED_VERSION_PY=$(grep '^__version__ = ' rapidfireai/version.py | sed 's/__version__ = "\(.*\)"/\1/')
    if [ "$UPDATED_VERSION_PY" != "$NEW_VERSION" ]; then
        print_error "Failed to update version in rapidfireai/version.py"
        exit 1
    fi
}

update_constants_tsx_file() {
    local NEW_VERSION="$1"
    # Update the JS constants version file
    print_info "Updating rapidfireai/frontend/src/common/constants.tsx..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^export const Version = '.*';/export const Version = '$NEW_VERSION';/" rapidfireai/frontend/src/common/constants.tsx
    else
        # Linux
        sed -i "s/^export const Version = '.*';/export const Version = '$NEW_VERSION';/" rapidfireai/frontend/src/common/constants.tsx
    fi
    # Verify JS constants was updated
    UPDATED_VERSION_JS=$(grep '^export const Version = ' rapidfireai/frontend/src/common/constants.tsx | sed "s/export const Version = '\\(.*\\)';/\\1/")
    if [ "$UPDATED_VERSION_JS" != "$NEW_VERSION" ]; then
        print_error "Failed to update version in rapidfireai/frontend/src/common/constants.tsx"
        exit 1
    fi
}

if ! is_github_actions; then
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
        NEW_MAJOR=$(bump_number "$MAJOR")
        NEW_MINOR=0
        NEW_PATCH=0
        print_info "Bumping major version: $MAJOR.$MINOR.$PATCH → $NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
        ;;
    minor)
        NEW_MAJOR=$MAJOR
        NEW_MINOR=$(bump_number "$MINOR")
        NEW_PATCH=0
        print_info "Bumping minor version: $MAJOR.$MINOR.$PATCH → $NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
        ;;
    patch)
        NEW_MAJOR=$MAJOR
        NEW_MINOR=$MINOR
        if [ "$(check_string_format $CURRENT_VERSION)" == "rc" ]; then
            NEW_PATCH=${PATCH%%rc*}
        elif [ "$(check_string_format $CURRENT_VERSION)" == "alpha" ]; then
            NEW_PATCH=${PATCH%%a*}
        elif [ "$(check_string_format $CURRENT_VERSION)" == "beta" ]; then
            NEW_PATCH=${PATCH%%b*}
        else
            NEW_PATCH=$(bump_number "$PATCH")
        fi
        print_info "Bumping patch version: $MAJOR.$MINOR.$PATCH → $NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
        ;;
    rc)
        NEW_MAJOR=$MAJOR
        NEW_MINOR=$MINOR
        if [ "$(check_string_format $CURRENT_VERSION)" == "number" ]; then
            NEW_PATCH=$(bump_number "$PATCH")rc1
        elif [ "$(check_string_format $CURRENT_VERSION)" == "alpha" ]; then
            NEW_PATCH=${PATCH%%a*}rc1
        elif [ "$(check_string_format $CURRENT_VERSION)" == "beta" ]; then
            NEW_PATCH=${PATCH%%b*}rc1
        else
            NEW_PATCH=$(bump_number "$PATCH")
        fi
        print_info "Bumping rc version: $MAJOR.$MINOR.$PATCH → $NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
        ;;
    alpha)
        NEW_MAJOR=$MAJOR
        NEW_MINOR=$MINOR
        if [ "$(check_string_format $CURRENT_VERSION)" == "number" ]; then
            NEW_PATCH=$(bump_number "$PATCH")a1
        elif [ "$(check_string_format $CURRENT_VERSION)" == "beta" ]; then
            NEW_PATCH=$(bump_number ${PATCH%%b*})a1
        elif [ "$(check_string_format $CURRENT_VERSION)" == "rc" ]; then
            NEW_PATCH=$(bump_number ${PATCH%%rc*})a1
        else
            NEW_PATCH=$(bump_number "$PATCH")
        fi
        print_info "Bumping alpha version: $MAJOR.$MINOR.$PATCH → $NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
        ;;
    beta)
        NEW_MAJOR=$MAJOR
        NEW_MINOR=$MINOR
        if [ "$(check_string_format $CURRENT_VERSION)" == "number" ]; then
            NEW_PATCH=$(bump_number "$PATCH")b1
        elif [ "$(check_string_format $CURRENT_VERSION)" == "alpha" ]; then
            NEW_PATCH=${PATCH%%a*}b1
        elif [ "$(check_string_format $CURRENT_VERSION)" == "rc" ]; then
            NEW_PATCH=$(bump_number ${PATCH%%rc*})b1
        else
            NEW_PATCH=$(bump_number "$PATCH")
        fi
        print_info "Bumping beta version: $MAJOR.$MINOR.$PATCH → $NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
        ;;
    *)
        if check_string_format "$BUMP_TYPE" > /dev/null; then
            if [[ "$BUMP_TYPE" == v* ]]; then
                BUMP_TYPE=${BUMP_TYPE#v}
            fi
            IFS='.' read -ra VERSION_PARTS <<< "$BUMP_TYPE"
            NEW_MAJOR=${VERSION_PARTS[0]}
            NEW_MINOR=${VERSION_PARTS[1]}
            NEW_PATCH=${VERSION_PARTS[2]}
            print_info "Setting specific version: $MAJOR.$MINOR.$PATCH → $NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"
        else
            print_error "Invalid bump type. Use: major, minor, patch, alpha, beta, rc or specific version"
            exit 1
        fi
        ;;
esac

NEW_VERSION="$NEW_MAJOR.$NEW_MINOR.$NEW_PATCH"

update_pyproject_toml_file "$NEW_VERSION"
update_version_py_file "$NEW_VERSION"
update_constants_tsx_file "$NEW_VERSION"
update_build_md_file "$NEW_VERSION"
update_requirements_txt_file "$NEW_VERSION"

print_success "Version updated to $NEW_VERSION"

if ! is_github_actions; then
    # Commit the changes
    print_info "Committing version bump..."
    git add pyproject.toml requirements.txt rapidfireai/version.py rapidfireai/frontend/src/common/constants.tsx BUILD.md README.md
    git commit -m "Bump version to $NEW_VERSION"

    # Create and push tag
    print_info "Creating git tag v$NEW_VERSION..."
    git tag -a "v$NEW_VERSION" -m "Release version $NEW_VERSION"

    print_success "🎉 Version $NEW_VERSION has been bumped and tagged!"
    print_info "To deploy to PyPI, push the tag and create a release:"
    echo "  git push origin v$NEW_VERSION"
    echo
    print_info "Or push all tags:"
    echo "  git push --tags"
else
    print_success "🎉 Version bump script completed successfully!"
    echo ""   
    # Show git status if available
    if command -v git >/dev/null 2>&1; then
        echo ""
        print_info "Modified files:"
        git status --porcelain 2>/dev/null || echo "Git status not available"
    fi  
fi
