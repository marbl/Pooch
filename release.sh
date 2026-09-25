#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: ./release.sh <version>"
    echo "Example: ./release.sh 0.2.0"
    exit 1
fi

VERSION="$1"
TAG="v${VERSION}"
TODAY=$(date +"%Y-%m-%d")

# Basic semantic version validation
if ! echo "$VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "Error: version must look like 0.2.0"
    exit 1
fi

# Make sure required files exist
for FILE in version.txt CHANGELOG.md CITATION.cff; do
    if [ ! -f "$FILE" ]; then
        echo "Error: required file not found: $FILE"
        exit 1
    fi
done

# Make sure working tree is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: working tree is not clean."
    echo "Commit or stash your changes before creating a release."
    exit 1
fi

# Make sure tag does not already exist
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Error: tag $TAG already exists."
    exit 1
fi

CURRENT_VERSION=$(cat version.txt)

echo "Preparing POOCH release"
echo "  Current version: $CURRENT_VERSION"
echo "  New version:     $VERSION"
echo

# Update version.txt
printf '%s\n' "$VERSION" > version.txt

# Update CITATION.cff
python3 - "$VERSION" <<'PY'
import sys
from pathlib import Path

version = sys.argv[1]
path = Path("CITATION.cff")

lines = path.read_text().splitlines()

found = False
new_lines = []

for line in lines:
    if line.startswith("version:"):
        new_lines.append(f'version: "{version}"')
        found = True
    else:
        new_lines.append(line)

if not found:
    raise SystemExit("Error: version field not found in CITATION.cff")

path.write_text("\n".join(new_lines) + "\n")
PY

# Add a new CHANGELOG section after the title/introduction
python3 - "$VERSION" "$TODAY" <<'PY'
import sys
from pathlib import Path

version = sys.argv[1]
date = sys.argv[2]

path = Path("CHANGELOG.md")
text = path.read_text()

header = f"## [{version}] - {date}"

if header in text or f"## [{version}]" in text:
    raise SystemExit(f"Error: version {version} already exists in CHANGELOG.md")

section = f"""\
## [{version}] - {date}

### Added
- TODO

### Changed
- TODO

### Fixed
- TODO

"""

lines = text.splitlines()

insert_at = None

for i, line in enumerate(lines):
    if line.startswith("## "):
        insert_at = i
        break

if insert_at is None:
    if text.endswith("\n"):
        text = text + "\n" + section
    else:
        text = text + "\n\n" + section
else:
    new_lines = (
        lines[:insert_at]
        + section.rstrip().splitlines()
        + [""]
        + lines[insert_at:]
    )
    text = "\n".join(new_lines) + "\n"

path.write_text(text)
PY

echo
echo "Updated:"
echo "  version.txt"
echo "  CITATION.cff"
echo "  CHANGELOG.md"
echo
echo "Please edit CHANGELOG.md and replace TODO entries."
echo
read -r -p "Press Enter when CHANGELOG.md is ready, or Ctrl-C to cancel..."

# Check that TODOs were removed from the new section
if grep -A 12 "## \[$VERSION\]" CHANGELOG.md | grep -q "TODO"; then
    echo "Error: TODO entries still exist in the $VERSION changelog section."
    exit 1
fi

git add version.txt CITATION.cff CHANGELOG.md

git commit -m "chore: prepare release $TAG"

git tag -a "$TAG" -m "POOCH $TAG"

echo
echo "Release prepared successfully."
echo
echo "Created:"
echo "  commit: chore: prepare release $TAG"
echo "  tag:    $TAG"
echo
echo "Review with:"
echo
echo "  git show $TAG"
echo
echo "If everything looks good, push with:"
echo
echo "  git push origin main"
echo "  git push origin $TAG"