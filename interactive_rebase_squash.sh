#!/bin/bash

# Improved interactive rebase script that handles the first commit issue
# This ensures the first commit is always 'pick' and subsequent ones are 'squash'

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}RapidFireAI Interactive Rebase Script (Fixed)${NC}"
echo "This version handles the 'first commit must be pick' issue"
echo ""

# Check if we're in the right directory
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

# Check current state
if [ -f ".git/rebase-merge/git-rebase-todo" ] || [ -f ".git/rebase-apply/next" ]; then
    echo -e "${YELLOW}Currently in a rebase state. Use the fix script instead:${NC}"
    echo "./fix_rebase_issue.sh"
    exit 1
fi

# Check if we're on main branch
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    echo -e "${RED}Error: Not on main branch (currently on: $current_branch)${NC}"
    echo "Please switch to main branch first: git checkout main"
    exit 1
fi

# Backup current branch if not already done
backup_pattern="main-backup-$(date +%Y%m%d)"
existing_backup=$(git branch --list "${backup_pattern}*" | head -1 | sed 's/^[* ] //')

if [ -z "$existing_backup" ]; then
    backup_branch="main-backup-$(date +%Y%m%d-%H%M%S)"
    echo -e "${YELLOW}Creating backup branch: $backup_branch${NC}"
    git checkout -b "$backup_branch" >/dev/null 2>&1
    git checkout main >/dev/null 2>&1
    echo -e "${GREEN}✓ Backup created: $backup_branch${NC}"
else
    backup_branch="$existing_backup"
    echo -e "${GREEN}✓ Using existing backup: $backup_branch${NC}"
fi
echo ""

# Analyze commits - get them in chronological order (oldest first)
echo -e "${YELLOW}Analyzing commits...${NC}"
all_commits_reverse=$(git log --reverse --oneline)
non_pr_commits=$(git log --oneline main --invert-grep --grep="#[0-9]")
pr_commits=$(git log --oneline main --grep="#[0-9]")

echo -e "${BLUE}Analysis:${NC}"
echo "- Total commits: $(git log --oneline | wc -l)"
echo "- PR commits (preserve): $(echo "$pr_commits" | wc -l)"
echo "- Direct commits (squash): $(echo "$non_pr_commits" | wc -l)"
echo ""

# Find the commit to rebase from (usually second commit from bottom)
root_commit=$(git log --oneline | tail -1 | cut -d' ' -f1)
second_commit=$(git log --oneline | tail -n 2 | head -n 1 | cut -d' ' -f1)

echo -e "${BLUE}Rebase Plan:${NC}"
echo "Root commit: $(git log --oneline -1 $root_commit)"
echo "Rebase from: $(git log --oneline -1 $second_commit)"
echo ""

# Show what will happen
echo -e "${YELLOW}Commits that will be squashed (but FIRST one will be pick):${NC}"
echo "$non_pr_commits" | nl -v 0 | while read num commit; do
    if [ "$num" -eq 0 ]; then
        echo "  → PICK (first):   $commit"
    else
        echo "  → SQUASH:         $commit"
    fi
done
echo ""

echo -e "${YELLOW}PR commits that will be preserved as pick:${NC}"
echo "$pr_commits" | head -5 | sed 's/^/  → PICK:           /'
if [ $(echo "$pr_commits" | wc -l) -gt 5 ]; then
    echo "  ... and $(( $(echo "$pr_commits" | wc -l) - 5 )) more PR commits"
fi
echo ""

# Create a detailed instruction file
instructions_file="/tmp/rebase_instructions_detailed_$(date +%s).txt"
cat > "$instructions_file" << EOF
# DETAILED Interactive Rebase Instructions for RapidFireAI
#
# CRITICAL: The first commit in the list MUST be 'pick', not 'squash'!
# 
# PATTERN TO FOLLOW:
# pick    <hash> <first direct commit>     ← MUST be 'pick'
# squash  <hash> <other direct commits>    ← Change these to 'squash'
# pick    <hash> <PR commit with #XX>      ← Keep as 'pick'
# pick    <hash> <another PR commit>       ← Keep as 'pick'
#
# WHAT TO DO:
# 1. First commit in the editor: KEEP as 'pick' (even if it's a direct commit)
# 2. Other direct commits: Change 'pick' to 'squash' (or 's')
# 3. PR commits (with #XX): Keep as 'pick'
# 4. Save and close (vim: :wq, nano: Ctrl+X then Y)
#
# DIRECT COMMITS - First one stays 'pick', others become 'squash':
$(echo "$non_pr_commits" | head -1 | sed 's/^/# PICK:    /')
$(echo "$non_pr_commits" | tail -n +2 | sed 's/^/# SQUASH:  /')
#
# PR COMMITS - All stay 'pick':
$(echo "$pr_commits" | sed 's/^/# PICK:    /')
#
# Your backup: $backup_branch
EOF

echo -e "${GREEN}Ready for interactive rebase!${NC}"
echo ""
echo -e "${YELLOW}KEY RULES:${NC}"
echo "1. First commit = PICK (never squash the first one!)"
echo "2. Other direct commits = SQUASH"  
echo "3. PR commits (#XX) = PICK"
echo ""
echo "Detailed instructions: $instructions_file"
echo ""

read -p "Start interactive rebase now? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Operation cancelled${NC}"
    echo "Start manually anytime with:"
    echo "  git rebase -i $second_commit"
    echo ""
    echo "Remember: First commit = pick, others = squash (for direct commits)"
    echo "Backup: $backup_branch"
    exit 0
fi

echo -e "${YELLOW}Starting interactive rebase...${NC}"
echo "Remember: First commit must stay as 'pick'!"
echo ""

# Start the interactive rebase
git rebase -i "$second_commit" || {
    echo -e "${RED}Rebase encountered issues${NC}"
    echo ""
    git status
    echo ""
    echo -e "${YELLOW}Common fixes:${NC}"
    echo "• If 'cannot squash without previous commit':"
    echo "  git rebase --edit-todo"
    echo "  Change first 'squash' to 'pick'"
    echo ""
    echo "• To continue: git rebase --continue"
    echo "• To abort: git rebase --abort"
    echo ""
    echo "• Use fix script: ./fix_rebase_issue.sh"
    echo ""
    echo "Backup: $backup_branch"
    exit 1
}

echo -e "${GREEN}Interactive rebase completed successfully!${NC}"
echo ""
echo "Verify results:"
echo "  git log --oneline | head -10"
echo ""
echo "If satisfied:"
echo "  git push --force-with-lease origin main"
echo ""
echo "If problems:"
echo "  git reset --hard $backup_branch"