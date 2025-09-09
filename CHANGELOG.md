# Change Log
All notable changes `rapidfireai` will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/).

## Types of changes
- Added for new features.
- Changed for changes in existing functionality.
- Deprecated for soon-to-be removed features.
- Removed for now removed features.
- Fixed for any bug fixes.
- Security in case of vulnerabilities.

## [v0.9.9]
### Added
- **MLflow 3.2+ Support**: Upgraded to MLflow 3.2.0+ for enhanced experiment tracking capabilities
- **Frontend Build Integration**: Automated frontend building in CI/CD pipelines for PyPI releases
- **Interactive Controller Icon**: New theme-aware controller icon with dark/light mode support
- **Enhanced Experiment Logs**: Improved experiment logging with intelligent caching and real-time updates
- **Version Management**: Automated version bumping and release management tools

### Changed
- **Frontend Logging**: Experiment logs now update independently of running experiment status for better debugging
- **Build Process**: Frontend build artifacts are now properly excluded from git and built during deployment

## [v0.9.9]
Initial open source release