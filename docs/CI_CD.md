# CI/CD Pipeline Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline for the Kalshi AI Trading Bot.

## Table of Contents

- [Overview](#overview)
- [Workflows](#workflows)
- [Status Badges](#status-badges)
- [Local Development](#local-development)
- [Troubleshooting](#troubleshooting)
- [Configuration Files](#configuration-files)

---

## Overview

The CI/CD pipeline is implemented using **GitHub Actions** and consists of four main workflows:

1. **Tests** - Automated testing with pytest
2. **Code Quality** - Linting, formatting, and type checking
3. **Security** - Vulnerability scanning and secret detection
4. **Docker** - Container image building and publishing

All workflows run automatically on:
- Push to `main`, `develop`, or `claude/*` branches
- Pull requests to `main` or `develop`
- Some workflows also run on schedule (weekly security scans)

---

## Workflows

### 1. Tests Workflow

**File**: `.github/workflows/test.yml`

**Purpose**: Run automated tests and measure code coverage

**Triggers**:
- Push to `main`, `develop`, `claude/*`
- Pull requests to `main`, `develop`

**What it does**:
- Sets up Python 3.12 environment
- Installs dependencies from `requirements.txt`
- Runs pytest for unit tests
- Runs pytest for integration tests (if available)
- Generates coverage report
- Uploads coverage to Codecov (if configured)
- Uploads test results as artifacts

**Passing Criteria**:
- All tests pass
- Code coverage > 70% (recommended)

**Example output**:
```
tests/unit/test_budget_validator.py ........  [ 25%]
tests/unit/test_volume_validator.py ........  [ 50%]
tests/unit/test_edge_validator.py ........    [ 75%]
tests/unit/test_position_limits_validator.py [100%]

========== 32 passed in 2.45s ==========
```

---

### 2. Code Quality Workflow

**File**: `.github/workflows/code-quality.yml`

**Purpose**: Enforce code quality standards and consistent formatting

**Triggers**:
- Push to `main`, `develop`, `claude/*`
- Pull requests to `main`, `develop`

**What it does**:

**Lint Job**:
- **Black**: Check code formatting (120 char line length)
- **isort**: Check import sorting
- **flake8**: Lint for syntax errors and code quality
- **mypy**: Type checking (ignoring missing imports)
- **pylint**: Static code analysis

**Format-check Job**:
- Runs Black to detect formatting issues
- Posts comment on PR if formatting is needed
- Suggests auto-format commands

**Configuration**:
- `pyproject.toml` - Black, isort, mypy, pylint, pytest
- `.flake8` - Flake8 configuration
- `.pylintrc` - Detailed pylint rules

**Auto-fixing issues locally**:
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check types
mypy src/

# Run all linters
flake8 src/ tests/
pylint src/
```

---

### 3. Security Workflow

**File**: `.github/workflows/security.yml`

**Purpose**: Scan for security vulnerabilities and secrets

**Triggers**:
- Push to `main`, `develop`, `claude/*`
- Pull requests to `main`, `develop`
- Weekly schedule (Mondays at 9 AM UTC)

**What it does**:

**Bandit Job** (Python Security):
- Scans source code for common security issues
- Checks for:
  - SQL injection vulnerabilities
  - Hardcoded passwords/secrets
  - Insecure random number generation
  - Unsafe YAML loading
  - Use of `eval()` or `exec()`
- Uploads results as artifact

**Safety Job** (Dependency Vulnerabilities):
- Checks all dependencies for known CVEs
- Uses Safety DB (updated daily)
- Reports vulnerable packages and versions
- Suggests upgrade paths

**Semgrep Job** (SAST):
- Static Application Security Testing
- Runs Python security rules
- Checks for secrets in code
- Security audit patterns

**Secret Scan Job** (Gitleaks):
- Scans entire git history for leaked secrets
- Detects API keys, tokens, passwords
- Prevents accidental credential commits

**Dependency Review Job** (PRs only):
- Reviews new dependencies added in PRs
- Checks licenses (allows: MIT, Apache-2.0, BSD, ISC, PSF)
- Fails on moderate+ severity vulnerabilities

**Configuration**:
- `.bandit` - Bandit security scanner config
- `pyproject.toml` - Bandit settings (also)

**Running locally**:
```bash
# Run Bandit
bandit -r src/

# Run Safety
safety check

# Run Gitleaks (requires Docker)
docker run --rm -v $(pwd):/path zricethezav/gitleaks:latest detect --source=/path -v
```

---

### 4. Docker Workflow

**File**: `.github/workflows/docker.yml`

**Purpose**: Build, test, and publish Docker images

**Triggers**:
- Push to `main`, `develop`
- Version tags (`v*.*.*`)
- Pull requests to `main`, `develop` (build only, no push)

**What it does**:

**Build-and-Test Job**:
- Builds Docker image with multi-stage build
- Runs basic smoke test in container
- Pushes to GitHub Container Registry (ghcr.io)
- Tags images with:
  - Branch name (`main`, `develop`)
  - Git SHA (`main-abc123`)
  - Version tags (`v1.0.0`, `1.0`, `1`)
  - `latest` (for main branch only)

**Docker-Compose-Test Job**:
- Starts full stack (PostgreSQL + Redis + Bot)
- Waits for services to be healthy
- Runs health checks inside container
- Tears down stack

**Vulnerability-Scan Job**:
- Scans Docker image with Trivy
- Checks for OS and package vulnerabilities
- Reports CRITICAL and HIGH severity issues
- Uploads results to GitHub Security

**Image Tags**:
```
ghcr.io/lazydayz137/kalshi-ai-trading-bot:latest
ghcr.io/lazydayz137/kalshi-ai-trading-bot:main
ghcr.io/lazydayz137/kalshi-ai-trading-bot:develop
ghcr.io/lazydayz137/kalshi-ai-trading-bot:v1.0.0
ghcr.io/lazydayz137/kalshi-ai-trading-bot:main-abc1234
```

**Pulling images**:
```bash
# Pull latest from main
docker pull ghcr.io/lazydayz137/kalshi-ai-trading-bot:latest

# Pull specific version
docker pull ghcr.io/lazydayz137/kalshi-ai-trading-bot:v1.0.0

# Pull development version
docker pull ghcr.io/lazydayz137/kalshi-ai-trading-bot:develop
```

---

## Status Badges

Add these badges to your `README.md` to show CI/CD status:

### Tests
```markdown
![Tests](https://github.com/Lazydayz137/kalshi-ai-trading-bot/workflows/Tests/badge.svg)
```

### Code Quality
```markdown
![Code Quality](https://github.com/Lazydayz137/kalshi-ai-trading-bot/workflows/Code%20Quality/badge.svg)
```

### Security
```markdown
![Security](https://github.com/Lazydayz137/kalshi-ai-trading-bot/workflows/Security%20Scanning/badge.svg)
```

### Docker
```markdown
![Docker](https://github.com/Lazydayz137/kalshi-ai-trading-bot/workflows/Docker%20Build%20and%20Push/badge.svg)
```

### Code Coverage (if using Codecov)
```markdown
[![codecov](https://codecov.io/gh/Lazydayz137/kalshi-ai-trading-bot/branch/main/graph/badge.svg)](https://codecov.io/gh/Lazydayz137/kalshi-ai-trading-bot)
```

---

## Local Development

### Running All Checks Locally

Before pushing code, run all checks locally to catch issues early:

```bash
# 1. Format code
black src/ tests/
isort src/ tests/

# 2. Run linters
flake8 src/ tests/
mypy src/ --ignore-missing-imports
pylint src/

# 3. Run security scans
bandit -r src/
safety check

# 4. Run tests
pytest tests/unit/ -v
pytest tests/integration/ -v --cov=src

# 5. Build Docker image
docker build -t kalshi-trading-bot:local .
docker compose up -d
docker compose exec trading-bot python -m src.utils.health_check
```

### Pre-commit Hooks (Optional)

Install pre-commit hooks to automatically run checks before each commit:

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.6
    hooks:
      - id: bandit
        args: ['-r', 'src/']
EOF

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Troubleshooting

### Test Failures

**Problem**: Tests fail in CI but pass locally

**Solutions**:
1. **Python version mismatch**: Ensure you're using Python 3.12 locally
   ```bash
   python --version  # Should be 3.12.x
   ```

2. **Missing dependencies**: Install all test dependencies
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-asyncio pytest-cov
   ```

3. **Environment variables**: CI may have different env vars
   - Check if tests depend on `.env` file
   - Mock external dependencies properly

4. **Timezone issues**: CI runs in UTC
   ```python
   # Use UTC in tests
   from datetime import timezone
   now = datetime.now(timezone.utc)
   ```

---

### Code Quality Failures

**Problem**: Black formatting check fails

**Solution**:
```bash
# Auto-fix formatting
black src/ tests/

# Check what would change (dry run)
black --check --diff src/ tests/
```

**Problem**: Import sorting fails

**Solution**:
```bash
# Auto-fix imports
isort src/ tests/

# Check what would change
isort --check-only --diff src/ tests/
```

**Problem**: Flake8 errors

**Solution**:
```bash
# Run flake8 to see errors
flake8 src/ tests/

# Common fixes:
# - Remove unused imports
# - Fix indentation
# - Break long lines
# - Add blank lines between functions/classes
```

**Problem**: MyPy type errors

**Solution**:
```bash
# Run mypy
mypy src/ --ignore-missing-imports

# Common fixes:
# - Add type hints: def func(x: int) -> str:
# - Use Optional for nullable: Optional[str]
# - Import from typing: from typing import List, Dict, Optional
```

---

### Security Scan Failures

**Problem**: Bandit reports hardcoded secrets

**Solution**:
```python
# BAD
API_KEY = "sk-1234567890abcdef"

# GOOD
import os
API_KEY = os.getenv("API_KEY")
```

**Problem**: Safety reports vulnerable dependency

**Solution**:
```bash
# Check vulnerability details
safety check --full-report

# Upgrade vulnerable package
pip install --upgrade <package-name>

# Update requirements.txt
pip freeze > requirements.txt
```

**Problem**: Gitleaks detects secret in git history

**Solution**:
```bash
# If false positive, add to .gitleaksignore
echo "path/to/file:line_number" >> .gitleaksignore

# If real secret was committed:
# 1. Rotate the secret immediately
# 2. Use git-filter-repo or BFG to remove from history
# 3. Force push (DANGEROUS - coordinate with team)
```

---

### Docker Build Failures

**Problem**: Docker build fails with dependency errors

**Solution**:
1. **Check requirements.txt is up to date**
   ```bash
   pip freeze > requirements.txt
   ```

2. **Try building locally**
   ```bash
   docker build -t kalshi-trading-bot:debug .
   ```

3. **Check build logs for specific error**
   - Missing system dependencies? Add to Dockerfile
   - Python version issue? Update Dockerfile base image

**Problem**: Docker health check fails

**Solution**:
```bash
# Test health check locally
docker compose up -d
docker compose ps  # Check health status
docker compose logs trading-bot  # Check logs

# Run health check manually
docker compose exec trading-bot python -m src.utils.health_check
```

**Problem**: Can't pull Docker image

**Solution**:
1. **Authenticate with GitHub Container Registry**
   ```bash
   echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
   ```

2. **Check image exists**
   - Go to repository → Packages
   - Verify image was published

3. **Check permissions**
   - Image must be public OR you need read access

---

### Workflow Permissions

**Problem**: Workflow can't push to registry or create comments

**Solution**:

1. **Check repository settings**:
   - Settings → Actions → General
   - Workflow permissions: "Read and write permissions"
   - Allow GitHub Actions to create PRs: ✓

2. **Check GITHUB_TOKEN permissions in workflow**:
   ```yaml
   permissions:
     contents: read
     packages: write
     pull-requests: write
     security-events: write
   ```

---

## Configuration Files

### Summary of Config Files

| File | Purpose | Tools Configured |
|------|---------|------------------|
| `pyproject.toml` | Modern Python project config | Black, isort, pytest, coverage, mypy, bandit, pylint |
| `.flake8` | Flake8 linter config | Flake8 |
| `.pylintrc` | Pylint static analysis | Pylint |
| `.bandit` | Security scanner config | Bandit |
| `pytest.ini` | Test configuration (legacy) | pytest |

### Best Practices

1. **Keep config in pyproject.toml when possible**
   - Modern standard (PEP 518)
   - Single source of truth
   - Better tool support

2. **Use separate files when required**
   - Flake8 doesn't support pyproject.toml yet
   - Some tools prefer dedicated config files

3. **Document any deviations from standards**
   - If you disable a rule, add comment explaining why
   - Keep ignored rules to minimum

4. **Version control all config files**
   - Ensures consistent behavior across environments
   - Makes CI/CD reproducible

---

## GitHub Actions Secrets

Some workflows may require secrets to be configured in GitHub:

### Required Secrets

None required for basic functionality. All workflows use `GITHUB_TOKEN` which is automatically provided.

### Optional Secrets

| Secret | Purpose | Workflow |
|--------|---------|----------|
| `CODECOV_TOKEN` | Upload coverage to Codecov | Tests |
| `DOCKER_HUB_USERNAME` | Push to Docker Hub (alternative) | Docker |
| `DOCKER_HUB_TOKEN` | Docker Hub auth | Docker |
| `GITLEAKS_LICENSE` | Gitleaks for organizations | Security |

### Setting Secrets

1. Go to repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add name and value
4. Click "Add secret"

---

## Monitoring CI/CD

### GitHub Actions Dashboard

View all workflow runs:
1. Go to repository
2. Click "Actions" tab
3. See all workflows and their status

### Workflow Run Details

Click on any workflow run to see:
- **Summary**: Overall status and duration
- **Jobs**: Individual job status
- **Annotations**: Errors and warnings
- **Artifacts**: Downloadable test reports
- **Logs**: Full console output

### Notifications

Configure notifications for CI/CD failures:
1. GitHub → Settings → Notifications
2. Enable "Actions" notifications
3. Choose email or web notifications

---

## Maintenance

### Updating Workflows

When updating workflows:

1. **Test changes in a branch first**
   ```bash
   git checkout -b update-ci-workflow
   # Edit .github/workflows/test.yml
   git add .github/workflows/test.yml
   git commit -m "ci: update test workflow to include integration tests"
   git push -u origin update-ci-workflow
   ```

2. **Create PR and verify workflow runs**
   - Workflow will run on the PR
   - Check for any errors
   - Review changes with team

3. **Merge when green**
   - All checks pass ✓
   - Approved by reviewer
   - Merge to main

### Updating Dependencies

Keep CI/CD dependencies up to date:

```bash
# Update Python dependencies
pip install --upgrade pip
pip list --outdated
pip install --upgrade <package>
pip freeze > requirements.txt

# Update GitHub Actions
# Edit .github/workflows/*.yml
# Update action versions:
# actions/checkout@v4 → actions/checkout@v5
# actions/setup-python@v5 → actions/setup-python@v6
```

### Scheduled Maintenance

**Weekly**:
- Review security scan results
- Check for dependency updates
- Monitor workflow execution times

**Monthly**:
- Update GitHub Actions versions
- Review and update linter configs
- Check for new best practices

**Quarterly**:
- Full audit of all workflows
- Update Python version if needed
- Review and optimize Docker images

---

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Docker Build Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

**Last Updated**: 2025-11-16
