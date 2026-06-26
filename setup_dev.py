#!/usr/bin/env python3
"""
AI Hub Academy — Development Setup Script
==========================================

Run once after cloning to get a fully working local environment:

    python setup_dev.py

What this script does (all automatic):
  1. Checks Python version (3.12+ required)
  2. Creates a virtual environment in ./venv/
  3. Installs all packages from requirements.txt
  4. Creates .env from .one-env.example (generates a real SECRET_KEY)
  5. Runs database migrations
  6. Seeds tutorial and training data
  7. Imports documentation from docs_source/
  8. Creates an admin superuser with a generated password

After it finishes, activate the venv and run the server:
    Windows  :  venv\\Scripts\\activate  &&  python manage.py runserver
    macOS/Linux:  source venv/bin/activate  &&  python manage.py runserver

Then open http://localhost:8000/
"""

import os
import platform
import secrets
import subprocess
import sys
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────
REQUIRED_PYTHON = (3, 12)
VENV_DIR = Path("venv")
ENV_FILE = Path(".env")
ENV_EXAMPLE = Path(".one-env.example")
ADMIN_USERNAME = "admin"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = os.environ.get("DJANGO_SUPERUSER_PASSWORD") or secrets.token_urlsafe(18)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    line = "=" * 58
    print(f"\n{line}\n  {text}\n{line}")


def _step(msg: str) -> None:
    print(f"\n[+] {msg} ...")


def _ok(msg: str) -> None:
    print(f"    ok  {msg}")


def _warn(msg: str) -> None:
    print(f"    --  {msg}")


def _abort(msg: str) -> None:
    print(f"\n[!] {msg}", file=sys.stderr)
    sys.exit(1)


def _run(cmd: list, *, env: dict | None = None, allow_failure: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if not allow_failure and result.returncode != 0:
        _abort(
            f"Command failed: {' '.join(str(c) for c in cmd)}\n"
            + (result.stdout[-1500:] if result.stdout else "")
            + (result.stderr[-1500:] if result.stderr else "")
        )
    return result


def _venv_python() -> Path:
    """Return the path to the Python executable inside the venv."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _django_env() -> dict:
    """Build an env dict with DJANGO_SETTINGS_MODULE set."""
    return {**os.environ, "DJANGO_SETTINGS_MODULE": "_core.settings"}


def _manage(python: Path, *args, extra_env: dict | None = None, allow_failure: bool = False) -> subprocess.CompletedProcess:
    env = _django_env()
    if extra_env:
        env.update(extra_env)
    return _run([str(python), "manage.py", *args], env=env, allow_failure=allow_failure)


# ── Steps ─────────────────────────────────────────────────────────────────────

def check_python() -> None:
    _step("Checking Python version")
    v = sys.version_info
    if (v.major, v.minor) < REQUIRED_PYTHON:
        _abort(
            f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ is required. "
            f"You have Python {v.major}.{v.minor}.\n"
            "Download it from https://www.python.org/downloads/"
        )
    _ok(f"Python {v.major}.{v.minor}.{v.micro}")


def create_venv() -> Path:
    _step("Creating virtual environment")
    python = _venv_python()
    if python.exists():
        _ok(f"venv already exists at {VENV_DIR}/")
    else:
        _run([sys.executable, "-m", "venv", str(VENV_DIR)])
        _ok(f"Created {VENV_DIR}/")
    return python


def install_deps(python: Path) -> None:
    _step("Installing dependencies from requirements.txt\n (This will take a while. If you go for a coffee, bring me one too!)")
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    _run([str(python), "-m", "pip", "install",
         "-r", "requirements.txt", "--quiet"])
    _ok("All packages installed")


def setup_env() -> None:
    _step("Setting up .env")
    if ENV_FILE.exists():
        _ok(".env already exists — keeping it as-is")
        return
    if not ENV_EXAMPLE.exists():
        _abort(f"{ENV_EXAMPLE} not found. Cannot create .env.")
    content = ENV_EXAMPLE.read_text(encoding="utf-8")
    # Replace the placeholder with a real random key
    secret = secrets.token_hex(50)
    content = content.replace("your-secret-key-here", secret)
    ENV_FILE.write_text(content, encoding="utf-8")
    _ok(".env created with a freshly generated SECRET_KEY")


def run_migrations(python: Path) -> None:
    _step("Running database migrations")
    _manage(python, "migrate")
    _ok("Database is up to date")


def seed_data(python: Path) -> None:
    _step("Seeding training data (providers, agents, tutorials, missions)")
    _manage(python, "seed_academy_training_data")
    _ok("Training data ready")


def import_docs(python: Path) -> None:
    _step("Importing documentation from docs_source/")
    _manage(python, "import_academy_docs")
    _ok("Documentation imported into database")


def create_admin(python: Path) -> None:
    _step(f"Creating admin superuser ({ADMIN_USERNAME})")
    result = _manage(
        python,
        "createsuperuser",
        "--noinput",
        f"--username={ADMIN_USERNAME}",
        f"--email={ADMIN_EMAIL}",
        extra_env={"DJANGO_SUPERUSER_PASSWORD": ADMIN_PASSWORD},
        allow_failure=True,
    )
    output = (result.stdout + result.stderr).lower()
    if result.returncode == 0:
        _ok(f"Admin user created:  {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
    elif "already exists" in output:
        _ok(f"Admin user '{ADMIN_USERNAME}' already exists — skipping")
    else:
        _warn(f"Could not create superuser automatically. Run: python manage.py createsuperuser")


def print_summary() -> None:
    _banner("Setup complete!")
    if platform.system() == "Windows":
        activate_cmd = r"venv\Scripts\activate"
    else:
        activate_cmd = "source venv/bin/activate"

    print(f"""
  To start the development server:

    {activate_cmd}
    python manage.py runserver

  Then open your browser:

    http://localhost:8000/            Academy home
    http://localhost:8000/dashboard/  Visual dashboard
    http://localhost:8000/admin/      Django admin

  Admin login:   {ADMIN_USERNAME} / {ADMIN_PASSWORD}
  Change the password after your first login.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _banner("AI Hub Academy — Dev Setup")
    print("  This script sets up a local development environment.")
    print(f"  Running on: {platform.system()} {platform.machine()}")

    # Guard: must run from the project root
    if not Path("manage.py").exists():
        _abort(
            "manage.py not found in the current directory.\n"
            "Run this script from the project root:\n"
            "    cd ai_hub-academy_app\n"
            "    python setup_dev.py"
        )

    check_python()
    python = create_venv()
    install_deps(python)
    setup_env()
    run_migrations(python)
    seed_data(python)
    import_docs(python)
    create_admin(python)
    print_summary()


if __name__ == "__main__":
    main()
