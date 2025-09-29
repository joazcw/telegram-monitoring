#!/usr/bin/env python3
import subprocess
import sys
import os


def run_command(cmd):
    """Run shell command and handle errors"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/migrate.py [upgrade|downgrade|current|history]")
        sys.exit(1)

    action = sys.argv[1]

    # Set environment for migrations
    os.environ.setdefault('PYTHONPATH', '.')

    if action == 'upgrade':
        print("Running database migrations...")
        if run_command("alembic upgrade head"):
            print("✅ Database migrations completed successfully")
        else:
            print("❌ Migration failed")
            sys.exit(1)

    elif action == 'downgrade':
        revision = sys.argv[2] if len(sys.argv) > 2 else '-1'
        print(f"Downgrading to revision: {revision}")
        run_command(f"alembic downgrade {revision}")

    elif action == 'current':
        print("Current database revision:")
        run_command("alembic current")

    elif action == 'history':
        print("Migration history:")
        run_command("alembic history --verbose")

    else:
        print(f"Unknown action: {action}")
        print("Usage: python scripts/migrate.py [upgrade|downgrade|current|history]")
        sys.exit(1)


if __name__ == "__main__":
    main()