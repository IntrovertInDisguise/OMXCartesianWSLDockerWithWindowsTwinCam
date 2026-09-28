#!/usr/bin/env python3
"""
Append a timestamped record of recent work to project_status.md.

Usage:
  tools/record_success.py "It works!"    # appends entry with message
  tools/record_success.py                 # defaults to message 'It works!'

This gathers recent git commits and changed files and writes them
under a clearly marked section in project_status.md so runs that
bring the project to a working state can be recorded reproducibly.
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_STATUS = ROOT / 'project_status.md'

def git(cmd):
    return subprocess.check_output(['git'] + cmd, cwd=ROOT).decode('utf-8', errors='replace')

def recent_commits(n=20):
    try:
        out = git(['log', f'-n{n}', '--pretty=format:%h %ad %s (%an)', '--date=short'])
        return out.strip().splitlines()
    except Exception:
        return ['(git log unavailable)']

def recent_changed_files(n=50):
    try:
        out = git(['diff', '--name-only', f'HEAD~{n}..HEAD'])
        files = [l for l in out.splitlines() if l]
        return files or ['(no recent changed files)']
    except Exception:
        return ['(git diff unavailable)']

def append_record(message="It works!"):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    header = f"\n## Success record: {ts}\n"
    body = []
    body.append(f"**Message:** {message}\n")
    body.append("**Recent commits (most recent first):**\n")
    for c in recent_commits(20):
        body.append(f"- {c}\n")
    body.append("\n**Recent changed files (git diff HEAD~50..HEAD):**\n")
    for f in recent_changed_files(50):
        body.append(f"- {f}\n")
    body.append("\n---\n")

    if not PROJECT_STATUS.exists():
        PROJECT_STATUS.write_text('# Project status\n\n')

    with PROJECT_STATUS.open('a', encoding='utf-8') as fh:
        fh.write(header)
        fh.writelines(body)

    print(f"Appended success record to {PROJECT_STATUS}")

if __name__ == '__main__':
    msg = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'It works!'
    append_record(msg)
