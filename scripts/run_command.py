"""Команды Makefile на Windows: Docker Desktop и Git Bash без правки PATH."""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit('Нужна команда: docker, bash или python')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    docker = shutil.which('docker')
    bash = shutil.which('bash')
    if os.name == 'nt':
        local = Path(os.environ['LOCALAPPDATA'])
        for path in [local / 'Programs/DockerDesktop/resources/bin/docker.exe',
                     Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Docker/Docker/resources/bin/docker.exe']:
            if path.is_file():
                docker = str(path)
                break
        git = shutil.which('git')
        candidates = [Path(git).resolve().parents[1] / 'bin/bash.exe'] if git else []
        candidates += [Path('C:/Program Files/Git/bin/bash.exe'), Path('D:/downloads_programs/Git/bin/bash.exe')]
        bash = next((str(path) for path in candidates if path.is_file()), None)
    if docker:
        env['PATH'] = str(Path(docker).parent) + os.pathsep + env.get('PATH', '')
    command = {'python': sys.executable, 'docker': docker, 'bash': bash}.get(args[0])
    if not command:
        raise SystemExit(f'Не найдена команда {args[0]}: проверьте установку Docker Desktop / Git Bash')
    return subprocess.call([command, *args[1:]], env=env)


if __name__ == '__main__':
    sys.exit(main())
