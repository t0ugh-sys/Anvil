#!/usr/bin/env node
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const cp = require('child_process');

const PACKAGE_ROOT = path.resolve(__dirname, '..');
const BRIDGE_ROOT = path.join(os.homedir(), '.loopagent', 'npm-bridge');
const VENV_DIR = path.join(BRIDGE_ROOT, 'venv');
const MARKER_FILE = path.join(BRIDGE_ROOT, 'install.json');

function run(cmd, args, options = {}) {
  const result = cp.spawnSync(cmd, args, {
    stdio: options.stdio || 'pipe',
    encoding: 'utf8',
    shell: options.shell || false
  });
  return result;
}

function parseVersion(text) {
  const match = String(text || '').trim().match(/^(\d+)\.(\d+)/);
  if (!match) return null;
  return { major: Number(match[1]), minor: Number(match[2]) };
}

function isSupportedVersion(version) {
  if (!version) return false;
  return version.major > 3 || (version.major === 3 && version.minor >= 11);
}

function detectPython() {
  const candidates = [];
  if (process.env.LOOPAGENT_PYTHON) {
    candidates.push({ cmd: process.env.LOOPAGENT_PYTHON, args: [], shell: true });
  }
  candidates.push({ cmd: 'python', args: [], shell: false });
  candidates.push({ cmd: 'python3', args: [], shell: false });
  candidates.push({ cmd: 'py', args: ['-3.11'], shell: false });
  candidates.push({ cmd: 'py', args: ['-3'], shell: false });

  for (const candidate of candidates) {
    const probe = run(
      candidate.cmd,
      [...candidate.args, '-c', 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")'],
      { shell: candidate.shell }
    );
    if (probe.status !== 0) continue;
    const version = parseVersion(probe.stdout);
    if (!isSupportedVersion(version)) continue;
    return candidate;
  }
  return null;
}

function readPackageVersion() {
  const file = path.join(PACKAGE_ROOT, 'package.json');
  const data = JSON.parse(fs.readFileSync(file, 'utf8'));
  return String(data.version || '0.0.0');
}

function shouldInstall() {
  if (process.env.LOOPAGENT_FORCE_REINSTALL === '1') return true;
  if (!fs.existsSync(path.join(VENV_DIR, 'pyvenv.cfg'))) return true;
  if (!fs.existsSync(MARKER_FILE)) return true;
  try {
    const marker = JSON.parse(fs.readFileSync(MARKER_FILE, 'utf8'));
    return marker.version !== readPackageVersion();
  } catch (_) {
    return true;
  }
}

function ensureBridgeDir() {
  fs.mkdirSync(BRIDGE_ROOT, { recursive: true });
}

function venvPythonPath() {
  if (process.platform === 'win32') {
    return path.join(VENV_DIR, 'Scripts', 'python.exe');
  }
  return path.join(VENV_DIR, 'bin', 'python');
}

function installPythonRuntime(pythonCandidate) {
  ensureBridgeDir();
  if (!fs.existsSync(path.join(VENV_DIR, 'pyvenv.cfg'))) {
    const mk = run(
      pythonCandidate.cmd,
      [...pythonCandidate.args, '-m', 'venv', VENV_DIR],
      { stdio: 'inherit', shell: pythonCandidate.shell }
    );
    if (mk.status !== 0) {
      process.exit(mk.status || 1);
    }
  }

  const py = venvPythonPath();
  const upgradePip = run(py, ['-m', 'pip', 'install', '--upgrade', 'pip'], { stdio: 'inherit' });
  if (upgradePip.status !== 0) process.exit(upgradePip.status || 1);

  const install = run(py, ['-m', 'pip', 'install', '--upgrade', PACKAGE_ROOT], { stdio: 'inherit' });
  if (install.status !== 0) process.exit(install.status || 1);

  fs.writeFileSync(
    MARKER_FILE,
    JSON.stringify(
      {
        version: readPackageVersion()
      },
      null,
      2
    ),
    'utf8'
  );
}

function ensureRuntime() {
  const pythonCandidate = detectPython();
  if (!pythonCandidate) {
    console.error('LoopAgent requires Python 3.11+ (set LOOPAGENT_PYTHON if needed).');
    process.exit(1);
  }
  if (shouldInstall()) {
    installPythonRuntime(pythonCandidate);
  }
}

function runLoopAgentCLI(userArgs) {
  const py = venvPythonPath();
  const args = ['-m', 'loop_agent.agent_cli', ...userArgs];
  const child = cp.spawn(py, args, { stdio: 'inherit' });
  child.on('exit', (code) => process.exit(code === null ? 1 : code));
}

function main() {
  ensureRuntime();
  runLoopAgentCLI(process.argv.slice(2));
}

main();
