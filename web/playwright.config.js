import { defineConfig } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const webDir=path.dirname(fileURLToPath(import.meta.url));
const python=process.platform==='win32'?'../.venv/Scripts/python.exe':'../.venv/bin/python';
export default defineConfig({
  testDir: './tests',
  workers: 1,
  // Keep failures inspectable through GitHub check annotations when log-storage
  // redirects are unavailable in a restricted development environment.
  reporter: process.env.CI ? [['list'], ['github']] : 'list',
  use: {
    baseURL: 'http://127.0.0.1:8000',
    launchOptions: {
      executablePath: process.env.CHROMIUM_PATH || (fs.existsSync('/usr/bin/chromium')?'/usr/bin/chromium':undefined),
      args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    },
  },
  webServer: {
    command: `${python} -m uvicorn robopup.server:app --host 127.0.0.1 --port 8000`,
    env: {MPLCONFIGDIR:path.resolve(webDir,'../.cache/matplotlib'),XDG_CACHE_HOME:path.resolve(webDir,'../.cache')},
    url: 'http://127.0.0.1:8000/api/health',
    reuseExistingServer: true,
  },
});
