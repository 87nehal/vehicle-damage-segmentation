# Vehicle damage inspection bundle

This folder contains the FastAPI backend, the hash-verified model artifacts, and the Next.js frontend in one place.

The directory is named `prod` as requested. The selected model is still marked **development-only** by the API; it is not a production approval or a guarantee of zero false positives/negatives.

## First-time setup

From PowerShell in this folder:

```powershell
cd backend
python -m pip install -e ".[serve]"
cd ..\frontend
npm install
cd ..
```

The model checkpoint is included in `backend\runs` (about 95 MB). No model download is needed.

## Start backend and frontend together

```powershell
.\start.ps1
```

Then open <http://localhost:3000>. The frontend proxies `/api/*` to the FastAPI service at <http://127.0.0.1:8001>.

The launcher writes backend logs to `logs\backend.out.log` and `logs\backend.err.log` and stops the backend it started when the frontend exits.

## Run services separately

Backend:

```powershell
cd backend
.\run_backend.ps1
```

Frontend (in a second terminal):

```powershell
cd frontend
npm run dev:web
```

Health check: <http://127.0.0.1:8001/api/health>

## Included model

`backend\runs\SELECTED_DEVELOPMENT_MODEL.json` records the selected checkpoint and calibration profile, including SHA-256 hashes. The current candidate is tuned with clean scenes and hard negatives for handles, fuel caps, body styling, reflections, and low-resolution images. It still needs an untouched clean-inclusive evaluation and formal release audit before production use.
