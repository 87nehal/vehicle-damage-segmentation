# Vehicle damage frontend

Next.js 16 App Router UI for the vehicle-damage development API. It uses
shadcn/ui and Tailwind CSS 4.

Install the Python serving dependencies once from the repository root, then
start the frontend:

```powershell
pip install -e ".[serve]"
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

`npm run dev` starts the hash-verified model API on port 8001 (or reuses an
already healthy instance), waits for the selected checkpoint to load, and then
starts Next.js. `next.config.ts` rewrites `/api/:path*` to
`http://127.0.0.1:8001/api/:path*`. The API never accepts a user-defined
threshold. Use `npm run dev:web` only when you intentionally manage the API in
a separate terminal with `vehicle-damage-api`.

| Route   | Page                                                                 |
| ------- | -------------------------------------------------------------------- |
| `/`     | Damage upload, segmentation overlay, and fail-closed safety decision |
| `/logs` | Legacy telemetry workspace; requires its separate API                |

The selected checkpoint is development-only and not production approved. The
UI preserves that warning and exposes manual-review and recapture outcomes.
