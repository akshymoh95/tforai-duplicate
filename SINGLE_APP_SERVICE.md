# Single App Service via custom container

This repo now supports running **both Next.js (SSR)** and the **FastAPI API** in a single Azure App Service by using a custom container.

## How it works
- The Docker image installs Python + Node.
- Builds the Next.js app at image build time.
- Runs **two processes** in one container via `supervisord`:
  - API: `uvicorn backend.main:app --port 8000`
  - Web: `npm run start` (port 3000)

## Azure App Service (Linux) settings
- **Container port**: set to `3000` (Web app)
- API runs on `8000` internally.

## Environment variables
Set these in App Service -> Configuration -> Application settings:

### API (required)
SNOWFLAKE_ACCOUNT
SNOWFLAKE_USER
SNOWFLAKE_ROLE
SNOWFLAKE_WAREHOUSE
SNOWFLAKE_DATABASE
SNOWFLAKE_SCHEMA
SNOWFLAKE_PASSWORD
SNOWFLAKE_AUTHENTICATOR (if required)
SNOWFLAKE_PASSCODE (if required)

### RAI config
RAI_CONFIG_FILE=/app/rai_config/raiconfig.toml
RAI_PROFILE=<profile name in raiconfig.toml>

### Web
NEXT_PUBLIC_API_URL=https://<your-app>.azurewebsites.net

## Important
- **Single App Service means single public URL**.
- The UI will call the API at the same host via `NEXT_PUBLIC_API_URL`.
- CORS in `backend/main.py` should include this same host.

## Build & Deploy (high level)
- Build and push the Docker image to ACR or Docker Hub.
- Configure App Service to use that image.

If you want, I can add a GitHub Actions workflow to build/push the image once you provide the registry info.
