# Free demo deployment: GitHub + Render

Training stays on your computer. The hosted application only loads the selected saved model and answers predictions. No AWS, Docker Hub account, MLflow server, or deployment token is required for this route.

## What is packaged

`src/serving/bundle/` contains an XGBoost native model and a manifest with its source MLflow run, feature order, threshold, metrics, and checksum. The selected run is `02266b2be48c4d9ebac114c1632dd273`, threshold `0.35`. The app serves this bundle, never an automatically chosen "latest" run. Include both files in GitHub.

The API requires all 19 customer fields, including `SeniorCitizen` (0 or 1). `/predict` returns a label, `churn_probability`, and `threshold`. Invalid inputs receive HTTP 422. The UI is at `/ui`; `/` is a health check.

## 1. Test locally

From the project root, with Docker Desktop's Linux engine running:

```powershell
docker build -f Dockerfile -t telco-churn-app .
docker run --rm --name telco-churn --memory=512m --memory-swap=512m -e PORT=8000 -p 8000:8000 telco-churn-app
```

Keep that terminal open. Open http://127.0.0.1:8000/ui in your browser. In a second terminal:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py
docker stats --no-stream telco-churn
```

This checks health, UI, API docs and an actual model prediction. The memory limit approximates the free service's RAM allowance, not its CPU speed or full hosting environment.

Without Docker, use a separate serving environment so training dependencies remain unchanged:

```powershell
.\.venv\Scripts\python.exe -m venv .venv-serving
.\.venv-serving\Scripts\python.exe -m pip install -r requirements-serving.txt
.\.venv-serving\Scripts\python.exe -m src.app.serve
```

Use this serving environment for the app; the older Gradio version in the training environment is not the deployment version.

## 2. Upload to your own GitHub repository

The existing remote points to `anesriad/Telco-Customer-Churn-ML`. If that is not your account, preserve it as `upstream` and use your own repository as `origin`. Check the original license/permission and retain attribution before publishing someone else's code.

Create an empty GitHub repository without initializing a new README. Then, only if `origin` is still the original repository:

```powershell
git remote rename origin upstream
git remote add origin https://github.com/YOUR_USERNAME/Telco-Customer-Churn-ML.git
git remote -v
git status --short
```

Review the changes and stage the intended files in VS Code Source Control. Include `Dockerfile`, `requirements-serving.txt`, `render.yaml`, `.github/workflows/ci.yml`, the serving code, tests, scripts, documentation, and BOTH files under `src/serving/bundle/`. Review your notebook output before publishing it. Do not add `.venv*`, credentials, data, `mlruns`, or cache directories. `.codebase-memory/` is optional and not needed for hosting.

Commit the reviewed files, then push your current `main` branch:

```powershell
git commit -m "Prepare churn demo for Docker and Render"
git push -u origin main
```

No force push is needed. If a destination repository already has history, use a fork or reconcile it before pushing rather than overwriting it.

## 3. Wait for GitHub Actions

Open the repository's **Actions** tab. **Test and build deployment** should pass. It runs regression tests, builds the Linux Docker image, starts it with `PORT=10000` and a 512 MB memory limit, then checks health, UI, docs, and inference. Memory usage and container logs are recorded in the job output.

Tests run without the original CSV or `mlruns`. The model is not retrained in CI. The workflow has read-only repository permissions and does not push images to Docker Hub.

For later changes: create a branch, push it, open a pull request, check CI, then merge to `main`.

## 4. Create the free Render service

Sign in to Render and connect your GitHub account. Choose **New > Blueprint**, select your repository, and review `render.yaml`. It specifies one Docker web service with `plan: free`, health path `/`, and automatic deploys after CI passes. Confirm the service is Free before creating it; do not add a paid disk or database.

Alternatively choose **New > Web Service** with the connected repository, Docker runtime, branch `main`, Dockerfile path `./Dockerfile`, Free instance, health path `/`, and **After CI Checks Pass** auto-deploy. A connected Git provider is needed for this automatic deployment flow.

Once the build finishes, open the Render service URL and append `/ui`. Use `/docs` for the interactive API. New successful commits to `main` will deploy automatically. No Docker Hub secrets or Render API keys are needed for Git-connected deployment.

The free instance has limited RAM and CPU and sleeps after inactivity. Its writable filesystem is temporary. Keep experiments and training artifacts locally; the model bundled in the image survives redeployment. Free allowances and terms can change: check bandwidth/build usage and billing settings in Render. If the image exceeds the free limit, stop and reduce memory use instead of selecting a paid plan.

## 5. Deliberately update the deployed model

Train normally with `.venv`, choose a successful run in MLflow, then export that specific run:

```powershell
.\.venv\Scripts\python.exe scripts/export_serving_model.py --run-id YOUR_FINISHED_RUN_ID
```

Review the updated manifest, rerun local tests, and commit BOTH bundle files. A model trained with different feature definitions requires matching serving code and tests; export checks feature names but does not infer arbitrary new preprocessing.

## Reference

- Render free limits: https://render.com/docs/free
- Blueprint fields: https://render.com/docs/blueprint-spec
- CI-gated deployments: https://render.com/docs/deploys
- XGBoost native model format: https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html

Creating the GitHub repository and Render service is a separate account action. Preparing these files does not publish or deploy the project.
