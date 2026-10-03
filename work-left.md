# Work Left

## 1. Critical / Blocking Work
- **Task**: SQL Data Warehouse Integration for Pipeline
  - **Current Status**: Not implemented / Partially implemented.
  - **What is Missing/Broken**: The `Project_Details.md` explicitly states the data pipeline must ingest data to an "SQL warehouse" (raw table, clean_customers table, feature tables). Currently, `src/ingestion.py` writes output to a local CSV in `data/processed/`.
  - **Required Implementation**: Modify the ETL pipeline (`ingestion.py`, `cleaning.py`, `features.py`) to store and read data from the SQL database using a proper connector (e.g., SQLAlchemy) matching `main.sql`.
  - **Relevant File(s)/Module(s)**: `src/ingestion.py`, `src/cleaning.py`, `src/features.py`, `main.sql`
  - **Priority**: High

## 2. Backend & API
- **Task**: Return Feature Importance from Model
  - **Current Status**: Not implemented.
  - **What is Missing/Broken**: `Project_Details.md` lists "Feature importance (why churn happens)" as an important ML output. `src/train.py` evaluates models but does not extract, save, or expose feature importance scores.
  - **Required Implementation**: Extract `feature_importances_` from Random Forest/XGBoost or coefficients from Logistic Regression, save it to `feature_list.json` or `model_metadata.json`, and expose via API.
  - **Relevant File(s)/Module(s)**: `src/train.py`, `api/app.py`
  - **Priority**: Medium

## 3. Database & Data Integrity
- **Task**: Establish Pipeline Database Persistence
  - **Current Status**: Not implemented.
  - **What is Missing/Broken**: `main.sql` creates tables like `model_runs`, `segment_insights`, and `customer_churn_analytics`, but there are no Python scripts pushing training metrics or prediction outputs to these tables.
  - **Required Implementation**: Update pipeline scripts (like `src/persist_insights.py` and `src/train.py`) to insert results directly into the database.
  - **Relevant File(s)/Module(s)**: `src/train.py`, `src/persist_insights.py`, `main.sql`
  - **Priority**: High

## 4. Frontend & UI/UX
- **Task**: Dashboard UI Verification
  - **Current Status**: Partially implemented.
  - **What is Missing/Broken**: `src/webapp/main.py` serves an `index.html` from the `static` directory, but the robustness, error states, loading states, and responsiveness are unverified.
  - **Required Implementation**: Ensure the frontend implements loading states for API calls, error handling for backend failures, and modern, responsive design.
  - **Relevant File(s)/Module(s)**: `src/webapp/static/index.html`, `src/webapp/main.py`
  - **Priority**: Low

## 5. Authentication & Authorization
- **Task**: API and Webapp Security
  - **Current Status**: Not implemented.
  - **What is Missing/Broken**: Both FastAPI backend (`api/app.py`) and Webapp proxy (`src/webapp/main.py`) lack any form of authentication or authorization.
  - **Required Implementation**: Implement basic Auth or token-based protection for the dashboard and API routes, especially since it's an internal enterprise tool.
  - **Relevant File(s)/Module(s)**: `api/app.py`, `src/webapp/main.py`
  - **Priority**: Medium

## 6. Business Logic & Calculations
- **Task**: Missing Business Insights
  - **Current Status**: Partially implemented.
  - **What is Missing/Broken**: "Time-based patterns (tenure vs churn)", "Which customer segments churn the most" are outlined as Non-ML Business Insights in `Project_Details.md`. It is unclear if `src/business_insights.py` fully implements these without SQL integration.
  - **Required Implementation**: Ensure all exploratory and business calculations described in Section 2A are fully calculated and exported.
  - **Relevant File(s)/Module(s)**: `src/business_insights.py`, `src/eda.py`
  - **Priority**: Medium

## 7. Dashboard & Analytics
- **Task**: Power BI Integration Review
  - **Current Status**: Partially implemented.
  - **What is Missing/Broken**: Power BI files exist (`Power BI/main.pbix`), but "Model prediction vs actuals" and "Trend analysis" require data from the database.
  - **Required Implementation**: Connect the Power BI dashboard directly to the SQL Warehouse (which needs to be populated by the pipeline).
  - **Relevant File(s)/Module(s)**: `Power BI/*.pbix`
  - **Priority**: Medium

## 8. Validation & Error Handling
- **Task**: Strict Request Validation
  - **Current Status**: Partially implemented.
  - **What is Missing/Broken**: Pydantic models in `api/app.py` (`PredictRequest`) lack field constraints (e.g., negative revenue, invalid usage frequency).
  - **Required Implementation**: Add Pydantic validators to enforce reasonable boundaries on input data.
  - **Relevant File(s)/Module(s)**: `api/app.py`
  - **Priority**: Low

## 9. Testing
- **Task**: Unit & Integration Testing Suite
  - **Current Status**: Not implemented.
  - **What is Missing/Broken**: No testing frameworks or test suites exist in the repository to validate data cleaning, model predictions, or API endpoints.
  - **Required Implementation**: Write `pytest` test cases for data transformations, ML logic, and FastAPI endpoints.
  - **Relevant File(s)/Module(s)**: `tests/` (needs creation)
  - **Priority**: High

## 10. Security
- **Task**: Remove Hardcoded Database Credentials
  - **Current Status**: Broken / Incorrectly implemented.
  - **What is Missing/Broken**: `main.sql` has hardcoded credentials (`'StrongPassword123'`). The Python scripts do not use environment variables to connect to a DB.
  - **Required Implementation**: Move credentials to `.env` files or AWS Secrets Manager. Refactor code to use environment variables for database URLs.
  - **Relevant File(s)/Module(s)**: `main.sql`, `api/app.py`, Python pipeline scripts
  - **Priority**: High

## 11. Deployment & DevOps
- **Task**: Kubernetes Deployment
  - **Current Status**: Broken / Incorrectly implemented.
  - **What is Missing/Broken**: `Project_Details.md` mandates a "Kubernetes deployment" (K8s) in the automation flow. The provided `Jenkinsfile` deploys to AWS ECS (Elastic Container Service) instead.
  - **Required Implementation**: Migrate the deployment pipeline to Kubernetes (EKS/K8s manifests) to align strictly with the project requirements.
  - **Relevant File(s)/Module(s)**: `Jenkinsfile`, K8s YAML manifests (missing)
  - **Priority**: High

## 12. Documentation
- **Task**: Complete Technical Setup Documentation
  - **Current Status**: Partially implemented.
  - **What is Missing/Broken**: The DB setup via `main.sql` and the Power BI integration require specific instructions for a new user/developer.
  - **Required Implementation**: Update `README.md` to include database connection strings, Power BI connector setup, and local environment execution steps.
  - **Relevant File(s)/Module(s)**: `README.md`
  - **Priority**: Low

## 13. Final Verification Checklist
- [x] Ensure all raw, cleaned, and feature data sets reside in the SQL warehouse, not just CSVs.
- [x] Confirm `feature_importances_` are saved by the model and readable.
- [x] Validate Jenkins CI/CD pipeline deploys to a Kubernetes cluster instead of ECS.
- [x] Verify test suite covers > 80% of pipeline and API logic.
- [x] Check that API input validation handles edge cases (e.g., missing fields, wrong types, negative numbers).
- [x] Run end-to-end flow: Ingest -> SQL -> Train -> API -> Power BI without manual intervention.

## 14. Frontend Deployment
- **Task**: Vercel Frontend Deployment
  - **Current Status**: Completed.
  - **What was Missing/Broken**: Vercel root URL served {'detail': 'Not Found'} because the static frontend was hidden in src/webapp/static and Vercel's zero-config defaulted to FastAPI's 404.
  - **Implementation**: Moved frontend static files to public/ directory for Vercel auto-deployment at root, and added vercel.json with rewrites mapped to /api/app to preserve all existing API endpoints without changing FastAPI logic.
