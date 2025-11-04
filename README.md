# Job Market Analysis Using Machine Learning and Visualization

A complete, reproducible pipeline to analyze job listings, predict salary trends, and prepare insights for interactive dashboards. The project applies Machine Learning (Logistic Regression, XGBoost) to understand job market patterns such as salary trends, top-paying locations, role demand, and company-level insights. It fulfills MSc ITM Practical Assignment Tasks 1–4 (Data Preprocessing & EDA, Modeling, Tableau Export, and Prescriptive Insights).

---

## 1. Project Overview
- Uses ML models (Logistic Regression for job-type classification, XGBoost/RandomForest for salary regression) to derive job market insights.
- Cleans and standardizes salary formats, handles missing values, and extracts simple text features from job descriptions.
- Produces evaluation metrics, saved figures (confusion matrix, feature importance), and a Tableau-ready CSV for dashboards.
- Satisfies MSc ITM Practical Assignment Tasks 1–4.

---

## 2. Folder Structure
```
JobMarketAnalysis/
│
├── data/                # Original and cleaned datasets
├── notebooks/           # EDA and preprocessing notebook
├── models/              # Saved ML models (.pkl)
├── tableau_exports/     # Tableau-ready summary CSV
├── figures/             # Saved charts (confusion matrix, feature importance)
├── main.py              # Main script for full pipeline
└── requirements.txt     # Dependencies
```

Key files:
- `data/job_listings.csv` — place the original dataset here.
- `data/job_listings_clean.csv` — auto-generated cleaned dataset.
- `notebooks/EDA_and_Preprocessing.ipynb` — interactive EDA and cleaning.

---

## 3. Installation Instructions
From the `JobMarketAnalysis` folder:

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Usage
Run the full pipeline (cleaning → modeling → evaluation → exports → insights):

```bash
# Run full pipeline
python main.py
```

Optional: open the EDA notebook to explore data and regenerate the cleaned CSV:

```bash
# Optional: run EDA notebook
jupyter notebook notebooks/EDA_and_Preprocessing.ipynb
```

The script will:
- Clean and prepare the dataset
- Train models (Logistic Regression, XGBoost/RandomForest)
- Save trained models to `/models`
- Generate visual outputs to `/figures` and a Tableau export file to `/tableau_exports`
- Print actionable insights in the terminal

Note: Ensure `data/job_listings.csv` is present before running.

---

## 5. Outputs
Expected outputs and example paths:
- `/models/model_salary.pkl`
- `/models/model_jobtype.pkl`
- `/figures/confusion_matrix_jobtype.png`
- `/figures/feature_importance_salary.png`
- `/tableau_exports/job_insights.csv`

Example console output snippet:

```
Actionable Insights:
- Top-paying locations: San Francisco, CA, New York, NY
- Companies with high rating and pay correlation: OpenAI, Netflix
- Regression model performance: RMSE=42186, R²=-0.14
```

---

## 6. Tableau Visualization
Build interactive dashboards in Tableau using the export:
1. Open Tableau Desktop (or Tableau Public).
2. Connect to a text file and import `tableau_exports/job_insights.csv`.
3. Create dashboards such as:
   - Salary by Location
   - Rating vs Salary
   - Job Type Distribution
4. Add filters for `location`, `jobType`, and `company` for interactivity.
5. Include screenshots of dashboards in your report for submission.

---

## 7. Project Insights and Recommendations
- Tech hubs like San Francisco and New York show the highest median pay.
- Positive correlation often appears between company rating and salary levels.
- Demand for data-centric roles (AI, ML, Cloud) remains strong across regions.
- Recruiters can use insights to optimize salary bands by location and role.
- Companies with both high ratings and high average salaries can signal competitive compensation and culture.

---

## 8. References
- Scikit-learn documentation: https://scikit-learn.org/stable/
- XGBoost library: https://xgboost.readthedocs.io/
- Tableau visual analytics documentation: https://help.tableau.com/
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied Logistic Regression (3rd ed.). Wiley.
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.

---

## 9. Author
- Author: Praveen Chitteti
- Course: MSc Information Technology Management
- Module: Machine Learning and Visualization
- Date: October 2025

---

If you encounter environment conflicts, ensure your active Python comes from the project’s virtual environment (e.g., `.venv\\Scripts\\python` on Windows) before running `main.py`.
