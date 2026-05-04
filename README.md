# Corporate_finance_ML_decisions

pyenv install 3.12.8
pyenv virtualenv 3.12.8 cfml-capstone
pyenv local cfml-capstone
pyenv activate cfml-capstone


Install notebook dependencies
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn plotly scipy statsmodels imbalanced-learn ipykernel

Register a Jupyter kernel for VS Code
python -m ipykernel install --user --name cfml-capstone --display-name "Python (cfml-capstone)"




User Request → API/Webhook → Auth → Agent Orchestrator (LLM + Rules + Memory) → Intent Classification → Draft/Reply Decision → Tool Calls (Gmail/Outlook, CRM, Search, Calendar) → Human Approval (optional)
Send Email → Queue/Workers → Logging/Monitoring → Analytics/Feedback Loop → Model/Prompt Updatesvi 