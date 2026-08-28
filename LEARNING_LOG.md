# Learning Log

## Day 0

- What I built: Set up Python, VS Code, Git, GitHub repository, virtual environment, project structure, and ServiceNow PDI.
- What broke and why:Nothing significant broke during the setup. I was able to complete the setup steps successfully.
- One question I could not answer:How do Git branches work, and when should I create a new branch?
### Day 1

- What I built: Created a Python package and learned how to use modules, functions, and file handling.
- What I learned: Learned how Python files can be organized and imported from different files.
- One question I could not answer: How do Python packages and modules work together internally?


### Day 2

- What I built: Created ticket utility functions and tested them using Python.
- What I learned: Learned about reading data, processing ticket information, and using Git to track and commit changes.
- One question I could not answer: When should I create a new Git branch?


### Day 3

- What I built: Created a 50,000-record incident dataset using Python, NumPy, and Pandas.
- What I learned: Learned how to create DataFrames, read CSV files, analyze data, and check SLA breaches.
- One question I could not answer: How can I identify useful patterns from a large dataset?
### Day 4

- What I built: Practiced API calls using Python requests and created APIs using FastAPI.
- What I learned: GET, POST, PUT, DELETE, query parameters, pagination, external API calls, and error handling.
- What I broke and why: Faced issues with requests import and FastAPI server setup, then fixed them.
- One question I could not answer: How should API authentication and API keys be handled securely?
## Day 5

### What I built
Practiced environment variables, `.env` files, `.env.example`,
configuration handling, and dependency management.

### What I learned
- How to use `requirements.txt`
- How `.gitignore` protects files from being committed
- How to load values from `.env`
- How to use `os.getenv()`
- How to provide default configuration values
- How to convert environment variables into Boolean values
- Why `.env.example` is useful

### What broke and why
I initially faced an issue while running the Python file because
I used the wrong file path. I corrected the command based on my
current folder location.

### One question I could not answer
How are environment variables securely managed in production?
## Day 6

### Baseline experiment

I created a baseline that predicts "No SLA breach" for every incident.

The dataset contains 50,000 incidents:
- 28,260 were not SLA breaches.
- 21,740 were SLA breaches.
- The baseline achieved 56.52% accuracy.

However, accuracy alone does not show that this baseline is useful. The confusion matrix shows that the model correctly identified 28,260 non-breaches but missed all 21,740 actual SLA breaches. Therefore, its recall is 0%, precision is 0%, and F1 score is 0%.

This demonstrates why accuracy can be misleading when evaluating a classification problem. A model should not be judged only by the percentage of correct predictions. For an SLA-breach problem, identifying actual breaches is important because missed breaches can affect customers, service commitments, and operational performance.

For this use case, I would focus more on recall, while also considering precision and the business cost of false positives.
## Day 7

### Classical ML Models and Feature Engineering

Built a classification workflow to predict SLA breaches using the synthetic incident dataset.

Feature engineering included time-based, business-hour, priority, reassignment, and ticket-related features. The model inputs were kept separate from the target and potentially leaked post-resolution information such as resolution_minutes.

Three model families were trained using scikit-learn pipelines:

- Logistic Regression
- Random Forest
- Gradient Boosting

Models were evaluated using PR-AUC.

Results:

- Logistic Regression: 0.7518
- Random Forest: 0.7249
- Gradient Boosting: 0.7523

Gradient Boosting achieved the highest PR-AUC, although its improvement over Logistic Regression was very small. Logistic Regression may therefore remain a practical alternative because of its simplicity and interpretability.

I also learned the importance of categorical encoding, numerical scaling, train/test separation, pipelines, feature engineering, and avoiding data leakage.