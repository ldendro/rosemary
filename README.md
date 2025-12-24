# rosemary
ML Trading Algorithm with Backtesting. Thorough explanation on algorithm workflow and each line of code as its expected to start as a learning experience. 


Getting started:
    1. Install these extensions:
        - Python
        - Jupyter
        - GitLens
        - CodeTour
        - Atlassian for VS Code     
    2. Clone the repository:
        - "git clone https://github.com/<your-username>/rosemary.git"
        - "cd rosemary"
        If you're working on a specific Jira ticket, create a branch:
            - "git checkout -b ROS-<ticket-number>-<short-description>"
            Ex: 
                - "git checkout -b ROS-2-data-ingestion"
    3. Create a Python virtual environment (keeps this project's dependencies isolatied from you system):
        - "python3 -m venv .venv"
        Activate it:
            - "source .venv/bin/activate"
    4. Install project dependencies
        - "pip install --upgrade pip"
        - "pip install -r requirements.txt"
        - "pip install -e ."
    5. Set up the project folder structure if its your first time running Rosemary: 
        - "mkdir -p data/raw data/curated"
        - "mkdir -p scripts notebooks"

    