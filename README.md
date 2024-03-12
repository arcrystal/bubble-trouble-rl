ray.dashboard.utils.FrontendNotFoundError: [Errno 2] Dashboard build directory not found. If installing from source, please follow the additional steps required to build the dashboard: **_cd python/ray/dashboard/client && npm ci && npm run build_**


# Bubble Trouble Reinforcement Learning
## 1. Clone the repository
git clone git@github.com:arcrystal/bubble-trouble-rl.git

cd bubble-trouble-rl
## 2. Create and activate an environment
conda create -n ENV_NAME python=3.12

conda activate ENV_NAME
## 3. Install packages
pip install -r requirements.txt