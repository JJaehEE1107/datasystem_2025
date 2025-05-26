<p align="left">
    <img src="assets/logo.webp" align="left" width="30%">
	
</p>
<p align="left"><h1 align="left">DATASYSTEM_2025</h1></p>
<p align="left">
	<em>Unleashing Data Power, Predicting Tomorrow's Gold Today!</em>
</p>
<p align="left">
	<img src="https://img.shields.io/github/license/JJaehEE1107/datasystem_2025?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/JJaehEE1107/datasystem_2025?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/JJaehEE1107/datasystem_2025?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/JJaehEE1107/datasystem_2025?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="left"><!-- default option, no dependency badges. -->
</p>
<p align="left">
	<!-- default option, no dependency badges. -->
</p>
<br>

<details><summary>Table of Contents</summary>

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

</details>
<hr>

##  Overview

The datasystem2025 project is a powerful, open-source tool designed to streamline financial data analysis and forecasting. It leverages cutting-edge technologies to extract, process, and visualize gold, Bitcoin, and US index prices, alongside key economic indicators. Users can explore data, train machine learning models, and make future price predictions through an intuitive dashboard. Ideal for financial analysts, data scientists, and enthusiasts, this project offers a comprehensive solution for data-driven decision making in the financial sector.

---

##  Features

|      | Feature         | Summary       |
| :--- | :---:           | :---          |
| ⚙️  | **Architecture**  | <ul><li>The project uses a microservices architecture, orchestrated with `docker-compose.yaml`.</li><li>It leverages Apache Airflow for ETL pipeline orchestration.</li><li>The project uses PostgreSQL for database management, Redis for task queueing, and MinIO for object storage.</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>The codebase is primarily written in Python, with some YAML configuration files.</li><li>It follows good practices of code organization, separating different functionalities into different files and folders.</li><li>The project uses Docker for environment setup, ensuring reproducibility across different systems.</li></ul> |
| 📄 | **Documentation** | <ul><li>The project provides detailed installation and usage instructions for both `pip` and `docker`.</li><li>It also includes test commands for running unit tests.</li><li>Each major file in the codebase has a brief description explaining its purpose and functionality.</li></ul> |
| 🔌 | **Integrations**  | <ul><li>The project integrates with PostgreSQL for database management.</li><li>It uses Redis for task queueing.</li><li>MinIO is used for object storage.</li></ul> |
| 🧩 | **Modularity**    | <ul><li>The project is highly modular, with separate services for the webserver, scheduler, worker, and triggerer.</li><li>Each service is defined and configured in the `docker-compose.yaml` file.</li><li>The ETL pipeline is modular, with separate tasks for fetching, processing, and uploading data.</li></ul> |
| 🧪 | **Testing**       | <ul><li>The project provides test commands for running unit tests using `pytest`.</li><li>However, there is no explicit mention of a testing framework or test cases in the provided codebase details.</li></ul> |
| ⚡️  | **Performance**   | <ul><li>The project uses Redis for task queueing, which can significantly improve performance for concurrent tasks.</li><li>It uses a lightweight Python environment for running the application, which can help reduce resource usage.</li></ul> |
| 🛡️ | **Security**      | <ul><li>The project uses Docker, which provides isolation between the application and the host system.</li><li>However, there is no explicit mention of any security measures or practices in the provided codebase details.</li></ul> |
| 📦 | **Dependencies**  | <ul><li>The project uses a variety of libraries for data analysis and visualization, such as `matplotlib`, `seaborn`, and `plotly`.</li><li>It uses machine learning libraries such as `scikit-learn`, `xgboost`, and `joblib`.</li><li>All dependencies are listed in the `app/requirements.txt` file.</li></ul> |

---

##  Project Structure

```sh
└── datasystem_2025/
    ├── Dockerfile
    ├── README.md
    ├── app
    │   ├── app.py
    │   └── requirements.txt
    ├── dags
    │   └── gold.py
    ├── data
    │   ├── CPI_data.csv
    │   ├── FedFunds_data.csv
    │   ├── Unemployment_data.csv
    │   ├── bitcoin_data.csv
    │   ├── gold_data.csv
    │   └── us_index_data.csv
    ├── docker-compose.yaml
    └── logs
        └── scheduler
            └── latest
```


###  Project Index
<details open>
	<summary><b><code>DATASYSTEM_2025/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/JJaehEE1107/datasystem_2025/blob/master/docker-compose.yaml'>docker-compose.yaml</a></b></td>
				<td>- The docker-compose.yaml file orchestrates the deployment of an Apache Airflow environment, utilizing services such as PostgreSQL for database management, Redis for task queueing, and MinIO for object storage<br>- It also sets up the Airflow webserver, scheduler, worker, and triggerer, ensuring they are properly initialized and configured.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/JJaehEE1107/datasystem_2025/blob/master/Dockerfile'>Dockerfile</a></b></td>
				<td>- The Dockerfile sets up a lightweight Python environment, installs necessary system packages, and sets up the application in a dedicated workspace<br>- It also exposes port 8501 for the Streamlit application and specifies the command to run the application<br>- This contributes to the project's portability and reproducibility across different systems.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- dags Submodule -->
		<summary><b>dags</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/JJaehEE1107/datasystem_2025/blob/master/dags/gold.py'>gold.py</a></b></td>
				<td>- The 'gold.py' file is a part of an ETL (Extract, Transform, Load) pipeline that fetches and processes data related to gold, Bitcoin, and US index prices, as well as economic indicators from FRED<br>- It then uploads the data to a MinIO data lake and inserts it into a PostgreSQL database<br>- This pipeline is orchestrated using Apache Airflow.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- app Submodule -->
		<summary><b>app</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/JJaehEE1107/datasystem_2025/blob/master/app/app.py'>app.py</a></b></td>
				<td>- The code in app/app.py is a comprehensive script for a Gold Price Prediction Dashboard<br>- It fetches historical financial data, preprocesses it, trains machine learning models, and makes future price predictions<br>- The script also includes a user interface for data exploration, model training, and viewing predictions<br>- It allows users to save trained models and load them for future predictions.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/JJaehEE1107/datasystem_2025/blob/master/app/requirements.txt'>requirements.txt</a></b></td>
				<td>- App/requirements.txt outlines the necessary libraries for the project, including data visualization tools like matplotlib, seaborn, and plotly, machine learning libraries such as scikit-learn, xgboost, and joblib, and other essential packages like streamlit and psycopg2-binary<br>- These dependencies ensure the smooth functioning of the project's data analysis and visualization tasks.</td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with datasystem_2025, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip
- **Container Runtime:** Docker


###  Installation

Install datasystem_2025 using one of the following methods:

**Build from source:**

1. Clone the datasystem_2025 repository:
```sh
❯ git clone https://github.com/JJaehEE1107/datasystem_2025
```

2. Navigate to the project directory:
```sh
❯ cd datasystem_2025
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r app/requirements.txt
```


**Using `docker`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Docker-2CA5E0.svg?style={badge_style}&logo=docker&logoColor=white" />](https://www.docker.com/)

```sh
❯ docker build -t JJaehEE1107/datasystem_2025 .
```

**Using `docker compose`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Docker-2CA5E0.svg?style={badge_style}&logo=docker&logoColor=white" />](https://www.docker.com/)

```sh
❯ % docker compose up -d .
```
**Required 'PostgreSQL Table Setup'`** &nbsp; <img align="center" src="https://img.shields.io/badge/PostgreSQL-4169E1.svg?style={badge_style}&logo=postgresql&logoColor=white" />
```sh
❯ psql -h localhost -U your_username -d datasystem_db
```
**Required 'Configuration' ** &nbsp; [<img align="center" src="https://img.shields.io/badge/.env-Informational.svg?style={badge_style}&logo=dotenv&logoColor=white" />]

The application requires a .env file in the root directory to load essential credentials and connection details.
This includes FRED API key, MinIO access info, and PostgreSQL database config.

```sh
env
Copy
Edit
# --- FRED API ---
FRED_API_KEY=your_fred_api_key_here
_PIP_ADDITIONAL_REQUIREMENTS=pandas yfinance==0.2.58 fredapi pandas_datareader psycopg2-binary virtualenv boto3 sqlalchemy pandas-datareader minio
# --- PostgreSQL ---
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=datasystem_db
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password

# --- MinIO (Optional, if used for raw data storage) ---
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_ENDPOINT=http://localhost:9000
✅ Make sure this file is created before running Docker or executing the app locally.
You can rename .env.example to .env and fill in your credentials.
```



###  Usage
Run datasystem_2025 using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python {entrypoint}
```


**Using `docker`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Docker-2CA5E0.svg?style={badge_style}&logo=docker&logoColor=white" />](https://www.docker.com/)

```sh
❯ docker run -it {image_name}
```



<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/JJaehEE1107/datasystem_2025/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=JJaehEE1107/datasystem_2025">
   </a>
</p>
</details>


