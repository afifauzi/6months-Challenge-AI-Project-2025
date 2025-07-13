# Project: End-to-end classification pipeline
1. Goal: Build reproducible ML pipelines
2. Key Tech: Docker, MLflow, Prefect, DVC
4. Dataset: CIFAR-100
5. Steps:
- Containerize training with Docker
- Version data/models with DVC
- Track experiments with MLflow
- Orchestrate workflow with Prefect
6. Outcome: Automated pipeline from data ingestion to model registry


### ASSISTANT
Here's a **structured 4-week roadmap** with daily tasks (2-3 hours/day) to build your end-to-end reproducible ML pipeline using Docker, MLflow, Prefect, and DVC:

---

### **Week 1: Project Setup & Docker Containerization**
**Goal**: Create a Dockerized training environment and baseline ML pipeline.
- **Day 1**:  
  - Set up project repository structure:  
    ```bash
    project-root/
    ├── src/                # Training scripts
    ├── Dockerfile          # Containerization
    ├── requirements.txt    # Dependencies
    └── prefect_flows/      # Prefect workflows (future)
    ```
  - Install Docker and test with `docker run hello-world`.

- **Day 2**:  
  - Write a basic Python script to load CIFAR-100 data and train a simple model (e.g., PyTorch ResNet baseline).
  - Add MLflow tracking to log accuracy/metrics (no server yet).

- **Day 3**:  
  - Create a `Dockerfile` to containerize the training script.  
    Example:  
    ```dockerfile
    FROM nvidia/cuda:12.1.0-base
    RUN apt-get update && apt-get install -y python3-pip
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY src/ /app
    WORKDIR /app
    CMD ["python", "train.py"]
    ```

- **Day 4**:  
  - Test Docker image locally:  
    ```bash
    docker build -t cifar100-train .
    docker run --gpus all cifar100-train
    ```
  - Verify MLflow logs in `mlruns/` directory.

- **Day 5**:  
  - Push Docker image to Docker Hub (free account) for later Prefect integration.  
    ```bash
    docker tag cifar100-train <your-dockerhub>/cifar100-train
    docker push <your-dockerhub>/cifar100-train
    ```

---

### **Week 2: Data/Model Versioning with DVC**
**Goal**: Version datasets and models, integrate with MLflow.
- **Day 1**:  
  - Install DVC and initialize:  
    ```bash
    pip install dvc
    dvc init
    git add . && git commit -m "Initialize DVC"
    ```
  - Use DVC to track raw CIFAR-100 data:  
    ```bash
    dvc get https://github.com/iterative/dataset-registry registry/cifar100
    dvc add cifar100/
    git add cifar100.dvc
    git commit -m "Add raw CIFAR-100 dataset"
    ```

- **Day 2**:  
  - Modify training script to use DVC-managed data.  
  - Add DVC pipeline stages for preprocessing:  
    ```bash
    dvc run -n preprocess -d src/preprocess.py -d data/raw -o data/processed python src/preprocess.py
    ```

- **Day 3**:  
  - Configure DVC remote storage (local or cloud):  
    ```bash
    dvc remote add -d local_remote /path/to/remote/storage
    dvc push
    ```
  - Version models via DVC:  
    ```bash
    dvc run -n train -d data/processed -d src/train.py -o models/model.pth python src/train.py
    ```

- **Day 4**:  
  - Link DVC and MLflow: Log DVC commit hashes in MLflow runs for traceability.  
    ```python
    import mlflow
    import subprocess

    dvc_hash = subprocess.check_output(["dvc", "status"]).decode("utf-8")
    mlflow.log_param("dvc_version", dvc_hash)
    ```

- **Day 5**:  
  - Test DVC reproducibility:  
    ```bash
    dvc repro
    git commit -am "Update dataset"
    dvc push
    ```

---

### **Week 3: Experiment Tracking with MLflow**
**Goal**: Centralize experiment tracking and model registry.
- **Day 1**:  
  - Set up MLflow tracking server:  
    ```bash
    mlflow server --host 0.0.0.0 --port 5000
    export MLFLOW_TRACKING_URI=http://localhost:5000
    ```
  - Configure DVC to use MLflow server.

- **Day 2**:  
  - Enhance training script to log hyperparameters, metrics, and artifacts to MLflow:  
    ```python
    with mlflow.start_run():
        mlflow.log_params({"lr": 0.001, "batch_size": 128})
        mlflow.log_metric("accuracy", val_acc)
        mlflow.pytorch.log_model(model, "model")
    ```

- **Day 3**:  
  - Run multiple experiments (e.g., vary learning rate) and compare in MLflow UI.
  - Tag best runs for registry:  
    ```bash
    mlflow experiments set-experiment-tag 0 "best_run" "run_id_here"
    ```

- **Day 4**:  
  - Register a model in MLflow Model Registry:  
    ```bash
    mlflow models create -n "cifar100-resnet" -d "CIFAR-100 classification model"
    mlflow models update -n "cifar100-resnet" -v 1 -s PRODUCTION
    ```

- **Day 5**:  
  - Automate MLflow/DVC alignment: Write a script to fetch the latest DVC data version for MLflow runs.

---

### **Week 4: Prefect Orchestration & Final Pipeline**
**Goal**: Automate the full pipeline with Prefect.
- **Day 1**:  
  - Install Prefect and start Orion server:  
    ```bash
    pip install prefect
    prefect orion start
    export PREFECT_API_URL=http://127.0.0.1:4200/api
    ```

- **Day 2**:  
  - Define Prefect tasks for each pipeline step (data download, preprocessing, training, logging).  
    Example:  
    ```python
    from prefect import task, flow

    @task
    def run_dvc_repro(stage: str):
        subprocess.run(["dvc", "repro", stage])

    @flow
    def ml_pipeline():
        run_dvc_repro("preprocess")
        run_dvc_repro("train")
    ```

- **Day 3**:  
  - Integrate Docker container with Prefect:  
    ```python
    from prefect_docker import DockerImage

    @task
    def docker_train():
        subprocess.run(["docker", "run", "--gpus", "all", "<your-dockerhub>/cifar100-train"])
    ```

- **Day 4**:  
  - Add MLflow/DVC hooks to Prefect flow for tracking and versioning.  
  - Schedule flow with Prefect agent:  
    ```bash
    prefect agent start -q default
    ml_pipeline.deploy(name="cifar100-pipeline", work_pool_name="default")
    ```

- **Day 5**:  
  - Test end-to-end pipeline: Trigger a run and verify data ingestion → training → MLflow logging → DVC versioning.
  - Document workflow and write a README for reproducibility.

---

### **Final Deliverables**
1. Dockerized training pipeline (`Dockerfile`, trained models).
2. DVC-tracked datasets and models (`data.dvc`, `models.dvc`).
3. MLflow experiments and model registry (hosted at `http://localhost:5000`).
4. Prefect flow dashboard (at `http://localhost:4200`).
5. Git repo with version-controlled code, DVC hashes, and Prefect configurations.

### **Tips for Success**
- Use `tmux` or Docker Desktop to keep servers running.
- Allocate 1-2 days for debugging tool integrations (e.g., Prefect + Docker GPU access).
- Prioritize modular code: Separate training logic, DVC stages, and Prefect tasks.
