# Synthetic Data Generation Framework

This repository implements a Synthetic Data Generation framework inspired by [Nature Digital Medicine's article](https://www.nature.com/articles/s41746-023-00888-7). The framework includes four key modules:

### 1. **Static Categorical Autoencoder**
- Designed for handling static categorical data.
- Captures latent representations to reconstruct synthetic categorical distributions.

### 2. **Temporal Categorical Autoencoder**
- Processes temporal data with categorical features.
- Embeds time-dependent patterns for sequence generation.

### 3. **Joint Autoencoder**
- Combines static and temporal categorical autoencoders for joint modeling.

### 4. **GAN (Generative Adversarial Network)**
- Implements a GAN architecture for generating synthetic data.

The project was developed using Python 3.10 and CUDA 12.4.

---

## Installation

To set up the environment, clone this repository and install the dependencies.

```bash
git clone <repository_url>
cd <repository_folder>
pip install -r requirements.txt
```

---

## Configuration

### 1. **Modify `rootpath` in `Conf/common/path.yaml`**

Update the `rootpath` in `Conf/common/path.yaml` to match the root directory of your project. This path will be used for file storage and retrieval across various components.

Example:

```yaml
rootpath: /path/to/your/project
```

### 2. **Modify `hydra.run.dir` in `Conf/config.yaml`**

Ensure the `hydra.run.dir` in `Conf/config.yaml` is updated to reflect the correct working directory for your project.

Example:

```yaml
hydra:
  run:
    dir: /path/to/your/project
```

---

## Usage

### 1. **Prepare Your Data**
   - Ensure your dataset is preprocessed and categorized into static and temporal features.
   
### 2. **Train Models**
   - Use the provided scripts to train each module.
   Examples:
   
   **Static Categorical Autoencoder:**
   ```bash
   python -m Trainer.train_static_categorical_ae train=static_categorical_ae train.general.num_epochs=100 
   ```
   
   **Temporal Categorical Autoencoder:**
   ```bash
   python -m Trainer.train_temporal_categorical_ae train=temporal_categorical_ae train.general.num_epochs=100 
   ```
   
   **Joint Autoencoder:**
   ```bash
   python -m Trainer.train_joint_ae train=joint_ae dataloader.batch_size=128 
   ```   
   
   **GAN:**
   ```bash
   python -m Trainer.train_gan train=gan train.general.eval_freq=100 
   ```
   
### 3. **Generate Synthetic Data**
   Run the following command to generate synthetic data. Adjust sample_size as needed:
   
   ```bash
   python -m Evaluation.generate_synthetic_data sample_size=100
   ```

---

## Requirements

- Python 3.10
- CUDA 12.4
- Required Python packages are listed in `requirements.txt`.

---

## References

- [Nature Digital Medicine: Synthetic Data Generation for Healthcare](https://www.nature.com/articles/s41746-023-00888-7)

---