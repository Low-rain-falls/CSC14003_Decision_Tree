# Project 2: Decision Tree Classifier for Real-World Datasets

## Overview

This repository contains the implementation of **Project 2** for the course *Introduction to Artificial Intelligence (CS14003)* at the University of Science, Faculty of Information Technology. The project focuses on building and evaluating **Decision Tree classifiers** using the `scikit-learn` library on three datasets:

- **UCI Heart Disease Dataset**: A binary classification dataset to predict the presence of heart disease (303 samples).
- **Palmer Penguins Dataset**: A multi-class classification dataset to identify penguin species (344 samples).
- **Breast Cancer Wisconsin Dataset**: A binary classification dataset to predict benign or malignant tumors (569 samples).

The project involves data preparation, model training, performance evaluation, visualization of decision trees, and a comparative analysis of model performance across the datasets.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Setup Instructions](#setup-instructions)
   <!-- - [Using Conda](#using-conda) -->
   - [Using pip](#using-pip)
4. [How to Run](#how-to-run)
5. [Datasets](#datasets)
6. [Results](#results)
7. [Team Members](#team-members)
8. [License](#license)

## Project Structure

```
project2_decision_tree/
│
├── notebooks/
│   ├── heart_disease.ipynb       # Analysis for UCI Heart Disease dataset
│   ├── palmer_penguins.ipynb    # Analysis for Palmer Penguins dataset
│   ├── breast_cancer.ipynb      # Analysis for Breast Cancer Wisconsin dataset
│
├── data/
│   ├── heart_disease.csv        # UCI Heart Disease dataset (optional)
│   ├── penguins.csv             # Palmer Penguins dataset (optional)
│   ├── breast_cancer.csv        # Breast Cancer Wisconsin dataset (optional)
│
├── report/
│   ├── project_report.pdf       # Final report with analysis and insights
│
├── README.md                    # This file
└── requirements.txt             # Dependencies for pip
```

## Requirements

The project requires **Python 3.8+** and the following libraries:

- `scikit-learn`
- `pandas`
- `matplotlib`
- `seaborn`
- `graphviz`
- `python-graphviz`
- `jupyter`
- `nbconvert`

## Setup Instructions

<!-- ### Using Conda -->
<!---->
<!-- 1. **Install Anaconda or Miniconda**: -->
<!--    - Download from [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). -->
<!--    - Verify installation: -->
<!---->
<!--      ```bash -->
<!--      conda --version -->
<!--      ``` -->
<!---->
<!-- 2. **Create a Conda environment**: -->
<!---->
<!--    ```bash -->
<!--    conda create -n decision_tree python=3.9 -->
<!--    ``` -->
<!---->
<!-- 3. **Activate the environment**: -->
<!---->
<!--    ```bash -->
<!--    conda activate decision_tree -->
<!--    ``` -->
<!---->
<!-- 4. **Install dependencies**: -->
<!---->
<!--    ```bash -->
<!--    conda install scikit-learn pandas matplotlib seaborn jupyter graphviz python-graphviz nbconvert -->
<!--    ``` -->
<!---->
<!-- 5. **Verify installation**: -->
<!---->
<!--    ```bash -->
<!--    conda list -->
<!--    ``` -->
<!---->
<!--    Check for `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `graphviz`, `jupyter`, and `nbconvert`. -->
<!---->

### Using pip

1. **Install Python**:
   - Download Python 3.8+ from [python.org](https://www.python.org/downloads/).
   - Verify:

     ```bash
     python --version
     ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv decision_tree_env
   source decision_tree_env/bin/activate  # On Windows: decision_tree_env\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Or manually:

   ```bash
   pip install scikit-learn pandas matplotlib seaborn jupyter graphviz nbconvert
   ```

4. **Install Graphviz**:
   - Download from [graphviz.org](https://graphviz.org/download/).
   - Add Graphviz to PATH (e.g., `C:\Program Files\Graphviz\bin` on Windows).
   - Verify:

     ```bash
     dot -V
     ```

5. **Verify libraries**:

   ```python
   import sklearn, pandas, matplotlib, seaborn, graphviz
   print("Libraries installed successfully!")
   ```

## How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/[your-repo]/project2_decision_tree.git
   cd project2_decision_tree
   ```

2. **Activate the environment**:
   - Conda:

     ```bash
     conda activate decision_tree
     ```

   - pip:

     ```bash
     source decision_tree_env/bin/activate  # On Windows: decision_tree_env\Scripts\activate
     ```

3. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

4. **Run notebooks**:
   - Open `notebooks/heart_disease.ipynb`, `palmer_penguins.ipynb`, or `breast_cancer.ipynb` in the Jupyter interface.
   - Execute all cells to perform the analysis.

5. **Export to PDF** (optional):

   ```bash
   jupyter nbconvert --to pdf notebooks/[notebook_name].ipynb
   ```

## Datasets

1. **UCI Heart Disease**:
   - Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
   - Description: 303 samples, 13 features, binary classification (0: no heart disease, 1: heart disease).
   - Access: Loaded via URL or `data/heart_disease.csv`.

2. **Palmer Penguins**:
   - Source: [seaborn](https://github.com/mwaskom/seaborn-data) or [Palmer Penguins](https://github.com/allisonhorst/palmerpenguins)
   - Description: 344 samples, 6 features, 3 classes (Adelie, Chinstrap, Gentoo).
   - Access: Loaded via `seaborn.load_dataset('penguins')` or `data/penguins.csv`.

3. **Breast Cancer Wisconsin**:
   - Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
   - Description: 569 samples, 30 features, binary classification (benign or malignant).
   - Access: Loaded via `sklearn.datasets.load_breast_cancer()` or `data/breast_cancer.csv`.

## Results

- **Notebooks**:
  - Data preparation with stratified splits (40/60, 60/40, 80/20, 90/10).
  - Decision tree training and visualization using Graphviz.
  - Evaluation with classification reports and confusion matrices.
  - Analysis of max_depth impact on accuracy (80/20 split).
- **Report** (`report/project_report.pdf`):
  - Visualizations and statistical results.
  - Comparative analysis of dataset characteristics and model performance.
  - Insights on decision tree behavior across datasets.

<!-- ## Team Members -->
<!---->
<!-- | Student ID | Full Name       | Task Assigned                        | Completion Rate | -->
<!-- |------------|-----------------|--------------------------------------|-----------------| -->
<!-- | [ID1]      | [Name1]         | Data preparation, Heart Disease      | [e.g., 90%]     | -->
<!-- | [ID2]      | [Name2]         | Model training, Penguins             | [e.g., 85%]     | -->
<!-- | [ID3]      | [Name3]         | Breast Cancer analysis, visualization | [e.g., 95%]     | -->
<!-- | [ID4]      | [Name4]         | Report writing, comparative analysis | [e.g., 90%]     | -->
<!---->

## License

This project is for educational purposes only. Datasets are used under their respective public licenses.

---

<!-- ### Changes Made -->
<!---->
<!-- - **Additional Dataset**: I assumed the **Breast Cancer Wisconsin Dataset** as the additional dataset (569 samples, binary classification) since you hadn’t specified one. It fits the project requirements (≥300 samples, supervised learning, binary classes). If you prefer another dataset (e.g., Wine, Iris, or Titanic), let me know, and I’ll update it. -->
<!-- - **Clarity**: Simplified some instructions for brevity while keeping all necessary details. -->
<!-- - **Consistency**: Ensured Conda and pip instructions are parallel and easy to follow. -->
<!-- - **Team Members**: Left placeholders for you to fill in your group’s details. -->
<!-- - **Repository URL**: Kept `[your-repo]` as a placeholder; replace it with your actual GitHub link if applicable. -->
<!---->
<!-- ### Questions for You -->
<!---->
<!-- 1. **Dataset Confirmation**: Is the **Breast Cancer Wisconsin Dataset** okay for the additional dataset, or do you want another (e.g., Wine, Titanic)? I can update the README accordingly. -->
<!-- 2. **Team Details**: Want me to help format the team members’ section if you have specific names/IDs? -->
<!-- 3. **Further Steps**: Should I help with: -->
<!--    - Writing code for one of the notebooks (e.g., Heart Disease data preparation)? -->
<!--    - Creating a Vietnamese version of the README? -->
<!--    - Setting up the project folder structure on your machine? -->
<!-- 4. **Anything Else**: Did I misunderstand your request? If you meant translating something else or modifying a specific part, please clarify. -->
