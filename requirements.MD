# Project Instructions and Requirements

## General Requirements
1.  **Set Seed:** Use student ID **100562234**.
2.  **Estimation Method:** Estimation will be done with Holdout (train/test).
    * **Main Metric:** Accuracy.
    * **Secondary Metric:** Confusion matrices shall also be included to report results.
3.  **Performance Tracking:** Execution time of the training process needs to be reported.
4.  **Preprocessing:** Should be conducted using **pipelines** when appropriate, utilizing the required preprocessing steps for each chosen method.
5.  **Presentation:** Present results with clarity. Use tables to present multiple results and compare alternatives.
6.  **Toolkit:** The objective of the assignment is to learn how to use Scikit. As much as possible we want to use Scikit-learn built-in methods and classes. We can also use numpy and pandas.
7. **Reporting:** LaTeX will be used to generate a report, and separate .py files will be used for each task, with a main .py file to run everything. When working on each task, comments will be included in the code to explain everything and a separate .tex file will be created with explanations of each task.
8. **Python Version:** Python 3.13.5 Conda will be used, located in C:\Users\96jsa\anaconda3\python.exe The general report will have a short explanation on how this environment was used, using Positron as an IDE, and a general description of the project structure.

## Tasks

### 1. Simplified EDA
Do a simplified Exploratory Data Analysis (EDA). Determine the following:
* How many features and instances are there?
* Which variables are categorical vs. numerical?
* Are there categorical variables with high cardinality?
* Which features have missing values and how many?
* Are there constant columns or ID columns?
* Is it a regression or classification problem?
* If classification, is it imbalanced?
* **Special Attention:** Pay particular attention to variable `pdays`; describe how this variable will be preprocessed.
* **Response Variable:** `deposit`.

### 2. Outer and Inner Evaluation
* Split data into **train** and **test**.
* Decide how the inner evaluation is going to be carried out.

### 3. Trees and KNN
Train, evaluate, and compare **Decision Trees** and **K-Nearest Neighbors (KNN)** with default hyperparameters, alongside a **Dummy method**.
* **For KNN:** Use a pipeline with preprocessing included (at least imputation and scaling).
* **Scaling:** Compare 2 scaling methods for KNN to determine which works better for this problem, then use that method moving forward.
* **Hyperparameter Tuning:** Perform tuning for the two basic methods (Trees and KNN) using **GridSearch** and/or **Random Search**.
* **Analysis:** Check if Hyperparameter Optimization (HPO) improves results over the default settings and report the computational cost (execution time).

### 4. Results and Final Model
* Report results from the inner evaluation and select the best one.
* Evaluate the best alternative on the **test set** to estimate future performance (outer evaluation).
* Using the best method, train the final model in an appropriate ML format and store the competition predictions in a CSV file.

### 5. Open-Choice Task
Decide on an additional task to improve results or explore an interesting angle. Justify your selection.
* *Examples:* Using external libraries like **XGBoost**, **LightGBM**, or **CatBoost**; or using advanced libraries like **FLAML**, etc.

---

## Appendix: Variable Description

| Variable | Short Description |
| :--- | :--- |
| **age** | age in years |
| **job** | type of job |
| **marital** | marital status |
| **education** | education level |
| **default** | has credit in default? |
| **balance** | average yearly balance |
| **housing** | has a housing loan? |
| **loan** | has a personal loan? |
| **contact** | contact communication type |
| **day_of_week** | last contact day |
| **month** | last contact month |
| **duration** | last contact duration, in seconds |
| **campaign** | number of contacts performed during this campaign and for this client |
| **pdays** | number of days that passed by after the client was last contacted from a previous campaign. The value is -1 if no/unknown contact was produced. |
| **previous** | number of contacts performed before this campaign and for this client |
| **poutcome** | outcome of the previous marketing campaign |
| **deposit** | has the client subscribed a term deposit? **TARGET VARIABLE** |