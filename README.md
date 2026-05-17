# Data Preprocessing and Statistical Analysis

This repository contains a comprehensive data preprocessing and exploratory data analysis (EDA) pipeline implemented in Python. The project focuses on cleaning, analyzing, and preparing the `bands.dat` dataset for future machine learning applications.

## 🚀 Project Overview
The main goal of this project is to implement standard data science workflows to transform raw data into a structured format. It covers critical steps such as missing value handling, descriptive statistics, outlier analysis, data normalization, discretization, and feature selection using Information Gain.

## 🛠️ Key Features & Steps

1. **Data Cleaning & Type Casting:**
   - Filters the raw dataset to focus on essential attributes: `ESA_amperage`, `Wax`, `Hardener`, `Roller_durometer`, and the target class `Band_type`.
   - Detects missing values represented by `?`, replaces them with `NaN`, and handles row-wise deletion.
   - Converts categorical-looking columns into proper numerical data types.

2. **Descriptive Statistics & Outlier Detection:**
   - Computes the **Five-Number Summary** (Minimum, Q1, Median, Q3, Maximum) for all numerical features.
   - Calculates key statistical metrics: Mean, Mode, Variance, and Standard Deviation.
   - Identifies outliers using the **IQR (Interquartile Range)** method by setting lower and upper bounds ($Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$).
   - Generates **Boxplots** using `matplotlib` to visually inspect data distributions and outliers.

3. **Data Normalization:**
   - **Min-Max Normalization:** Scales features linearly to a standard range of $[0, 1]$.
   - **Z-Score Normalization:** Standardizes features to have a mean of 0 and a standard deviation of 1.

4. **Data Discretization & Visualization:**
   - Converts continuous numerical features into 5 equal-width categorical bins (`pd.cut`).
   - Plots **Histograms** to display the frequency distribution across different bins.

5. **Feature Selection (Information Gain):**
   - Utilizes `scikit-learn`'s `mutual_info_classif` to compute the Information Gain of each numerical attribute relative to the target variable (`Band_type`).
   - Compares the feature significance using both **4-bin** and **5-bin** discretization strategies presented in clean tabular formats.

## 📦 Requirements

To run this script locally, make sure you have Python installed along with the following libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
