import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#VERI OKUMA 

lines = open("bands.dat").readlines()

#SADECE VERİ KISMINI ALMA

data_started = False
data = []

for line in lines:

    line = line.strip()

    if line.lower() == "@data":
        data_started = True
        continue

    if data_started and line != "":
        data.append(line.split(","))

#SUTUN ISIMLERI

columns = [
    "Proof_cut",
    "Viscosity",
    "Caliper",
    "Ink_temperature",
    "Humifity",
    "Roughness",
    "Blade_pressure",
    "Varnish_pct",
    "Press_speed",
    "Ink_pct",
    "Solvent_pct",
    "Esa_voltage",
    "ESA_amperage",
    "Wax",
    "Hardener",
    "Roller_durometer",
    "Density",
    "Anode_ratio",
    "Chrome_content",
    "Band_type"
]

#DATAFRAME OLUSTURMA

df = pd.DataFrame(data, columns=columns)

#SADECE GEREKLI NITELIKLERI ALMA

df = df[[
    "ESA_amperage",
    "Wax",
    "Hardener",
    "Roller_durometer",
    "Band_type"
]]

#ILK 5 SATIRI GORME

print(df.head())

#EKSIK VERILERI TEMIZLEME

df.replace("?", np.nan, inplace=True)

#SAYISAL VERIYE CEVIRME

numeric_cols = [
    "ESA_amperage",
    "Wax",
    "Hardener",
    "Roller_durometer"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col])

#SINIF BİLGİSİ

target = df["Band_type"]

# FIVE NUMBER SUMMARY

for col in numeric_cols:

    print("\n")
    print("ATTRIBUTE:", col)

    print("Minimum:", df[col].min())

    print("Q1:",
          df[col].quantile(0.25))

    print("Median:",
          df[col].median())

    print("Q3:",
          df[col].quantile(0.75))

    print("Maximum:",
          df[col].max())

# MEAN MODE VARIANCE STD

for col in numeric_cols:

    print("\n")
    print("ATTRIBUTE:", col)

    print("Mean:",
          df[col].mean())

    print("Mode:",
          df[col].mode()[0])

    print("Variance:",
          df[col].var())

    print("Standard Deviation:",
          df[col].std())

# IQR

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    print("\n")
    print(col)

    print("IQR:",
          IQR)

# OUTLIER

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower)
        |
        (df[col] > upper)
    ]

    print("\n")
    print(col)

    print("Outlier Count:",
          len(outliers))

    print(outliers[col])

# BOXPLOT

for col in numeric_cols:

    plt.figure(figsize=(5,3))

    plt.boxplot(df[col])

    plt.title(col + " Boxplot")

    plt.show()


#TABLO OLUSTURMA

summary_table = []

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower)
        |
        (df[col] > upper)
    ]
    summary_table.append({

        "Attribute": col,

        "Minimum": df[col].min(),

        "Q1": Q1,

        "Median": df[col].median(),

        "Q3": Q3,

        "Maximum": df[col].max(),

        "Mean": df[col].mean(),

        "Mode": df[col].mode()[0],

        "IQR": IQR,

        "Variance": df[col].var(),

        "Std": df[col].std(),

        "Outlier Count": len(outliers)
    })

summary_df = pd.DataFrame(summary_table)

print(summary_df)

#MIN MAX NORMALIZATION
minmax_df = pd.DataFrame()

for col in numeric_cols:

    min_val = df[col].min()

    max_val = df[col].max()

    minmax_df[col] = (
        (df[col] - min_val)
        /
        (max_val - min_val)
    )

print("\nMIN MAX NORMALIZATION")
print(minmax_df.head())

#Z SCORE NORMALIZATION
zscore_df = pd.DataFrame()

for col in numeric_cols:

    mean = df[col].mean()

    std = df[col].std()

    zscore_df[col] = (
        (df[col] - mean)
        /
        std
    )

print("\nZ SCORE NORMALIZATION")
print(zscore_df.head())


#DISCRETIZED DATA

n_bins = 5

discretized_df = pd.DataFrame()

for col in numeric_cols:

    discretized_df[col] = pd.cut(
        df[col],
        bins=n_bins,
        labels=False
    )

print("\nDISCRETIZED DATA")
print(discretized_df.head())

#HISTOGRAM

for col in numeric_cols:

    plt.figure(figsize=(6,4))

    plt.hist(
        discretized_df[col],
        bins=n_bins
    )

    plt.title(col + " Histogram")

    plt.xlabel("Bins")

    plt.ylabel("Frequency")

    plt.show()


from sklearn.feature_selection import mutual_info_classif

# NaN degerleri temizleme
clean_df = df.dropna()

# Target yeniden oluşturma
target = clean_df["Band_type"]

# 4 Equal Width
disc4_df = pd.DataFrame()

for col in numeric_cols:

    disc4_df[col] = pd.cut(
        clean_df[col],
        bins=4,
        labels=False
    )

# Information Gain hesaplama
ig4 = mutual_info_classif(
    disc4_df,
    target
)

# Tablo olusturma
ig4_table = pd.DataFrame({

    "Attribute": numeric_cols,

    "Information Gain (4 bins)": ig4
})

print("\nINFORMATION GAIN - 4 BINS")

print(ig4_table)

# 5 Equal Width
disc5_df = pd.DataFrame()

for col in numeric_cols:

    disc5_df[col] = pd.cut(
        clean_df[col],
        bins=5,
        labels=False
    )

# Information Gain hesaplama
ig5 = mutual_info_classif(
    disc5_df,
    target
)

# Tablo olusturma
ig5_table = pd.DataFrame({

    "Attribute": numeric_cols,

    "Information Gain (5 bins)": ig5
})

print("\nINFORMATION GAIN - 5 BINS")

print(ig5_table)
