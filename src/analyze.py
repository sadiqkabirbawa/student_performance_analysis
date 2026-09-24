import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from pathlib import Path

DATA = Path("data/student-mat.csv")
OUT = Path("visualizations")
OUT.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError("Download data/student-mat.csv from the official UCI dataset page.")

df = pd.read_csv(DATA, sep=";")
print("Shape:", df.shape)
print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head())
print("\nDescriptive statistics:\n", df[["age","studytime","failures","absences","G1","G2","G3"]].describe().round(2))
print("\nMean G3:", round(df["G3"].mean(), 2))
print("Median G3:", round(df["G3"].median(), 2))
print("\nMean G3 by study time:\n", df.groupby("studytime")["G3"].mean().round(2))
print("\nCorrelations with G3:\n", df[["age","studytime","failures","absences","G1","G2","G3"]].corr()["G3"].sort_values().round(3))

lower = df.loc[df["studytime"].isin([1,2]), "G3"]
higher = df.loc[df["studytime"].isin([3,4]), "G3"]
t_stat, p_value = ttest_ind(lower, higher, equal_var=False)
print("\nWelch t-test")
print("Lower-study-time mean:", round(lower.mean(), 2))
print("Higher-study-time mean:", round(higher.mean(), 2))
print("t-statistic:", round(t_stat, 3))
print("p-value:", round(p_value, 4))

plt.figure(figsize=(8,5))
sns.histplot(df["G3"], bins=15, kde=True)
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig(OUT/"final_grade_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(8,5))
df.groupby("studytime")["G3"].mean().plot(kind="bar")
plt.title("Average Final Grade by Study Time")
plt.xlabel("Study Time Category")
plt.ylabel("Average Final Grade")
plt.tight_layout()
plt.savefig(OUT/"average_grade_by_study_time.png", dpi=150)
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="absences", y="G3", alpha=.55)
plt.title("Absences vs Final Grade")
plt.tight_layout()
plt.savefig(OUT/"absences_vs_final_grade.png", dpi=150)
plt.close()

print("\nDone. Charts saved to visualizations/.")
