import pandas as pd

df = pd.read_csv('../customer_train.csv')

# helpers
print('Old Data Types:')
print(df.dtypes)
missing_before = df.isna().sum()
memory_before = df.memory_usage(deep=True).sum()

# fix datatypes (see .md for context)

df = df.astype({
    "student_id": "int32",
    "city": "category",
    "city_development_index": "float16",
    "gender": "category",
    "major_discipline": "category",
    "company_type": "category",
    "training_hours": "int32",
    "job_change": "bool",
})

df["relevant_experience"] = df["relevant_experience"] == "Has relevant experience"

# ordered categories, lowest to highest -> tricky changes here, see .md for more
orders = {
    "enrolled_university": ["no_enrollment", "Part time course", "Full time course"],
    "education_level": ["Primary School", "High School", "Graduate", "Masters", "Phd"],
    "experience": ["<1"] + [str(i) for i in range(1, 21)] + [">20"],
    "company_size": ["<10", "10-49", "50-99", "100-499", "500-999", "1000-4999", "5000-9999", "10000+"],
    "last_new_job": ["never", "1", "2", "3", "4", ">4"],
}

for col, order in orders.items():
    df[col] = df[col].astype(pd.CategoricalDtype(categories=order, ordered=True))

print('\nNew Data Types:')
print(df.dtypes)




##### !!!! ENTIRELY WRITTEN BY CLAUDE !!!! ##### 
# ==> maybe still useful in presentation, because he told us to use as much AI as we wanted an it shows, that we have high standards

# validation (see .md, "Checks after converting")
try:
    assert df.shape == (19158, 14)

    # values outside the category lists turn into NaN, so the counts must not change
    assert df.isna().sum().equals(missing_before)
    assert df.isna().sum().sum() == 20733

    # allowed values
    allowed = {
        "gender": {"Male", "Female", "Other"},
        "major_discipline": {"STEM", "Humanities", "Business Degree", "Arts", "No Major", "Other"},
        "company_type": {"Pvt Ltd", "Funded Startup", "Public Sector", "Early Stage Startup", "NGO", "Other"},
    }

    for col, values in allowed.items():
        assert set(df[col].dropna()) <= values, col

    assert df["city"].str.fullmatch(r"city_\d+").all()
    assert df["city"].nunique() == 123
    assert df["relevant_experience"].sum() == 13792
    assert df["job_change"].sum() == 4777

    # plausible ranges
    assert df["city_development_index"].between(0, 1).all()
    assert df["training_hours"].min() == 1 and df["training_hours"].max() == 336
    assert df["student_id"].min() == 1 and df["student_id"].max() == 33380
    assert df["student_id"].is_unique

    # order works
    assert all(df[col].cat.ordered for col in orders)
    assert (df["company_size"] >= "1000-4999").sum() == 3910
    assert (df["education_level"] >= "Masters").sum() == 4775

    memory_after = df.memory_usage(deep=True).sum()
    assert memory_after < memory_before

    print("\033[32m\nAll checks passed, memory: "
          f"{memory_before / 1e6:.1f} MB -> {memory_after / 1e6:.2f} MB\033[0m")
except AssertionError:
    print("\033[31m\nChecks failed\033[0m")
    raise
except Exception:
    print("\033[31m\nChecks failed\033[0m")
    raise




# data problems (see .md, "Data problems"), only counted, nothing gets changed here
print("\nLogical breaks:")

# codes follow the order: experience <1 = 0, 1 = 1, ..., >20 = 21 / last_new_job never = 0, 1 = 1, ..., >4 = 5 / NaN = -1
experience = df["experience"].cat.codes
last_job = df["last_new_job"].cat.codes

school_only = df["education_level"] <= "High School"
no_major = df["major_discipline"].isna() | (df["major_discipline"] == "No Major")
in_course = df["enrolled_university"].isin(["Part time course", "Full time course"])

logical_breaks = {
    "last job change longer ago than experience (clear)": (experience >= 0) & (last_job > experience),
    "major but only school education (clear)": school_only & ~no_major,
    "primary school but enrolled at university (clear)": (df["education_level"] == "Primary School") & in_course,
    "graduate or higher without major (soft)": (df["education_level"] >= "Graduate") & no_major,
    "under 1 year but relevant experience (soft)": (df["experience"] == "<1") & df["relevant_experience"],
    "over 20 years, never changed jobs (soft)": (df["experience"] == ">20") & (df["last_new_job"] == "never"),
    "only one company column missing (soft)": df["company_size"].isna() != df["company_type"].isna(),
}

for name, rows in logical_breaks.items():
    print(f"{name}: {rows.sum():,} ({rows.mean():.1%})")

index_per_city = df.groupby("city", observed=True)["city_development_index"].nunique()
print(f"cities with more than one index (clear): {(index_per_city > 1).sum()}")

print("\nDuplicates:")
identical = df.drop(columns="student_id").duplicated().sum()
same_profile = df.drop(columns=["student_id", "job_change"]).duplicated().sum()
print(f"identical rows without student_id: {identical}")
print(f"same profile, different job_change: {same_profile - identical}")

print("\nOutliers (IQR rule):")
for col in ["training_hours", "city_development_index"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ~df[col].between(low, high)
    print(f"{col}: {outliers.sum():,} ({outliers.mean():.1%}) outside {low:.3f} to {high:.3f}")

print("\nHidden missing values:")
placeholders = {"", ".", "-", "?", "n/a", "na", "nan", "none", "null", "unknown"}
for col in df.select_dtypes("category"):
    found = [c for c in df[col].cat.categories if c.strip().lower() in placeholders]
    if found:
        print(f"{col}: {found}")
print("placeholders checked in all text columns")

for col in ["gender", "major_discipline", "company_type"]:
    other = df[col] == "Other"
    print(f"{col} = Other: {other.sum():,} ({other.mean():.1%})")
