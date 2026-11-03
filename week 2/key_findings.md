# Key findings: data types and data problems

Details and code: fix_data.ipynb

## Step 2: Data types

All 14 columns now have a type that fits their measurement level.

| Measurement level | Columns | Type |
|---|---|---|
| Nominal | city, gender, major_discipline, company_type | category |
| Ordinal | enrolled_university, education_level, experience, company_size, last_new_job | ordered category |
| Binary | relevant_experience, job_change | bool |
| Numeric | student_id, training_hours | int32 |
| Numeric | city_development_index | float16 |

```python
df = df.astype({"city": "category", "student_id": "int32", "job_change": "bool"})

# text -> bool: compare instead of astype(bool)
df["relevant_experience"] = df["relevant_experience"] == "Has relevant experience"

# ordered category: the order is given as a list
order = ["<10", "10-49", "50-99", "100-499", "500-999", "1000-4999", "5000-9999", "10000+"]
df["company_size"] = df["company_size"].astype(pd.CategoricalDtype(order, ordered=True))
```

- Memory went from 11.0 MB to 0.42 MB (about 96 % less). Almost all of it comes from the ten text columns, because a category stores each text only once.
- No value was lost: 20,733 missing values before and after, the same in every column.
- relevant_experience needed a comparison instead of astype(bool), otherwise "No relevant experience" would also become True.
- Order of last_new_job: we put "never" at the bottom (never < 1 < ... < >4). This is our decision, it could also be read as the longest time.
- The converted data is saved as customer_train_typed.pkl, the original CSV stays unchanged.

```python
df.memory_usage(deep=True).sum()          # 11.0 MB -> 0.42 MB
df.isna().sum().equals(missing_before)    # True: same missing values as before
```

### Allowed values and plausible ranges

| Check | Result |
|---|---|
| Categories only contain known values | yes, no typos or different spellings |
| city always "city_" + number | yes, 123 cities |
| city_development_index between 0 and 1 | yes, 0.448 to 0.949 |
| training_hours | 1 to 336, no zero or negative values |
| student_id | all 19,158 unique |
| Hidden missing values ("n/a", "-", "unknown" ...) | none |

```python
set(df["gender"].dropna()) <= {"Male", "Female", "Other"}   # True: only allowed values
df["city_development_index"].between(0, 1).all()            # True: plausible range
df["student_id"].is_unique                                   # True
```

No invalid values. The data is valid, the real problem stays the missing values.

## Step 3: Logical breaks, duplicates, outliers

Only counted so far, nothing has been changed.

| Check | Rows | Share | Assessment |
|---|---|---|---|
| Last job change longer ago than total experience | 350 | 1.8 % | possible contradiction, partly rounding (<1 vs. 1 year) |
| Primary School but enrolled at a university | 44 | 0.2 % | unusual, the course itself may count as enrollment |
| Major listed but only school education | 0 | 0 % | fine |
| City with more than one development index | 0 | 0 % | fine |
| Only one of company_size / company_type missing | 1,358 | 7.1 % | mostly missing together, see missing data step |
| Graduate or higher without a major | 251 | 1.3 % | possible |
| Under 1 year but relevant experience | 164 | 0.9 % | possible |
| Over 20 years, never changed jobs | 126 | 0.7 % | possible, same employer |
| Identical rows (without student_id) | 49 | 0.3 % | can be different people, keep |
| Same profile, different job_change | 25 | 0.1 % | keep, mention |
| training_hours outliers (IQR, above 185.5) | 984 | 5.1 % | real values from a skewed distribution, keep |
| city_development_index outliers (IQR, below 0.471) | 17 | 0.1 % | real cities, keep |
| "Other" in major / gender / company_type | 381 / 191 / 121 | 2.0 / 1.0 / 0.6 % | valid answer, keep |

Logical break, e.g. last job change vs. experience:

```python
exp = df["experience"].cat.codes      # <1 = 0, 1 = 1, ..., >20 = 21
last = df["last_new_job"].cat.codes   # never = 0, 1 = 1, ..., >4 = 5
((exp >= 0) & (last > exp)).sum()     # 350
```

Duplicates:

```python
df.drop(columns="student_id").duplicated().sum()   # 49
```

Outliers with the IQR rule:

```python
q1, q3 = df["training_hours"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)            # 185.5
(df["training_hours"] > upper).sum()    # 984
```

## Missing company information is not random

Details and code: missing_company.ipynb

company_size and company_type are the columns with the most missing values. We checked if the missingness depends on relevant_experience.

| Missing share | No relevant experience (n = 5,366) | Relevant experience (n = 13,792) | Chi² | p |
|---|---|---|---|---|
| company_size | 62.6 % | 18.7 % | 3,470 | < 0.001 |
| company_type | 58.9 % | 21.6 % | 2,474 | < 0.001 |

Without relevant experience, company information is missing about three times as often. So the values are not missing completely at random (not MCAR). A plausible reason: many of these students don't have a current employer.

Steps:

```python
# 1. missingness indicator: 1 = missing, 0 = present
df["company_size_missing"] = df["company_size"].isna().astype(int)

# 2. crosstab and chi-square test (null hypothesis: independent)
table = pd.crosstab(df["relevant_experience"], df["company_size_missing"])
chi2, p, _, _ = chi2_contingency(table)     # Chi² = 3470, p < 0.001

# 3. missing share per group
df.groupby("relevant_experience")["company_size_missing"].mean() * 100   # 62.6 % vs. 18.7 %
```

Consequence: no mode imputation. Filling the gaps with the most common value ("Pvt Ltd", "50-99") would give students without a job a company they don't have. Better: a separate category like "Unknown" and keeping the missing indicator as a feature.

### Who is behind the missing values?

The other way around: if company information is missing, what does relevant_experience say?

| Missing | No relevant experience | Relevant experience | Total |
|---|---|---|---|
| company_size | 3,357 (56.5 %) | 2,581 (43.5 %) | 5,938 |
| company_type | 3,163 (51.5 %) | 2,977 (48.5 %) | 6,140 |

```python
missing_rows = df[df["company_size"].isna()]
missing_rows["relevant_experience"].value_counts()   # No: 3,357, Yes: 2,581
```

- Not all missing company values belong to students without relevant experience, both groups are almost equally large.
- Without experience the values are missing three times as often, but this group is much smaller (5,366 vs. 13,792), so in absolute numbers both end up close together.
- "No current employer" explains only part of the gaps. We can't fill them with a fixed rule like "no relevant experience means no company", "Unknown" plus the missing indicator stays the safer choice.

### How to handle the missing values

There are at least two reasons behind the gaps, and the data can't tell them apart:

- **No employer:** there is no company to describe, the field is really "empty".
- **Not filled in:** there is a company, but the student skipped the field.

| Method | Fits here? | Why |
|---|---|---|
| Drop rows | ❌ | about 30 % of the data lost, mostly students without experience, the sample gets skewed |
| Mode imputation | ❌ | students without a job get a company they don't have |
| Fixed rule ("no experience means no company") | ❌ | almost half of the gaps come from students with experience |
| Own category "Unknown" | ✅ | simple, honest, keeps the information "something is missing" |
| Keep the `_missing` indicator | ✅ | a model can learn by itself if "missing" predicts job_change |
| Predict from other columns (e.g. KNN) | ⚠️ | a lot of effort, and without an employer there is no real value to estimate |

Decision: "Unknown" category + `_missing` indicator. The two columns are mostly missing together (only 1,358 rows / 7.1 % have just one missing), which also fits the "no employer" reason.

To watch out for:

- Both columns are `category`, so "Unknown" has to be added as a category before `fillna` works (`.cat.add_categories()`).
- company_size is ordered, "Unknown" has no sensible place in that order.
- Check afterwards that `.isna().sum()` is 0.

## Takeaways for the presentation

1. Types fixed, memory down by 96 %, no values lost.
2. Validity is fine: no invalid values, no out-of-range numbers, no hidden missing values.
3. Logical breaks are rare: every single one under 2 % of the rows.
4. Outliers and duplicates are plausible, so we keep them.
5. The big problem is still completeness: 8 of 14 columns have missing values.
6. Missing company information depends on relevant experience (p < 0.001), so it is not random and we don't fill it with the mode.
7. Still, about 43–49 % of the rows with missing company information do have relevant experience, so "no employer" doesn't explain all of the gaps.
8. Plan for the company columns: fill the gaps with an own category "Unknown" and keep the `_missing` indicator as a feature.
