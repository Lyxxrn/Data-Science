# Data types

Source: customer_train.csv (19,158 rows, 14 columns)

## Target types

| Column | Now | Target | Notes |
|---|---|---|---|
| student_id | int64 | int32 | |
| city | object | category | keep the labels, the numbers mean nothing |
| city_development_index | float64 | float16 | no rounding needed, max. 3 decimals already |
| gender | object | category | |
| relevant_experience | object | bool | "Has relevant experience" = True, don't use astype(bool) directly |
| enrolled_university | object | ordered category | see below |
| education_level | object | ordered category | see below |
| major_discipline | object | category | |
| experience | object | ordered category | see below |
| company_size | object | ordered category | see below |
| company_type | object | category | |
| last_new_job | object | ordered category | see below, rename to years_since_last_job_change afterwards |
| training_hours | int64 | int32 | |
| job_change | float64 | bool | no missing values, astype(bool) is fine |

Missing values stay NaN for now. "Unknown" and "No Major" are handled in the missing data step.

## Why some columns can't just use astype

- relevant_experience: the column holds text, and astype(bool) turns every non-empty text into True. "No relevant experience" would become True as well, so we compare with "Has relevant experience" instead.
- Ordered categories: astype("category") doesn't know any order, so pandas needs the order as a list (pd.CategoricalDtype with ordered=True).
- df.dtypes shows ordered categories as plain "category". Use df[col].cat.ordered or df[col].dtype to see the difference.

## Category order

Lowest to highest:

- enrolled_university: no_enrollment < Part time course < Full time course
- education_level: Primary School < High School < Graduate < Masters < Phd
- experience: <1 < 1 < 2 < ... < 20 < >20
- company_size: <10 < 10-49 < 50-99 < 100-499 < 500-999 < 1000-4999 < 5000-9999 < 10000+
- last_new_job: never < 1 < 2 < 3 < 4 < >4

"never" at the bottom is our decision, it could also be read as the longest time. We should be able to explain this in the presentation.

## Checks after converting

A value that is not in the category list turns into NaN without an error, so compare the missing values first.

| Check | Expected |
|---|---|
| Rows / columns | 19,158 / 14 |
| Missing values, total | 20,733 |
| company_type | 6,140 missing |
| company_size | 5,938 missing |
| gender | 4,508 missing |
| major_discipline | 2,813 missing |
| education_level | 460 missing |
| last_new_job | 423 missing |
| enrolled_university | 386 missing |
| experience | 65 missing |
| relevant_experience | 13,792 True, 5,366 False |
| job_change | 4,777 True, 14,381 False |
| training_hours | min 1, max 336 |
| student_id | min 1, max 33,380, all unique |
| city_development_index | min 0.448, max 0.949 |
| city | 123 categories |
| company_size >= "1000-4999" | 3,910 rows |
| education_level >= "Masters" | 4,775 rows |
| Memory (deep=True) | well below the 11.0 MB we started with |

Saving as CSV loses all of these types, so use Parquet or Pickle if later steps load the file.

## Data problems

Everything here is only counted, nothing gets changed yet. "Clear" means it can't be true, "soft" means it's unusual but possible.

Logical breaks:

| Problem | Type | How we check it |
|---|---|---|
| Last job change longer ago than total experience | clear | compare last_new_job and experience (category codes work because both are ordered) |
| Major listed but only school education | clear | education_level at most High School, major_discipline filled and not "No Major" |
| Primary School but enrolled at a university | clear | education_level is Primary School, enrolled_university is a course |
| Graduate or higher without a major | soft | education_level at least Graduate, major_discipline empty or "No Major" |
| Under 1 year of experience but relevant experience | soft | experience is <1, relevant_experience is True |
| Over 20 years of experience, never changed jobs | soft | experience is >20, last_new_job is never |
| Only one of company_size and company_type missing | soft | isna() differs between the two columns |
| City with more than one development index | clear | group by city, count unique index values |

Duplicates:

| Problem | How we check it |
|---|---|
| Same row with a different student_id | duplicated() without student_id |
| Same profile but different job_change | duplicated() without student_id and job_change, minus the identical rows |

Identical rows can also be two people with the same profile, so a match alone doesn't prove an error.

Outliers:

| Problem | How we check it |
|---|---|
| Unusual training_hours or city_development_index | IQR rule: below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR |

An outlier is not automatically wrong, so we look at them before removing anything.

Hidden missing values:

| Problem | How we check it |
|---|---|
| Placeholders like "n/a", "-", "unknown" | compare the categories of all text columns with a list of placeholders |
| "Other" used instead of a real answer | count "Other" in gender, major_discipline and company_type |
