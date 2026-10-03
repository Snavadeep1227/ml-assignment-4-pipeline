# ML Assignment 4 — End-to-End: Raw File to Model-Ready Data

**Repository Name**: `ml-assignment-4-pipeline`  
**Submission Task**: Task 4 (Machine Learning Assignments 1 to 4\)

---

## 1\. Dataset Information & Source Link

- **Dataset**: Hotel Booking Demand Dataset  
- **Source Link**: [TidyTuesday Hotel Bookings (2020-02-11)](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv)  
- **Original Reference**: Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49.  
- **Description**: Real-world reservation data for two hotels (City Hotel in Lisbon and Resort Hotel in Algarve, Portugal) from 2015 to 2017, containing 119,390 rows and 32 raw columns with genuine missing values, extreme anomalies, inconsistent categorical codes, and target leakage.

---

## 2\. Before and After Cleaning & Preprocessing Summary (Question 7\)

| Metric | Raw Stage | Final Stage (Cleaned Matrix) | Change / Notes |
| :---- | :---- | :---- | :---- |
| **Row Count** | 119,390 | 86,940 | \-32,450 rows (-31,994 duplicates, \-180 zero guests, \-715 zero-night stays, \-2 extreme ADR errors) |
| **Column Count** | 32 | 45 | \+13 net columns (-3 dropped leak/sparse, \+5 engineered features, one-hot encoded categorical expansion) |
| **Missing Values** | 129,425 | 0 | All NaNs resolved via structured median/mode imputation and 'Unknown' binning |
| **Duplicate Rows** | 31,994 | 0 | 100% duplicate-free to guarantee no leakage between train and test splits |

---

## 3\. Four Core Functions Architecture (Question 4\)

The pipeline is designed modularly into four distinct functional components:

1. `load_data()`: Reads raw data seamlessly from the remote URL or cached local CSV.  
2. `clean(df)`: Filters invalid records (zero guests, zero-night stays, negative ADR), removes 31,994 duplicates, resolves inconsistent labels (`meal` 'Undefined' \-\> 'SC'), imputes missing values, and drops leakage columns (`company`, `reservation_status`, `reservation_status_date`).  
3. `add_features(df)`: Constructs 5 domain features (`total_stay_nights`, `total_guests`, `room_type_changed`, `net_cancellation_history`, `has_agent`).  
4. `build_pipeline(numeric_cols, categorical_cols)`: Builds a scikit-learn `ColumnTransformer` inside a `Pipeline` utilizing `SimpleImputer`, `RobustScaler`, and `OneHotEncoder(handle_unknown='ignore', drop='first')`.

---

## 4\. Verification & Assertions (Question 6\)

All model-ready matrices are strictly verified with automated assertions:

- `assert not np.isnan(X_train).any()` & `assert not np.isnan(X_test).any()` (Zero NaNs remain).  
- `assert X_train.shape[1] == X_test.shape[1]` (Feature dimensions match perfectly).  
- `assert np.all(np.var(X_train, axis=0) > 0)` (No degenerate zero-variance / constant features).

---

## 5\. Artifacts Included

- `assignment4_endtoend.ipynb`: Fully executable Jupyter Notebook running from start to finish without errors.  
- `cleaned_hotel_bookings.csv`: Cleaned dataset exported as CSV.  
- `pipeline.joblib`: Fitted scikit-learn preprocessing pipeline ready for downstream inference.  
- `README.md`: Project summary and verification metrics.