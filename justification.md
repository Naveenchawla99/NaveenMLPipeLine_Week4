# Engineering Justification & System Design Document: Production Pipeline for Melbourne Housing Valuation

**Dataset:** Melbourne Housing Market (`Melbourne_housing_FULL.csv`)  
**Deliverable:** End-to-End Scikit-Learn Production Pipeline (`pipeline.py`)  

---

## 1. The Dataset and Four Messiness Properties

The Melbourne Housing Market dataset represents real-world administrative and scraped real estate records, which inherently suffer from severe data degradation. Empirical analysis revealed four critical messiness properties that shaped the pipeline's defensive architecture:

1. **Structural Missingness (Not Missing at Random):** Entire structural blocks—such as `BuildingArea`, `YearBuilt`, and `Car`—exhibit high rates of missing values. These are frequently missing because older properties or certain real estate listings omit administrative metadata rather than because the data was missing purely at random.
2. **Invalid Physical Values (Outliers & Corruptions):** The data contains impossible physical entries resulting from administrative typos or scanner errors. For example, `YearBuilt` contains entries like `1196` (a medieval typo for 1996), and `BuildingArea` contains entries recorded as `0` square meters.
3. **High-Cardinality Categoricals:** Features like `Suburb` (containing hundreds of distinct districts) and `SellerG` (containing hundreds of fragmented real estate agencies) present a high-cardinality challenge that would cause severe dimensionality explosion under naive one-hot encoding.
4. **Heavy Right-Skewness & Multi-Scale Dispersion:** Monetary values (`Price`) and spatial/physical dimensions (`Landsize`, `Distance`) span multiple orders of magnitude with heavy right tails, violating the variance stability and normality assumptions of linear estimators.

---

## 2. The Audit Table

| Column Group | Raw Columns | Action / Imputation | Scaling / Transformation | Encoding |
| :--- | :--- | :--- | :--- | :--- |
| **Target** | `Price` | Drop rows with missing target; natural log transformation ($y = \ln(\text{Price})$) | None | None |
| **Dropped** | `Address`, `Postcode`, `Bedroom2` | Drop (redundant identifiers or high-noise duplicates) | None | None |
| **Standard Numerical** | `Rooms`, `YearBuilt`, `Lattitude`, `Longtitude`, `Year`, `Month` | Group median or global median fallback | `StandardScaler` | None |
| **Log-Standard Numerical** | `Distance`, `Propertycount`, `Car`, `Bathroom` | Group median imputation + binary missing indicator flags | `log1p` $\rightarrow$ `StandardScaler` | None |
| **Clipped Log-Standard** | `BuildingArea` | Quantile clipping (1%–99%) $\rightarrow$ imputation | `log1p` $\rightarrow$ `StandardScaler` | None |
| **Yeo-Johnson Numerical** | `Landsize` | Quantile clipping (1%–99%) $\rightarrow$ zero-indicator flag | `PowerTransformer(yeo-johnson)` $\rightarrow$ `StandardScaler` | None |
| **Low-Card Categorical** | `Type`, `Method`, `Regionname` | Map missing/unseen strings to `"Unknown"` | None | `OneHotEncoder(handle_unknown="ignore")` |
| **High-Card Categorical** | `CouncilArea`, `SellerG` | Map missing/unseen strings to `"Unknown"` | None | `OneHotEncoder(min_frequency=..., handle_unknown="infrequent_if_exist")` |
| **Spatial Categorical** | `Suburb` | Map missing/unseen strings to `"Unknown"` | None | Out-of-fold `TargetEncoder(smooth="auto")` |

---

## 3. Design Decisions (By Column Group)

### Target Variable (`Price`)
> **I saw** that housing prices span orders of magnitude with an extreme right skew, **so I did** drop missing target rows and apply a natural logarithm transformation ($y = \ln(\text{Price})$), **because** it stabilizes variance, reduces heteroscedasticity, and aligns prediction errors with linear regression optimization objectives.

### High-Cardinality Categorical (`Suburb`)
> **I saw** hundreds of distinct suburbs with highly distinct pricing tiers, **so I did** route them through an out-of-fold `TargetEncoder` with automated smoothing, **because** it maps high-cardinality strings directly into continuous target-mean features without inflating feature space dimensionality.

### Skewed Physical Metrics (`Landsize`, `BuildingArea`)
> **I saw** extreme positive outliers alongside structural zero-inflation (e.g., apartments with zero land size), **so I did** implement a custom `QuantileClipper` (clipping training distributions between the 1st and 99th percentiles) followed by `Yeo-Johnson` or `log1p` transforms, **because** unclipped extremes would otherwise dominate gradient updates and bias linear coefficients.

### Missing Data and Imputation (`BuildingArea`, `YearBuilt`, `Car`)
> **I saw** that missing physical values strongly correlate with property types, **so I did** implement a hierarchical `GroupMedianImputer` (grouped primarily by property `Type`), falling back to global training medians and ultimately `0.0`, **because** preserving sub-market context prevents structural distortion while guaranteeing zero NaN propagation into downstream estimators.

---

## 4. The Leak Experiments and What They Showed

1. **Experiment 1 (The Deliberate Leak Test):** Fitting a preprocessor (or target encoder) globally on the entire dataset prior to cross-validation artificially inflated cross-validated $R^2$ scores compared to the correct out-of-fold pipeline. Even when the numerical difference in performance metrics appeared small on large datasets, the leak breaks fundamental out-of-sample generalization guarantees by allowing validation target labels to contaminate training transformations.
2. **Experiment 2 (The Unseen Category Test):** Injecting a production test row with brand-new, unseen categorical values (e.g., `Suburb="AtlantisCity"`, `Type="MegaMansions"`) caused strict models lacking unknown handling to crash with a `ValueError`. Conversely, the correct pipeline handled them gracefully via `handle_unknown="ignore"`, infrequent group collapsing, or the `TargetEncoder`'s built-in global training mean fallback.

---

## 5. What I Would Do Differently with More Time / More Data

1. **Tree-Based Ensemble Integration:** Replace or blend the linear `Ridge` regressor with gradient-boosted decision tree models (such as `LightGBM` or `XGBoost`), which natively handle non-linear interactions, missing data patterns, and heavy skews without requiring manual clipping or scaling.
2. **Advanced Geospatial Engineering:** Instead of treating latitude and longitude as independent linear components, engineer spatial metrics such as Haversine distance to the Melbourne Central Business District (CBD) or DBSCAN spatial density clusters.
3. **Temporal Out-of-Time Validation:** Replace standard randomized `KFold` cross-validation with `TimeSeriesSplit` based on the property sale date, ensuring the evaluation protocol reflects real-world deployment where future housing prices cannot be predicted using past random folds.

---

## 6. Pipeline Failure Modes & What Breaks It

Every robust production pipeline has explicit operational boundaries. This pipeline will fail or break under the following conditions:

1. **Upstream Schema Drift and Rename:** If upstream data engineering pipelines alter raw column names (e.g., renaming `Rooms` to `Room_Count`) or supply unexpected data types that cannot be safely coerced to strings or floats, the custom `Cleaner` will reindex incorrectly or raise casting exceptions.
2. **Extreme Out-of-Distribution Numerical Shocks:** While `QuantileClipper` successfully caps training-set extremes, if production test data introduces numerical values orders of magnitude outside historical training bounds (e.g., a building area recorded as 1,000,000 $m^2$ due to database corruption), standardization will generate massive Z-scores that destabilize linear regression weights.