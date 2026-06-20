import json
import warnings
import re
from pathlib import Path
from datetime import datetime
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_colwidth", 60)

SOURCE_DIR = Path("data")
OUTPUT_DIR = Path("Fact__Dimensions")
OUTPUT_DIR.mkdir(exist_ok=True)

SALES_JSON    = SOURCE_DIR / "Sales.json"
FORECAST_JSON = SOURCE_DIR / "forecast.json"
DATE_ANCHOR   = pd.Timestamp("2008-01-01")

print(f"Source  : {SOURCE_DIR.resolve()}")
print(f"Output  : {OUTPUT_DIR.resolve()}")


# Data Quality Helpers
_dq_log: list = []


def regex_check(df, col, pattern, label="", sample_n=3):
    lbl      = label or col
    compiled = re.compile(pattern)
    mask     = df[col].apply(lambda x: bool(compiled.fullmatch(str(x))) if pd.notna(x) else True)
    n_viol   = int((~mask).sum())
    pct      = n_viol / len(df) * 100
    icon     = "✅" if n_viol == 0 else "⚠️ "
    print(f"  {icon} {lbl:<38} regex={pattern[:40]:<42} violations={n_viol:>7,} ({pct:.2f}%)")
    if 0 < n_viol <= sample_n:
        print(f"      └─ Sample: {df.loc[~mask, col].head(sample_n).tolist()}")
    _dq_log.append({"check": "regex", "column": lbl, "pattern": pattern, "violations": n_viol, "pct": round(pct, 4)})
    return n_viol


def enum_check(df, col, allowed, label="", sample_n=3):
    lbl         = label or col
    mask        = df[col].apply(lambda x: x in allowed if pd.notna(x) else True)
    n_viol      = int((~mask).sum())
    pct         = n_viol / len(df) * 100
    icon        = "✅" if n_viol == 0 else "⚠️ "
    allowed_str = str(sorted(allowed))[:60]
    print(f"  {icon} {lbl:<38} enum={allowed_str:<42} violations={n_viol:>7,} ({pct:.2f}%)")
    if 0 < n_viol <= sample_n:
        print(f"      └─ Sample: {df.loc[~mask, col].head(sample_n).tolist()}")
    _dq_log.append({"check": "enum", "column": lbl, "allowed": sorted(allowed), "violations": n_viol, "pct": round(pct, 4)})
    return n_viol


def range_check(df, col, min_val=None, max_val=None, label="", sample_n=3):
    lbl    = label or col
    mask   = df[col].notna()
    if min_val is not None: mask &= df[col] >= min_val
    if max_val is not None: mask &= df[col] <= max_val
    n_viol = int((~mask).sum())
    pct    = n_viol / len(df) * 100
    icon   = "✅" if n_viol == 0 else "⚠️ "
    rng    = f"[{min_val}, {max_val}]"
    print(f"  {icon} {lbl:<38} range={rng:<42} violations={n_viol:>7,} ({pct:.2f}%)")
    if 0 < n_viol <= sample_n:
        print(f"      └─ Sample: {df.loc[~mask, col].head(sample_n).tolist()}")
    _dq_log.append({"check": "range", "column": lbl, "range": rng, "violations": n_viol, "pct": round(pct, 4)})
    return n_viol


def null_check(df, col, label=""):
    lbl    = label or col
    n_null = int(df[col].isna().sum())
    icon   = "✅" if n_null == 0 else "❌"
    print(f"  {icon} {lbl:<38} null check{'':42} nulls={n_null:>7,}")
    _dq_log.append({"check": "null", "column": lbl, "violations": n_null})
    return n_null


def uniqueness_check(df, col, label=""):
    lbl   = label or col
    n_dup = int(df[col].duplicated().sum())
    icon  = "✅" if n_dup == 0 else "❌"
    print(f"  {icon} {lbl:<38} uniqueness check{'':41} dups={n_dup:>7,}")
    _dq_log.append({"check": "unique", "column": lbl, "violations": n_dup})
    return n_dup


def fk_check(child_df, child_col, parent_df, parent_col, child_table="", parent_table=""):
    parent_vals = set(parent_df[parent_col].dropna())
    orphans     = int((~child_df[child_col].dropna().isin(parent_vals)).sum())
    rel         = f"{child_table}.{child_col} → {parent_table}.{parent_col}"
    icon        = "✅" if orphans == 0 else "❌"
    print(f"  {icon} FK  {rel:<65}  orphans={orphans:>6,}")
    _dq_log.append({"check": "fk", "relationship": rel, "violations": orphans})
    return orphans


# BRONZE: Raw Ingestion 

print("\n⏳ Loading Sales.json ...")
t0 = datetime.now()
with open(SALES_JSON, "r", encoding="utf-8") as fh:
    bronze_sales = pd.DataFrame(json.load(fh))
print(f"   ✅ Loaded in {(datetime.now() - t0).total_seconds():.1f}s — {len(bronze_sales):,} records")

print("⏳ Loading forecast.json ...")
with open(FORECAST_JSON, "r", encoding="utf-8") as fh:
    bronze_forecast = pd.DataFrame(json.load(fh))
print(f"   ✅ Loaded — {len(bronze_forecast):,} records")

print("\n=== Bronze — Sales Schema ===")
bronze_sales.info()

print("\n=== Bronze — Forecast Schema ===")
bronze_forecast.info()

print("\n=== Bronze Volume & Cardinality ===")
metrics = {
    "Total Sales Transactions"      : len(bronze_sales),
    "Unique Products (ProductKey)"  : bronze_sales["ProductKey"].nunique(),
    "Unique Customers (CustomerKey)": bronze_sales["CustomerKey"].nunique(),
    "Unique Cities"                 : bronze_sales["City"].nunique(),
    "Unique Countries"              : bronze_sales["CountryRegion"].nunique(),
    "Unique Brands"                 : bronze_sales["Brand"].nunique(),
    "Unique Order Dates"            : bronze_sales["OrderDate"].nunique(),
    "Total Forecast Records"        : len(bronze_forecast),
    "Forecast Countries"            : bronze_forecast["CountryRegion"].nunique(),
    "Forecast Brands"               : bronze_forecast["Brand"].nunique(),
}
for k, v in metrics.items():
    print(f"  {k:<45} {v:>10,}")

print("\n=== Bronze Sales — Null Census ===")
null_df = pd.DataFrame({
    "Column"        : bronze_sales.columns,
    "Null Count"    : bronze_sales.isnull().sum().values,
    "Null %"        : (bronze_sales.isnull().mean() * 100).round(2).values,
    "Non-Null Count": bronze_sales.notnull().sum().values,
    "Dtype"         : bronze_sales.dtypes.astype(str).values,
}).sort_values("Null Count", ascending=False)
print(null_df.reset_index(drop=True).to_string())


# ─── SILVER: Cleansing & Normalisation ──────────────────────────────────────

silver_sales    = bronze_sales.copy()
silver_forecast = bronze_forecast.copy()

for col in silver_sales.select_dtypes(include="object").columns:
    silver_sales[col] = silver_sales[col].str.strip()
for col in silver_forecast.select_dtypes(include="object").columns:
    silver_forecast[col] = silver_forecast[col].str.strip()

color_mismatch_pct = (silver_sales["Color"] != silver_sales["Subcategory"]).mean() * 100
print(f"\nDQ — Color ≠ Subcategory rows: {color_mismatch_pct:.1f}%")
silver_sales["Color_DQ_Flag"] = (
    silver_sales["Color"] == silver_sales["Subcategory"]
).map({True: "SUSPECT_EQUALS_SUBCATEGORY", False: "OK"})

silver_sales["OrderDate_dt"] = pd.to_datetime(silver_sales["OrderDate"], format="%m/%d/%Y", errors="coerce")
silver_sales["OrderDateKey"] = ((silver_sales["OrderDate_dt"] - DATE_ANCHOR).dt.days + 1).astype("int32")
print(f"Date range   : {silver_sales['OrderDate_dt'].min().date()} → {silver_sales['OrderDate_dt'].max().date()}")
print(f"DateKey range: {silver_sales['OrderDateKey'].min()} → {silver_sales['OrderDateKey'].max()}")

silver_forecast["ForecastDate"]    = pd.to_datetime(silver_forecast["Year"].astype(str) + "-01-01")
silver_forecast["ForecastDateKey"] = ((silver_forecast["ForecastDate"] - DATE_ANCHOR).dt.days + 1).astype("int32")

for col in ["Name", "Education", "Occupation"]:
    silver_sales[col] = silver_sales[col].fillna("Unknown")

type_map = {
    "ProductKey": "int32", "CustomerKey": "int32",
    "Quantity": "int32", "Net Price": "float64", "OrderDateKey": "int32",
}
for col, dtype in type_map.items():
    silver_sales[col] = silver_sales[col].astype(dtype)

silver_forecast["Forecast"]        = silver_forecast["Forecast"].astype("int64")
silver_forecast["ForecastDateKey"] = silver_forecast["ForecastDateKey"].astype("int32")
silver_forecast["Year"]            = silver_forecast["Year"].astype("int32")

print("\n=== Outlier Analysis (IQR) ===")
for col in ["Quantity", "Net Price"]:
    q1, q3 = silver_sales[col].quantile(0.25), silver_sales[col].quantile(0.75)
    iqr    = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out  = ((silver_sales[col] < lo) | (silver_sales[col] > hi)).sum()
    print(f"  {col:<15} Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}  Fence=[{lo:.2f}, {hi:.2f}]  Outliers={n_out:,}")

_dq_log.clear()
VALID_CONTINENTS = {"Asia", "Europe", "North America", "South America", "Africa", "Australia", "Antarctica"}
VALID_EDU        = {"Bachelors", "Graduate Degree", "Partial College", "High School", "Partial High School", "Unknown"}
VALID_OCC        = {"Management", "Professional", "Skilled Manual", "Clerical", "Manual", "Unknown"}

print("\n=== Silver DQ — sales ===")
regex_check(silver_sales, "ProductKey",  r"\d+",                       "ProductKey")
regex_check(silver_sales, "CustomerKey", r"\d+",                       "CustomerKey")
range_check(silver_sales, "ProductKey",  min_val=1,                    label="ProductKey ≥ 1")
range_check(silver_sales, "CustomerKey", min_val=1,                    label="CustomerKey ≥ 1")
regex_check(silver_sales, "OrderDate",   r"\d{1,2}/\d{1,2}/\d{4}",    "OrderDate (M/D/YYYY)")
range_check(silver_sales, "OrderDateKey", min_val=1, max_val=731,      label="OrderDateKey ∈ [1, 731]")
regex_check(silver_sales, "Product Name", r"[A-Za-z0-9\s\.&\-\(\)\'\\+\,\_\#\!\%]+", "Product Name")
regex_check(silver_sales, "Brand",        r"[A-Za-z0-9\s\.&\-\']+",   "Brand")
regex_check(silver_sales, "Category",     r"[A-Za-z0-9\s&\-]+",       "Category")
regex_check(silver_sales, "Subcategory",  r"[A-Za-z0-9\s&\-\_\/\%]+", "Subcategory")
regex_check(silver_sales, "Color",        r"[A-Za-z0-9\s&\-\_\/\%]+", "Color")
regex_check(silver_sales, "City",         r"[A-Za-z0-9\s\.\-\']+",    "City")
regex_check(silver_sales, "State",        r"[A-Za-z0-9\s\.\-\']+",    "State")
regex_check(silver_sales, "CountryRegion", r"[A-Za-z\s]+",            "CountryRegion")
enum_check( silver_sales, "Continent", VALID_CONTINENTS,               "Continent (7 values)")
range_check(silver_sales, "Quantity",  min_val=1,                      label="Quantity ≥ 1")
range_check(silver_sales, "Net Price", min_val=0.0,                    label="Net Price ≥ 0")
regex_check(silver_sales, "Name",      r"Unknown|[A-Za-z][A-Za-z\s\-\.\' ]+", "Name")
enum_check( silver_sales, "Education", VALID_EDU,                      "Education")
enum_check( silver_sales, "Occupation", VALID_OCC,                     "Occupation")

print("\n=== Silver DQ — forecast ===")
regex_check(silver_forecast, "CountryRegion", r"[A-Za-z\s]+",              "CountryRegion")
regex_check(silver_forecast, "Brand",         r"[A-Za-z0-9\s\.&\-\']+",   "Brand")
range_check(silver_forecast, "Year",          min_val=2009, max_val=2009,  label="Year == 2009")
range_check(silver_forecast, "ForecastDateKey", min_val=367, max_val=367, label="ForecastDateKey == 367")
range_check(silver_forecast, "Forecast",      min_val=1,                   label="Forecast ≥ 1")
null_check( silver_forecast, "CountryRegion")
null_check( silver_forecast, "Brand")
null_check( silver_forecast, "Forecast")

total_viol = sum(r["violations"] for r in _dq_log)
print(f"\n  DQ checks run: {len(_dq_log)}  |  Total violations: {total_viol:,}")


# GOLD: Dimensional Modelling 

# Dim_Brand
dim_brand = (
    silver_sales[["Brand"]]
    .drop_duplicates()
    .sort_values("Brand")
    .reset_index(drop=True)
)
dim_brand.insert(0, "BrandKey", range(1, len(dim_brand) + 1))
dim_brand["BrandKey"] = dim_brand["BrandKey"].astype("int32")
print(f"\nDim_Brand: {len(dim_brand)} rows | PK unique={dim_brand['BrandKey'].is_unique}")

print("\n=== Dim_Brand DQ ===")
null_check(dim_brand, "BrandKey"); uniqueness_check(dim_brand, "BrandKey")
null_check(dim_brand, "Brand"); regex_check(dim_brand, "Brand", r"[A-Za-z0-9\s\.&\-\']+", "Brand")


# Dim_Country
dim_country = (
    silver_sales[["CountryRegion"]]
    .rename(columns={"CountryRegion": "Country"})
    .drop_duplicates()
    .sort_values("Country")
    .reset_index(drop=True)
)
dim_country.insert(0, "CountryKey", range(1, len(dim_country) + 1))
dim_country["CountryKey"] = dim_country["CountryKey"].astype("int32")
print(f"\nDim_Country: {len(dim_country)} rows | PK unique={dim_country['CountryKey'].is_unique}")
print(f"  Countries: {dim_country['Country'].tolist()}")

print("\n=== Dim_Country DQ ===")
null_check(dim_country, "CountryKey"); uniqueness_check(dim_country, "CountryKey")
null_check(dim_country, "Country"); regex_check(dim_country, "Country", r"[A-Za-z\s]+", "Country")


# Dim_Customer — one row per customer, best non-null value per column from bronze
cust_raw = bronze_sales[["CustomerKey", "Name", "Education", "Occupation"]].copy()

dim_customer = (
    cust_raw
    .groupby("CustomerKey", as_index=False)
    .first()                   # picks first non-null per column across all rows for that customer
    .sort_values("CustomerKey")
    .reset_index(drop=True)
)
dim_customer["CustomerKey"] = dim_customer["CustomerKey"].astype("int32")

# Fill nulls: unknown name → "Customer_<key>", missing profile → "Unknown"
dim_customer["Name"]      = dim_customer.apply(
    lambda r: f"Customer_{r['CustomerKey']}" if pd.isna(r["Name"]) else r["Name"], axis=1
)
dim_customer["Education"]  = dim_customer["Education"].fillna("Unknown")
dim_customer["Occupation"] = dim_customer["Occupation"].fillna("Unknown")

# Customer_Type: guest if Education is Unknown (no profile data), registered otherwise
dim_customer["Customer_Type"] = dim_customer["Education"].apply(
    lambda edu: "guest" if str(edu).strip().lower() == "unknown" else "registered"
)

fact_ck = set(silver_sales["CustomerKey"].unique())
missing = fact_ck - set(dim_customer["CustomerKey"].unique())
print(f"\nDim_Customer: {len(dim_customer):,} rows | PK unique={dim_customer['CustomerKey'].is_unique}")
print(f"  FK coverage gap (sales - dim): {len(missing)} {'✅' if not missing else '⚠️'}")
print(f"  Customer_Type distribution:\n{dim_customer['Customer_Type'].value_counts().to_string()}")

VALID_EDU_DIM = {"Bachelors", "Graduate Degree", "Partial College", "High School", "Partial High School", "Unknown"}
VALID_OCC_DIM = {"Management", "Professional", "Skilled Manual", "Clerical", "Manual", "Unknown"}

print("\n=== Dim_Customer DQ ===")
null_check(dim_customer, "CustomerKey"); uniqueness_check(dim_customer, "CustomerKey")
null_check(dim_customer, "Name"); regex_check(dim_customer, "Name", r"Customer_\d+|[A-Za-z][A-Za-z\-\'\s\.]+", "Name")
enum_check(dim_customer, "Education", VALID_EDU_DIM, "Education")
enum_check(dim_customer, "Occupation", VALID_OCC_DIM, "Occupation")
enum_check(dim_customer, "Customer_Type", {"guest", "registered"}, "Customer_Type (2-value domain)")


# Dim_Product
# Color is derived from the last word of Product Name (title-cased).
# If it matches a known color → use it; otherwise the product has no color → "No Color".
KNOWN_COLORS = {"Azure","Black","Blue","Brown","Gold","Green","Grey","Orange","Pink","Purple","Red","Silver","White","Yellow"}

dim_product = (
    silver_sales[["ProductKey", "Product Name", "Brand", "Category", "Subcategory"]]
    .drop_duplicates(subset=["ProductKey"])
    .sort_values("ProductKey")
    .reset_index(drop=True)
    .merge(dim_brand[["BrandKey", "Brand"]], on="Brand", how="left")
    [["ProductKey", "Product Name", "BrandKey", "Category", "Subcategory"]]
    .copy()
)
dim_product["Color"] = (
    dim_product["Product Name"]
    .str.split()
    .str[-1]
    .str.title()
    .apply(lambda w: w if w in KNOWN_COLORS else "No Color")
)
dim_product["ProductKey"] = dim_product["ProductKey"].astype("int32")
dim_product["BrandKey"]   = dim_product["BrandKey"].astype("int32")
print(f"\nDim_Product: {len(dim_product):,} rows | Unresolved BrandKey: {dim_product['BrandKey'].isna().sum()}")
print(f"  Color distribution:\n{dim_product['Color'].value_counts().to_string()}")

print("\n=== Dim_Product DQ ===")
null_check(dim_product, "ProductKey"); uniqueness_check(dim_product, "ProductKey")
null_check(dim_product, "BrandKey");   fk_check(dim_product, "BrandKey", dim_brand, "BrandKey", "Dim_Product", "Dim_Brand")
regex_check(dim_product, "Category",   r"[A-Za-z0-9\s&\-]+",      "Category")
regex_check(dim_product, "Subcategory", r"[A-Za-z0-9\s&\-\_\/\%]+", "Subcategory")
enum_check(dim_product, "Color", KNOWN_COLORS | {"No Color"}, "Color (known palette + No Color)")


# Dim_City
dim_city_working = (
    silver_sales[["City", "State", "Continent", "CountryRegion"]]
    .drop_duplicates()
    .sort_values(["CountryRegion", "State", "City"])
    .reset_index(drop=True)
)
dim_city_working.insert(0, "CityKey", range(1, len(dim_city_working) + 1))
dim_city_working = dim_city_working.merge(
    dim_country[["CountryKey", "Country"]].rename(columns={"Country": "CountryRegion"}),
    on="CountryRegion", how="left"
)
city_key_lookup = dim_city_working[["City", "State", "CountryRegion", "CityKey"]].copy()

dim_city = dim_city_working[["CityKey", "City", "State", "Continent", "CountryKey"]].copy()
dim_city["CityKey"]    = dim_city["CityKey"].astype("int32")
dim_city["CountryKey"] = dim_city["CountryKey"].astype("int32")
print(f"\nDim_City: {len(dim_city)} rows | Unresolved CountryKey: {dim_city['CountryKey'].isna().sum()}")

print("\n=== Dim_City DQ ===")
null_check(dim_city, "CityKey"); uniqueness_check(dim_city, "CityKey")
null_check(dim_city, "CountryKey"); fk_check(dim_city, "CountryKey", dim_country, "CountryKey", "Dim_City", "Dim_Country")
regex_check(dim_city, "City",  r"[A-Za-z0-9\s\.\-\']+", "City")
regex_check(dim_city, "State", r"[A-Za-z0-9\s\.\-\']+", "State")
enum_check( dim_city, "Continent", VALID_CONTINENTS,     "Continent (7 values)")


# Dim_Date
date_range = pd.date_range(
    start=silver_sales["OrderDate_dt"].min(),
    end=silver_sales["OrderDate_dt"].max(),
    freq="D"
)
dim_date = pd.DataFrame({"Date": date_range})
dim_date["DateKey"]    = ((dim_date["Date"] - DATE_ANCHOR).dt.days + 1).astype("int32")
dim_date["Year"]       = dim_date["Date"].dt.year.astype("int32")
dim_date["Quarter"]    = dim_date["Date"].dt.quarter.astype("int32")
dim_date["Month"]      = dim_date["Date"].dt.month.astype("int32")
dim_date["Month Name"] = dim_date["Date"].dt.strftime("%B")
dim_date["Day"]        = dim_date["Date"].dt.day.astype("int32")
dim_date["Day Name"]   = dim_date["Date"].dt.strftime("%A")
print(f"\nDim_Date: {len(dim_date)} rows | Range: {dim_date['Date'].min().date()} → {dim_date['Date'].max().date()}")

VALID_MONTHS = {"January","February","March","April","May","June","July","August","September","October","November","December"}
VALID_DAYS   = {"Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"}

print("\n=== Dim_Date DQ ===")
null_check(dim_date, "DateKey"); uniqueness_check(dim_date, "DateKey")
range_check(dim_date, "DateKey", min_val=1, max_val=731, label="DateKey ∈ [1, 731]")
range_check(dim_date, "Year",    min_val=2008, max_val=2009, label="Year ∈ {2008, 2009}")
range_check(dim_date, "Quarter", min_val=1,    max_val=4)
range_check(dim_date, "Month",   min_val=1,    max_val=12)
range_check(dim_date, "Day",     min_val=1,    max_val=31)
enum_check( dim_date, "Month Name", VALID_MONTHS, "Month Name")
enum_check( dim_date, "Day Name",   VALID_DAYS,   "Day Name")
gaps = set(range(1, len(dim_date) + 1)) - set(dim_date["DateKey"])
print(f"  {'✅' if not gaps else '❌'}  DateKey sequence gaps: {len(gaps)}")


# Fact_Sales
fact_sales = silver_sales[["ProductKey", "CustomerKey", "City", "State", "CountryRegion", "OrderDateKey", "Quantity", "Net Price"]].copy()
fact_sales = fact_sales.merge(city_key_lookup, on=["City", "State", "CountryRegion"], how="left")
fact_sales.insert(0, "SaleKey", range(1, len(fact_sales) + 1))
fact_sales["SaleKey"]     = fact_sales["SaleKey"].astype("int32")
fact_sales["CustomerKey"] = fact_sales["CustomerKey"].astype("int32")
fact_sales["CityKey"]     = fact_sales["CityKey"].astype("int32")
fact_sales = fact_sales[["SaleKey", "OrderDateKey", "ProductKey", "CityKey", "CustomerKey", "Quantity", "Net Price"]]
print(f"\nFact_Sales: {len(fact_sales):,} rows | SaleKey unique={fact_sales['SaleKey'].is_unique}")

print("\n=== Fact_Sales DQ ===")
null_check(fact_sales, "SaleKey"); uniqueness_check(fact_sales, "SaleKey")
fk_check(fact_sales, "OrderDateKey", dim_date,     "DateKey",     "Fact_Sales", "Dim_Date")
fk_check(fact_sales, "ProductKey",   dim_product,  "ProductKey",  "Fact_Sales", "Dim_Product")
fk_check(fact_sales, "CityKey",      dim_city,     "CityKey",     "Fact_Sales", "Dim_City")
fk_check(fact_sales, "CustomerKey",  dim_customer, "CustomerKey", "Fact_Sales", "Dim_Customer")
range_check(fact_sales, "Quantity",  min_val=1,   label="Quantity ≥ 1")
range_check(fact_sales, "Net Price", min_val=0.0, label="Net Price ≥ 0")


# Fact_Forecast_2009
fact_forecast = (
    silver_forecast[["Brand", "CountryRegion", "Forecast", "ForecastDateKey"]]
    .copy()
    .merge(dim_brand[["BrandKey", "Brand"]], on="Brand", how="left")
    .merge(dim_country[["CountryKey", "Country"]].rename(columns={"Country": "CountryRegion"}), on="CountryRegion", how="left")
    [["ForecastDateKey", "BrandKey", "CountryKey", "Forecast"]]
    .copy()
)
fact_forecast["Forecast"]        = fact_forecast["Forecast"].astype("int64")
fact_forecast["ForecastDateKey"] = fact_forecast["ForecastDateKey"].astype("int32")
fact_forecast["BrandKey"]        = fact_forecast["BrandKey"].astype("int32")
fact_forecast["CountryKey"]      = fact_forecast["CountryKey"].astype("int32")
print(f"\nFact_Forecast_2009: {len(fact_forecast)} rows")

print("\n=== Fact_Forecast DQ ===")
fk_check(fact_forecast, "ForecastDateKey", dim_date,    "DateKey",    "Fact_Forecast_2009", "Dim_Date")
fk_check(fact_forecast, "BrandKey",        dim_brand,   "BrandKey",   "Fact_Forecast_2009", "Dim_Brand")
fk_check(fact_forecast, "CountryKey",      dim_country, "CountryKey", "Fact_Forecast_2009", "Dim_Country")
for col in ["ForecastDateKey", "BrandKey", "CountryKey", "Forecast"]:
    null_check(fact_forecast, col)
range_check(fact_forecast, "Forecast", min_val=1, label="Forecast ≥ 1")
dup_grain = fact_forecast.duplicated(subset=["BrandKey", "CountryKey"]).sum()
print(f"  {'✅' if dup_grain == 0 else '❌'}  Grain (BrandKey × CountryKey) duplicates: {dup_grain}")


# Gold Layer Schema Validation

gold_tables = {
    "Dim_Brand"         : dim_brand,
    "Dim_Country"       : dim_country,
    "Dim_Customer"      : dim_customer,
    "Dim_Product"       : dim_product,
    "Dim_City"          : dim_city,
    "Dim_Date"          : dim_date,
    "Fact_Sales"        : fact_sales,
    "Fact_Forecast_2009": fact_forecast,
}

print("\n" + "=" * 72)
print("  GOLD LAYER — SCHEMA VALIDATION")
print("=" * 72)
print(f"\n  {'Table':<24} {'Rows':>9}  {'Cols':>4}  {'Nulls':>7}  Columns")
for name, df in gold_tables.items():
    print(f"  {name:<24} {len(df):>9,}  {len(df.columns):>4}  {df.isnull().sum().sum():>7,}  {list(df.columns)}")

pk_map = [
    ("Dim_Brand",    "BrandKey"),
    ("Dim_Country",  "CountryKey"),
    ("Dim_Customer", "CustomerKey"),
    ("Dim_Product",  "ProductKey"),
    ("Dim_City",     "CityKey"),
    ("Dim_Date",     "DateKey"),
    ("Fact_Sales",   "SaleKey"),
]
schema_ok = True
print("\n── PK Uniqueness ──")
for tname, pk_col in pk_map:
    df      = gold_tables[tname]
    is_uniq = df[pk_col].is_unique
    no_null = df[pk_col].notna().all()
    ok      = is_uniq and no_null
    if not ok: schema_ok = False
    print(f"  {'✅' if ok else '❌'}  {tname:<24} PK={pk_col:<16}  unique={str(is_uniq):<5}  no_nulls={no_null}")

fk_map = [
    ("Fact_Sales",         "OrderDateKey",    "Dim_Date",     "DateKey"),
    ("Fact_Sales",         "ProductKey",      "Dim_Product",  "ProductKey"),
    ("Fact_Sales",         "CityKey",         "Dim_City",     "CityKey"),
    ("Fact_Sales",         "CustomerKey",     "Dim_Customer", "CustomerKey"),
    ("Fact_Forecast_2009", "ForecastDateKey", "Dim_Date",     "DateKey"),
    ("Fact_Forecast_2009", "BrandKey",        "Dim_Brand",    "BrandKey"),
    ("Fact_Forecast_2009", "CountryKey",      "Dim_Country",  "CountryKey"),
    ("Dim_Product",        "BrandKey",        "Dim_Brand",    "BrandKey"),
    ("Dim_City",           "CountryKey",      "Dim_Country",  "CountryKey"),
]
print("\n── Referential Integrity ──")
for ft, fk_col, dt, pk_col in fk_map:
    child_vals  = gold_tables[ft][fk_col].dropna()
    parent_vals = set(gold_tables[dt][pk_col].values)
    orphans     = int((~child_vals.isin(parent_vals)).sum())
    if orphans > 0: schema_ok = False
    print(f"  {'✅' if orphans == 0 else '⚠️ '}  {ft}.{fk_col:<20}  →  {dt}.{pk_col:<16}  orphans={orphans:,}")

print()
print("✅ Validation PASSED." if schema_ok else "⚠️  Validation FAILED — review flagged items above.")

# Export 
export_map = {
    "Dim_Brand.csv"         : dim_brand,
    "Dim_Country.csv"       : dim_country,
    "Dim_Customer.csv"      : dim_customer,
    "Dim_Product.csv"       : dim_product,
    "Dim_City.csv"          : dim_city,
    "Dim_Date.csv"          : dim_date,
    "Fact_Sales.csv"        : fact_sales,
    "Fact_Forecast_2009.csv": fact_forecast,
}

print("\n=== Exporting Gold Tables ===")
for filename, df in export_map.items():
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  ✅  {filename:<30}  {len(df):>10,} rows  {path.stat().st_size / 1024:>9.1f} KB")

print(f"\n📁 Saved to: {OUTPUT_DIR.resolve()}")

print("\n=== Round-Trip Verification ===")
all_ok = True
for path in sorted(OUTPUT_DIR.glob("*.csv")):
    df_check = pd.read_csv(path)
    in_mem   = export_map.get(path.name)
    row_ok   = in_mem is not None and len(df_check) == len(in_mem)
    if not row_ok: all_ok = False
    print(f"  {'✅' if row_ok else '❌'} {path.name:<30}  {str(df_check.shape):<20}  nulls={df_check.isnull().sum().sum():,}")

print("\n✅ All exports verified." if all_ok else "⚠️  Row-count mismatch — review above.")


# Pipeline Summary 

print("\n" + "=" * 72)
print("  MEDALLION ETL — PIPELINE SUMMARY")
print("=" * 72)

total_gold = sum(len(df) for df in export_map.values())
print(f"\n  Total Gold rows exported : {total_gold:,}")
print(f"  Output files             : {len(export_map)}")
print(f"  Output directory         : {OUTPUT_DIR.resolve()}")

print("\n  Galaxy Schema FK Map:")
for line in [
    "Fact_Sales.OrderDateKey      → Dim_Date.DateKey",
    "Fact_Sales.ProductKey        → Dim_Product.ProductKey",
    "Fact_Sales.CityKey           → Dim_City.CityKey",
    "Fact_Sales.CustomerKey       → Dim_Customer.CustomerKey",
    "Dim_Product.BrandKey         → Dim_Brand.BrandKey",
    "Dim_City.CountryKey          → Dim_Country.CountryKey",
    "Fact_Forecast.ForecastDateKey → Dim_Date.DateKey",
    "Fact_Forecast.BrandKey       → Dim_Brand.BrandKey",
    "Fact_Forecast.CountryKey     → Dim_Country.CountryKey",
]:
    print(f"    {line}")

print("\n✅ Pipeline complete — Galaxy schema ready for BI consumption.")