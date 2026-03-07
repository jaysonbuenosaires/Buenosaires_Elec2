from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, round


spark = SparkSession.builder \
    .master("spark://192.168.254.119:7077") \
    .appName("Lab2_Buenosaires_Group") \
    .getOrCreate()

# Load the CSV dataset [cite: 125, 126]
# Ensure 'sales_data_sample.csv' is in your VS Code project folder
df = spark.read.option("header", "true").option("inferSchema", "true").csv("sales_data_sample.csv")

# --- STRATEGY 1: Hash Partitioning ---
# Data is distributed based on the hash value of a key (STATUS) [cite: 49, 50]
# Ideal for groupByKey and local processing of related data [cite: 51]
df_hashed = df.repartition(4, "STATUS")

# Transformation Pipeline: Summarization
status_summary = df_hashed.groupBy("STATUS") \
    .agg(round(sum("SALES"), 2).alias("Total_Sales"), 
         round(avg("SALES"), 2).alias("Avg_Sales")) \
    .filter(col("Total_Sales") > 5000)

# --- STRATEGY 2: Range Partitioning ---
# Data is split based on sorted ranges of the 'ORDERDATE' key [cite: 54, 55]
# Reduces shuffle operations for sorted data like sortBy [cite: 56, 57]
df_ranged = df.repartitionByRange(3, "ORDERDATE")

# Transformation Pipeline: Filter and Sort
processed_data = df_ranged.filter(col("COUNTRY") == "USA") \
    .sort("ORDERDATE", ascending=True)

# Organized Result Display
print("\n" + "="*60)
print(" HASH PARTITIONING: SALES SUMMARY BY STATUS ")
print("="*60)
status_summary.show(truncate=False)

print("\n" + "="*60)
print(" RANGE PARTITIONING: CHRONOLOGICAL USA SALES ")
print("="*60)
processed_data.select("ORDERDATE", "COUNTRY", "SALES").show(5, truncate=False)

spark.stop()