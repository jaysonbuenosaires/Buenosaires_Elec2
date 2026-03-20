from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, round
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==========================================
# 1. SPARK SETUP & INGESTION
# ==========================================
spark = SparkSession.builder \
    .master("spark://192.168.254.119:7077") \
    .appName("Lab4_Buenosaires_Group") \
    .getOrCreate()

print("Loading and preprocessing data...")
# Load dataset
df = spark.read.option("header", "true").option("inferSchema", "true").csv("suicide_data_sample.csv")

# Filter and Clean
df_filtered = df.select('country', 'year', 'sex', 'age_group', 'suicide_rate') \
    .filter((col('suicide_rate') > 0) & (col('age_group') != 'ALL')) \
    .dropna()

# ==========================================
# 2. PARTITIONING & AGGREGATION
# ==========================================
print("Applying partitioning and transformations...")
# Hash Partitioning
df_hashed = df_filtered.repartition(4, "country")

# Aggregation: Top 10 Countries by Average Rate
top_countries = df_hashed.groupBy("country") \
    .agg(round(avg("suicide_rate"), 2).alias("avg_suicide_rate")) \
    .orderBy(col("avg_suicide_rate").desc()) \
    .limit(10)

# Prepare Data for Visualization
viz_df = top_countries.toPandas()
# Limit to 1000 rows for cleaner scatter/distribution plots
full_viz_df = df_filtered.limit(1000).toPandas() 

# ==========================================
# 3. VISUALIZATIONS
# ==========================================
print("Generating Matplotlib and Seaborn visualizations...")

# --- MATPLOTLIB VISUALIZATIONS (5) ---
# 1. Bar Chart
plt.figure(figsize=(12, 6))
plt.bar(viz_df['country'], viz_df['avg_suicide_rate'], color='steelblue', edgecolor='black')
plt.title('Top 10 Countries by Average Suicide Rate', fontsize=14, fontweight='bold')
plt.xlabel('Country', fontsize=12)
plt.ylabel('Average Rate (per 100k)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('1_mpl_bar.png')

# 2. Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(viz_df['avg_suicide_rate'], labels=viz_df['country'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Proportional Distribution Among Top 10 Countries', fontweight='bold')
plt.savefig('2_mpl_pie.png')

# 3. Histogram
plt.figure(figsize=(10, 6))
plt.hist(full_viz_df['suicide_rate'], bins=30, color='darkseagreen', edgecolor='black')
plt.title('Frequency Distribution of Suicide Rates', fontsize=14, fontweight='bold')
plt.xlabel('Suicide Rate')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig('3_mpl_hist.png')

# 4. Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(full_viz_df['year'], full_viz_df['suicide_rate'], alpha=0.6, c='coral', edgecolors='red')
plt.title('Suicide Rates Over Time (Sample Data)', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Suicide Rate')
plt.grid(True, linestyle='--')
plt.savefig('4_mpl_scatter.png')

# 5. Line Plot
yearly_avg = full_viz_df.groupby('year')['suicide_rate'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.plot(yearly_avg['year'], yearly_avg['suicide_rate'], marker='o', color='purple', linewidth=2)
plt.title('Global Average Suicide Rate Trend', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Average Rate')
plt.grid(True)
plt.savefig('5_mpl_line.png')

# --- SEABORN VISUALIZATIONS (5) ---
sns.set_theme(style="whitegrid")

# 6. Horizontal Barplot
plt.figure(figsize=(10, 6))
sns.barplot(data=viz_df, x='avg_suicide_rate', y='country', palette='rocket')
plt.title('Top 10 Countries (Seaborn Barplot)', fontsize=14, fontweight='bold')
plt.xlabel('Average Rate')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('6_sns_bar.png')

# 7. Lineplot with Hue (Gender)
plt.figure(figsize=(12, 6))
sns.lineplot(data=full_viz_df, x='year', y='suicide_rate', hue='sex', marker='s', palette='Set1')
plt.title('Suicide Rate Trends by Gender Over Time', fontsize=14, fontweight='bold')
plt.savefig('7_sns_line.png')

# 8. Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=full_viz_df, x='age_group', y='suicide_rate', palette='viridis')
plt.title('Spread of Suicide Rates by Age Group', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('8_sns_box.png')

# 9. Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=full_viz_df, x='sex', y='suicide_rate', split=False, inner="quartile", palette='muted')
plt.title('Density and Quartiles of Suicide Rates by Gender', fontsize=14, fontweight='bold')
plt.savefig('9_sns_violin.png')

# 10. KDE Plot (Kernel Density Estimate)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=full_viz_df, x='suicide_rate', hue='sex', fill=True, common_norm=False, palette='crest', alpha=.5)
plt.title('Density Distribution of Suicide Rates by Gender', fontsize=14, fontweight='bold')
plt.savefig('10_sns_kde.png')

print("All 10 visualizations saved successfully.")
spark.stop()