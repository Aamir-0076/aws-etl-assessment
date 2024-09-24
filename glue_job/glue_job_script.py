import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import udf, col, to_date, year, month, sum as spark_sum
from pyspark.sql.types import StringType, DoubleType
from awsglue.dynamicframe import DynamicFrame
from dateutil import parser
import datetime

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

def parse_date(date_string):
    try:
        return parser.parse(date_string, dayfirst=True).strftime('%Y-%m-%d')
    except:
        return None

# Register UDF
parse_date_udf = udf(parse_date, StringType())

# Read the input data
datasource = glueContext.create_dynamic_frame.from_catalog(
    database="demodatadb",
    table_name="databucketsourcedemo"
)

df = datasource.toDF()

# Drop duplicates
df = df.dropDuplicates()

# Apply the date parsing UDF
df = df.withColumn('sale_date', parse_date_udf(col('sale_date')))

# Convert the parsed string to date type
df = df.withColumn('sale_date', to_date(col('sale_date')))

# Rename 'price' to 'total_sales' as it already represents the total sale amount
df = df.withColumnRenamed('price', 'total_sales')

# Ensure the total_sales column is of DoubleType
df = df.withColumn('total_sales', col('total_sales').cast(DoubleType()))

# Extract year and month from sale_date to group by month
df = df.withColumn('sale_year', year(col('sale_date')))
df = df.withColumn('sale_month', month(col('sale_date')))

# Group by product_id, year, and month, and calculate total sales per product per month
aggregated_df = df.groupBy(col('product_id'), col('sale_year'), col('sale_month')) \
    .agg(spark_sum('total_sales').alias('monthly_total_sales'))

# Order by product_id, sale_year, and sale_month
aggregated_df = aggregated_df.orderBy(col('product_id'), col('sale_year'), col('sale_month'))

# Coalesce the DataFrame to a single partition
aggregated_df = aggregated_df.coalesce(1)

# Convert back to DynamicFrame
aggregated_dynamic_frame = DynamicFrame.fromDF(aggregated_df, glueContext, "aggregated_dynamic_frame")

# Write the aggregated DynamicFrame to S3 as a single CSV file
glueContext.write_dynamic_frame.from_options(
    frame=aggregated_dynamic_frame,
    connection_type="s3",
    connection_options={
        "path": "s3://databucketdestinationdemo/",
        "partitionKeys": []
    },
    format="csv",
    format_options={
        "quoteChar": -1,
        "writeHeader": True,
        "separator": ",",
        "compression": "none"
    },
    transformation_ctx="write_data"
)

job.commit()