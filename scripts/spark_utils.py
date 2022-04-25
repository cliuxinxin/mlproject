from pyspark.sql import SparkSession

log_file = 'readme.me'
spark = SparkSession.builder.appName("test").getOrCreate()
log_data = spark.read.text(log_file).cache()

numAs = log_data.filter(log_data.value.contains('a')).count()
numBs = log_data.filter(log_data.value.contains('b')).count()

print("Lines with a: %i, lines with b: %i" % (numAs, numBs))

spark.stop()