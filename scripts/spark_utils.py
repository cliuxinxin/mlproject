from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.sql.functions import explode


if __name__ == "__main__":
    spark = SparkSession.builder.appName("StructuredNetworkWordCount").getOrCreate()

    spark.sparkContext.setLogLevel('WARN')

    lines = spark.readStream.format("json").load("test")

    query = lines.writeStream.outputMode("complete").format("console").start()

    query.awaitTermination()




