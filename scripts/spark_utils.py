from pyspark.sql.types import StructType
from data_utils import * 

from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.sql.functions import explode
import json



if __name__ == "__main__":
    spark = SparkSession.builder.appName("StructuredNetworkWordCount").getOrCreate()

    df = spark.read.json("../assets/train.json")

    schemaFromJson = StructType.fromJson(json.loads(df.schema.json()))

    spark.sparkContext.setLogLevel('WARN')

    nlp = b_load_best_model()

    lines = spark.readStream.format("json").schema(schemaFromJson).option("latestFirst", True).load("../data")

    lines = lines.rdd.map(lambda x: (x.data, nlp(x.data)))  

    query = lines.writeStream.outputMode("append").format("console").start()

    query.awaitTermination()