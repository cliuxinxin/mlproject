from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("test").getOrCreate()

train = '../assets/train.json'

df = spark.read.json(train)

df.show()

df['data_source'].show()

df.printSchema()

df.select('data_source').show()

df.select(df['data'],df['dataset']'test').show()

df.createTempView('train')

sql_df = spark.sql("SELECT * FROM train")

sql_df.show()