from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, expr
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, StringType, TimestampType
from pyspark.ml import PipelineModel
import json
import requests

# Funzione per inviare i dati a Elasticsearch
def send_to_es(partition):
    for row in partition:
        try:
            # conversione in dizionario ed eventuale gestione dei timestamp
            doc = {col: (val.isoformat() if hasattr(val, "isoformat") else val) 
                   for col, val in row.asDict().items()}

            resp = requests.post(
                "http://elasticsearch:9200/airquality/_doc", 
                headers={"Content-Type": "application/json"},
                data=json.dumps(doc)
            )
            print(f"[ES] Inviato {resp.status_code}: {resp.text}")
        except Exception as e:
            print("Errore connessione a Elasticsearch:", e)


# Creazione della sessione spark
spark = SparkSession.builder \
    .appName("KafkaMultipleTopics") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Definizione degli schemi json
weather_schema = StructType([
    StructField("temperature", FloatType(), True),
    StructField("humidity", IntegerType(), True),
    StructField("wind_speed", FloatType(), True),
    StructField("city", StringType(), True),
    StructField("measurement_time", TimestampType(), True),
    StructField("@timestamp", TimestampType(), True)
])

air_schema = StructType([
    StructField("aqi", IntegerType(), True),
    StructField("co", FloatType(), True),
    StructField("no", FloatType(), True),
    StructField("no2", FloatType(), True),
    StructField("o3", FloatType(), True),
    StructField("so2", FloatType(), True),
    StructField("pm2_5", FloatType(), True),
    StructField("pm10", FloatType(), True),
    StructField("nh3", FloatType(), True),
    StructField("city", StringType(), True),
    StructField("location", StringType(), True),
    StructField("measurement_time", TimestampType(), True),
    StructField("@timestamp", TimestampType(), True)
])

# Letture degli stream da kafka
weather_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "weatherdata") \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()

air_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "airquality") \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()

# Decodifica dei dati letti da kafka
weather_df_parsed = weather_df.select(
    from_json(col("value").cast("string"), weather_schema).alias("data")
).select("data.*")

air_df_parsed = air_df.select(
    from_json(col("value").cast("string"), air_schema).alias("data")
).select("data.*")

# Filtraggio dei duplicati
weather = weather_df_parsed.dropDuplicates(["city", "measurement_time"])
air = air_df_parsed.dropDuplicates(["city", "measurement_time"])

# Join tra i due dataframe
joined = weather.join(
    air,
    (weather["city"] == air["city"]) &
    (weather["@timestamp"].between(air["@timestamp"] - expr("INTERVAL 2 SECONDS"),
                                   air["@timestamp"] + expr("INTERVAL 2 SECONDS")))
)

# Selezione dei campi da conservare nel dataframe finale
joined_df = joined.select(
    weather["measurement_time"].alias("weather_measurement_time"),
    air["measurement_time"].alias("air_measurement_time"),
    weather["city"].alias("city"),
    air["location"].alias("location"),
    weather["@timestamp"].alias("timestamp"),
    weather["temperature"],
    weather["humidity"],
    weather["wind_speed"],
    air["aqi"], air["co"], air["no"], air["no2"], air["o3"],
    air["so2"], air["pm2_5"], air["pm10"], air["nh3"]
)

# Load del modello salvato
model = PipelineModel.load("models/random_forest_v2")

# Applicazione del modello
predictions = model.transform(joined_df)

# Incremento della prediction di 1
results = predictions.withColumn("prediction", col("prediction") + 1)

# Risultato finale
results = results.select("weather_measurement_time", "air_measurement_time","aqi","co","no","no2","o3","so2","pm2_5","pm10","nh3","city", "location", "temperature", "wind_speed", "humidity","timestamp","prediction")


# Output su Elasticsearch
query_es = (results.writeStream
    .foreachBatch(lambda batch_df, _: batch_df.foreachPartition(send_to_es))
    #.option("checkpointLocation", "/opt/spark-checkpoints/airquality")
    .outputMode("append")
    .option("checkpointLocation", "/tmp/spark-checkpoints/es")
    .start()
)

# Output su console 
query_console = (
    results.writeStream
    .format("console")
    .outputMode("append")
    .option("truncate", False)  
    .option("checkpointLocation", "/tmp/spark-checkpoints/console")
    .start()
)

spark.streams.awaitAnyTermination()
