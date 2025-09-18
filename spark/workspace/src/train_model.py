from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# Creazione della sessione spark
spark = SparkSession.builder.appName("AirQualityTraining").getOrCreate()

# Load dataset di training
df = spark.read.csv("dataset/dataset_training.csv", header=True, inferSchema=True, sep=";")

# 3. Definizione delle features
feature_cols = [
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5",
    "pm10", "nh3", "temperature", "wind_speed", "humidity"]

# Cast della label 'next_aqi' a intero
label_indexer = StringIndexer(inputCol="next_aqi", outputCol="label")

# Assemblamento delle feature in un unico vettore
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")


# Classificazione con random forest
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
pipeline = Pipeline(stages=[label_indexer, assembler, rf])

# Training del modello
model = pipeline.fit(df)

# Salvataggio del modello
model.save("models/random_forest_v2")

print("Modello addestrato e salvato con successo!")
