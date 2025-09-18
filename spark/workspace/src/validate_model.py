from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# Creazione della sessione spark
spark = SparkSession.builder.appName("AirQualityValidation").getOrCreate()

# Load del dataset di validation
df_val = spark.read.csv("dataset/dataset_validation.csv", header=True, inferSchema=True, sep=";")

# Load del modello salvato
model = PipelineModel.load("models/random_forest_v2")

"""
# Stampa dell'importanza delle features per random forest
rf_model = model.stages[-1] # estrazione del classificatore dalla pipeline
importances = rf_model.featureImportances.toArray()
feature_cols = [
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5",
    "pm10", "nh3", "temperature, wind_speed", "humidity"]

print("Stampa dell'importanza delle features")
for feat, imp in zip(feature_cols, importances):
    print(f"{feat}: {imp:.4f}")
"""

# Applicazione del modello al dataset di validation
predictions = model.transform(df_val)

# Valutazione dei risultati
# Accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy sul validation set: {accuracy:.4f}")

results = predictions.withColumn("prediction", col("prediction") + 1).select("datetime", "next_aqi", "prediction")

# F1-score
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="f1"
)
f1_score = evaluator_f1.evaluate(predictions)
print(f"F1-score: {f1_score:.4f}")