import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{dayofmonth, hour, month}
import org.apache.spark.sql.types._

object CrimeTrainer {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().appName("chicagoCrime")
      .master("spark://concord.cs.colostate.edu:46010")
      .getOrCreate()

    //    val beatSchema = StructType(Array(
    //      StructField("actualBeat", IntegerType, true),
    //      StructField("newBeat", IntegerType, true)
    //    ))

    val customSchema = StructType(Array(
      StructField("id1", IntegerType, true),
      StructField("ID", IntegerType, true),
      StructField("Case Number", StringType, true),
      StructField("Date", TimestampType, true),
      StructField("Block", StringType, true),
      StructField("IUCR", StringType, true),
      StructField("Primary Type", StringType, true),
      StructField("Description", StringType, true),
      StructField("Location Description", StringType, true),
      StructField("Arrest", StringType, true),
      StructField("Domestic", StringType, true),
      StructField("Beat", DoubleType, true),
      StructField("District", DoubleType, true),
      StructField("Ward", DoubleType, true),
      StructField("Community Area", DoubleType, true),
      StructField("FBI Code", StringType, true),
      StructField("X Coordinate", DoubleType, true),
      StructField("Y Coordinate", DoubleType, true),
      StructField("Year", IntegerType, true),
      StructField("Updated On", StringType, true),
      StructField("Latitude", DoubleType, true),
      StructField("Longitude", DoubleType, true),
      StructField("Location", StringType, true)
    ))

    //    val beatData = spark.read.format("csv")
    //      .option("header", "false")
    //      .schema(beatSchema)
    //      .load("hdfs://concord:30101/cs455Project/beats/uniqueBeatsPair.csv")
    //      .persist()


    // Read in CSV file
    val data = spark.read.format("csv")
      .option("header", "true")
      .option("timestampFormat","MM/dd/yyyy hh:mm:ss a")
      .schema(customSchema)
      .load(args(0))
      .persist()


    //    val beatRdd = beatData.rdd.map{
    //      case Row(actualBeat: Int, newBeat: Int) => (actualBeat, newBeat)
    //    }.collectAsMap()
    //
    //    // Use a user defined function to replace given beat numbers with ones that we provide
    //    def func: (Int => Double) = {b => beatRdd.get(b).get.toDouble}
    //    val correctBeats = udf(func)


    val additions = data
      .withColumn("hourOfDay", hour(data("Date")))
      .withColumn("day", dayofmonth(data("Date")))
      .withColumn("month", month(data("Date")))
    //      .withColumn("adjBeats", correctBeats(crime("Beat")))

    //    val set = additions.select("year", "month", "day", "hourOfDay", "IUCR", "adjBeats")

    val set = additions.select("year", "month", "day", "hourOfDay", "IUCR", "Beat")

    //    println(set.show(25))

    val indexerIUCR = new StringIndexer()
      .setInputCol("IUCR")
      .setOutputCol("indexedIUCR")
    indexerIUCR.setHandleInvalid("skip")

    val indexedIUCR = indexerIUCR.fit(set).transform(set)

    val assembler = new VectorAssembler()
      .setInputCols(Array("year","month","day","hourOfDay","indexedIUCR"))
      .setOutputCol("features")

    val features = assembler.transform(indexedIUCR)
    features.show(5)

    val labelIndexer = new StringIndexer().setInputCol("Beat").setOutputCol("indexedBeat").fit(features)
    println(s"Found labels: ${labelIndexer.labels.mkString("[", ", ", "]")}")

    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(405).fit(features)

    val splits = features.randomSplit(Array(0.6, 0.4))
    val trainingData = splits(0)
    val testData = splits(1)



    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](5, 100, 200, 305)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setLabelCol("indexedBeat")
      .setFeaturesCol("indexedFeatures")
      .setBlockSize(128)
      .setSeed(System.currentTimeMillis())
      .setMaxIter(1500)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))

    // train the model
    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)
    predictions.show(30)

    model.save(args(1))


    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedBeat")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))


  }

}
