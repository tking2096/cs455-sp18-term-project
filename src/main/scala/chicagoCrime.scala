import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.{dayofmonth, hour, map, month, udf}

object chicagoCrime {

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
    val crime = spark.read.format("csv")
      .option("header", "true")
      .option("timestampFormat","MM/dd/yyyy hh:mm:ss a")
      .schema(customSchema)
      .load("hdfs://concord:30101/cs455Project/data/*.csv")
      .persist()


//    val beatRdd = beatData.rdd.map{
//      case Row(actualBeat: Int, newBeat: Int) => (actualBeat, newBeat)
//    }.collectAsMap()
//
//    // Use a user defined function to replace given beat numbers with ones that we provide
//    def func: (Int => Double) = {b => beatRdd.get(b).get.toDouble}
//    val correctBeats = udf(func)


    val additions = crime
      .withColumn("hourOfDay", hour(crime("Date")))
      .withColumn("day", dayofmonth(crime("Date")))
      .withColumn("month", month(crime("Date")))
//      .withColumn("adjBeats", correctBeats(crime("Beat")))

//    val set = additions.select("year", "month", "day", "hourOfDay", "IUCR", "adjBeats")

    val set = additions.select("year", "month", "day", "hourOfDay", "IUCR", "Beat")

//    println(set.show(25))

    val indexerIUCR = new StringIndexer()
      .setInputCol("IUCR")
      .setOutputCol("indexedIUCR")
    indexerIUCR.setHandleInvalid("skip")

    val indexedIUCR = indexerIUCR.fit(set).transform(set)

    val indexerBeats = new StringIndexer()
      .setInputCol("Beat")
      .setOutputCol("indexedBeat")
    indexerBeats.setHandleInvalid("skip")

    val indexedBeats = indexerBeats.fit(indexedIUCR).transform(indexedIUCR)

//    println(indexed.show(25))

    val assembler = new VectorAssembler()
      .setInputCols(Array("year","month","day","hourOfDay","indexedIUCR"))
      .setOutputCol("features")

    val output = assembler.transform(indexedBeats)

//    println(output.show(25))

    // Split the data into train and test



    val splits = output.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](5, 100, 200, 305)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setFeaturesCol("features")
      .setLabelCol("indexedBeat")
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(50)

    // train the model
    val model = trainer.fit(train)

    model.save("hdfs://concord:30101/cs455Project/model")

    // compute accuracy on the test set
    val result = model.transform(test).show(25)




//    val predictionAndLabels = result.select("prediction")
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setMetricName("accuracy")
//
//    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")


  }

}
