import java.sql.Date

import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.functions.{dayofmonth, hour, month}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}

object CrimePredictor {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().appName("chicagoCrime")
      .master("spark://concord.cs.colostate.edu:46010")
      .getOrCreate()


    // Create input dataframe.
    println("******"+args(1))
    val values = Seq(
      Row(args(1), args(2), "0")
    )

    val someSchema = List(
      StructField("Date", DateType, true),
      StructField("IUCR", StringType, true),
      StructField("Beat", DoubleType, true)
    )

    val data = spark.createDataFrame(
      spark.sparkContext.parallelize(values),
      StructType(someSchema)
    )

    val additions = data
      .withColumn("hourOfDay", hour(data("Date")))
      .withColumn("day", dayofmonth(data("Date")))
      .withColumn("month", month(data("Date")))
    //      .withColumn("adjBeats", correctBeats(crime("Beat")))

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
    features.show(25)

    val labelIndexer = new StringIndexer().setInputCol("Beat").setOutputCol("indexedBeat").fit(features)
    println(s"Found labels: ${labelIndexer.labels.mkString("[", ", ", "]")}")

    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(405).fit(features)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // train the model
    val model = MultilayerPerceptronClassificationModel.load(args(0))

    val predictions = model.transform(features)
    predictions.show(75)

  }

}
