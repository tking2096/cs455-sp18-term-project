import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.{dayofmonth, hour, map, month, udf}

object chicagoCrime {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().appName("chicagoCrime")
      .master("spark://concord.cs.colostate.edu:46010")
      .getOrCreate()

    val beatSchema = StructType(Array(
      StructField("actualBeat", IntegerType, true),
      StructField("newBeat", IntegerType, true)
    ))

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
      StructField("Beat", IntegerType, true),
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

    val beatData = spark.read.format("csv")
      .option("header", "false")
      .schema(beatSchema)
      .load(args(1))
      .persist()


    // Read in CSV file
    val crime = spark.read.format("csv")
      .option("header", "true")
      .option("timestampFormat","MM/dd/yyyy hh:mm:ss a")
      .schema(customSchema)
      .load(args(0))
      .persist()


    val beatRdd = beatData.rdd.map{
      case Row(actualBeat: Int, newBeat: Int) => (actualBeat, newBeat)
    }.collectAsMap()


    def func: (Int => Int) = {b => beatRdd.get(b).get}

    val correctBeats = udf(func)

//    val withCorrBeat = crime.withColumn("adjBeats", correctBeats(crime("Beat")))

//    println(withCorrBeat.show(25))


    // Test extraction of hour
    val additions = crime
      .withColumn("hourOfDay", hour(crime("Date")))
      .withColumn("day", dayofmonth(crime("Date")))
      .withColumn("month", month(crime("Date")))
      .withColumn("adjBeats", correctBeats(crime("Beat")))

    val set = additions.select("year", "month", "day", "hourOfDay", "IUCR", "Beat", "adjBeats")

    println(set.show(25))


//    val compactCrimes = crime.select("Date", "Latitude", "Longitude")
//
//    compactCrimes.coalesce(1)
//      .write.format("csv")
//      .option("header", "true")
//      .save("hdfs://concord:30101/cs455Project/compacted/01_04_CrimesDataSet")
//


  }

}
