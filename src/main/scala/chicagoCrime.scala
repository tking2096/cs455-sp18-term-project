import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.{hour, dayofmonth, month}

object chicagoCrime {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().appName("chicagoCrime")
      .master("spark://concord.cs.colostate.edu:46010")
      .getOrCreate()

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

    // Read in CSV file
    val crime = spark.read.format("csv")
      .option("header", "true")
      .option("timestampFormat","MM/dd/yyyy hh:mm:ss a")
      .schema(customSchema)
      .load(args(0))
      .persist()


    // Test extraction of hour
    val additions = crime
      .withColumn("hourOfDay", hour(crime("Date")))
      .withColumn("day", dayofmonth(crime("Date")))
      .withColumn("month", month(crime("Date")))

    val set = additions.select("year", "month", "day", "hourOfDay", "IUCR", "Beat")

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
