//Reference : https://www.youtube.com/watch?v=YX1xxWVxM0w

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, IDF, Tokenizer,StringIndexer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.NaiveBayes

val path = "articles/*"
val testPath = "testarticles/*"

val articlesRawData = sc.wholeTextFiles(path)
val testarticlesRawData = sc.wholeTextFiles(testPath)


//Get file path
val filepath = articlesRawData.map{case(filepath,text) => (filepath)}
val testfilepath = testarticlesRawData.map{case(filepath,text) => (filepath)}

//Get text
val text = articlesRawData.map{case(filepath,text) => text}
val testtext = testarticlesRawData.map{case(filepath,text) => text}

//Using fileName as id
val id = filepath.map(filepath => (filepath.split("/").takeRight(1))(0))
val testid = testfilepath.map(testfilepath => (testfilepath.split("/").takeRight(1))(0))

//Extract topic from filepath
val topic = filepath.map (filepath => (filepath.split("/").takeRight(2))(0))
val testtopic = testfilepath.map (testfilepath => (testfilepath.split("/").takeRight(2))(0))

//Defining a class and converting into data frame
case class articlesCaseClass(id: String, text: String, topic: String)

val articles = articlesRawData.map{case (filepath, text) =>
    val id = filepath.split("/").takeRight(1)(0)
    val topic = filepath.split("/").takeRight(2)(0)
    articlesCaseClass(id, text, topic)}.toDF()

val testarticles = testarticlesRawData.map{case (testfilepath, testtext) =>
    val testid = testfilepath.split("/").takeRight(1)(0)
    val testtopic = testfilepath.split("/").takeRight(2)(0)
    articlesCaseClass(testid, testtext, testtopic)}.toDF()

articles.groupBy("topic").count().show()

val Array(training, test) = articles.randomSplit(Array(0.9, 0.1), seed = 12345)

val indexer = new StringIndexer().setInputCol("topic").setOutputCol("label")
val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(false)
val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
val nb = new NaiveBayes()

val pipeline = new Pipeline().setStages(Array(indexer,tokenizer, remover, hashingTF, idf, nb))

val model = pipeline.fit(articles)

val predictions = model.transform(testarticles)

predictions.show(5)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

