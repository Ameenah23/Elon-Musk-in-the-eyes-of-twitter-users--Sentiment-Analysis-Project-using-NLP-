/*
    Ameenah Al-Haidari
                                               SQL Codes from Athena
 */
 #################################################################################################
# Raw Data

CREATE EXTERNAL TABLE IF NOT EXISTS `b18_ElonMusk`.`ElonMusk_demo` (
  `tweet_id` double,
  `name` string,
  `username` string,
  `tweet` string,
  `public_metrics` float,
  `location` string,
  `author_id` string,
  `created_at` string
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
WITH SERDEPROPERTIES (
  'field.delim' = '\t',
  'collection.delim' = '\u0002',
  'mapkey.delim' = '\u0003'
)
STORED AS INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat' OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION 's3://b18-ameenah/ElonMusk_BigDataProject/tweetRaw_ElonMusk.csv/';

#########################################
# Create Queries and table from the raw data

select * from elonmusk_demo limit 10;

select tweet from elonmusk_demo limit 10;



SELECT names, count(*) as cont FROM elonmusk_demo
CROSS JOIN UNNEST(split(tweet, ' ')) as t(names)
group by names;


CREATE TABLE word_count AS
SELECT names, count(*) as cont FROM elonmusk_demo
CROSS JOIN UNNEST(split(tweet, ' ')) as t(names)
group by names;

###############################################################################################

# Prediction DATA

CREATE EXTERNAL TABLE IF NOT EXISTS `b18_Pred_ElonMusk`.`Prediction_ElonMusk` (
  `tweet` string,
  `Sentiment` int,
  `label` int,
  `prediction` int
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
WITH SERDEPROPERTIES (
  'field.delim' = '\t',
  'collection.delim' = '\u0002',
  'mapkey.delim' = '\u0003'
)
STORED AS INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat' OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION 's3://b18-ameenah/EMusk_FinalPred.csv/';


##################################################################################################
# Create Queries and table from the Predictions data
select * from prediction_elonmusk limit 10;


SELECT Prednames, count(*) as pred_cont FROM prediction_elonmusk
CROSS JOIN UNNEST(split(tweet, ' ')) as t(Prednames)
group by Prednames;



CREATE TABLE Prediction_word_count AS
SELECT Prednames, count(*) as pred_cont FROM prediction_elonmusk
CROSS JOIN UNNEST(split(tweet, ' ')) as t(Prednames)
group by Prednames;

##################################################################################
