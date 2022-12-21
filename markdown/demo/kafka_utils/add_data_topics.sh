#!/bin/bash
source ./kafka_base.sh
$KAFKA_PATH/bin/kafka-topics.sh --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 6 --topic movie-train
$KAFKA_PATH/bin/kafka-topics.sh --describe --bootstrap-server 127.0.0.1:9092 --topic movie-train
