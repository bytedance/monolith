#!/bin/bash
source ./kafka_base.sh
$KAFKA_PATH/bin/kafka-topics.sh --delete --bootstrap-server 127.0.0.1:9092 --topic movie-train