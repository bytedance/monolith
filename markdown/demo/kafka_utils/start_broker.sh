#!/bin/bash
source ./kafka_base.sh
bash $KAFKA_PATH/bin/zookeeper-server-start.sh -daemon $KAFKA_PATH/config/zookeeper.properties
sleep 10
bash $KAFKA_PATH/bin/kafka-server-start.sh -daemon $KAFKA_PATH/config/server.properties
