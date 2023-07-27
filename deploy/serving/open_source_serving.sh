#!/bin/bash
set -eux

export SHARD_ID=`expr $MLP_SHARD_ID - 1`
echo "The SHARD_ID {$SHARD_ID}"

export MY_POD_NAME=$MLP_POD_NAME
echo "The MY_POD_NAME {$MY_POD_NAME}"

export byterec_host_shard_n=$MLP_SHARD_NUM
echo "The byterec_host_shard_n {$byterec_host_shard_n}"

if [ $MLP_IDC ]; then
    export TCE_INTERNAL_IDC=$MLP_IDC
else
    export TCE_INTERNAL_IDC="cn-beijing-b"
fi

echo "The TCE_INTERNAL_IDC {$TCE_INTERNAL_IDC}"

#export TCE_CLUSTER=$
export TCE_CLUSTER=default
echo "The TCE_CLUSTER {$TCE_CLUSTER}"

# TCE_PSM for metrics
PSM_PREFIX="data.tob.monolith_serving_"

SHELL_FOLDER=/opt/tiger/monolith_serving
export PATH=$SHELL_FOLDER:$PATH

if [ $MLP_ROLE_NAME = 'PS' ]; then
    export SERVER_TYPE='ps'
    export ENABLE_BATCHING=false
    export TCE_PSM=$PSM_PREFIX"ps-"$MLP_SERVICE_NAME
elif [ $MLP_ROLE_NAME == 'Entry' ]; then
    export SERVER_TYPE='entry'  
    export ENABLE_BATCHING=false
    export TCE_PSM=$PSM_PREFIX"en-"$MLP_SERVICE_NAME
elif [ $MLP_ROLE_NAME == 'DenseNN' ]; then
    export SERVER_TYPE='dense'
    export CUDA_MPS_PIPE_DIRECTORY=/dev/shm
    export ENABLE_BATCHING=true
    export TCE_PSM=$PSM_PREFIX"de-"$MLP_SERVICE_NAME
    nvidia-cuda-mps-control -d
fi
echo "THE SERVER_TYPE {$SERVER_TYPE}"
echo "THE ENABLE_BATCHING {$ENABLE_BATCHING}"
echo "The TCE_PSM {$TCE_PSM}"

cd $SHELL_FOLDER
echo "The shell folder is {$SHELL_FOLDER}"
PYV=$(python -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")

echo "Using sparse_dense_serving: {$DENSE_ALONE}"

cat agent.conf | sed -e "s/{{bzid}}/${BZID}/g" -e "s/{{base_name}}/${BASE_NAME}/g" -e "s?{{base_path}}?${BASE_PATH}?g" -e "s/{{num_ps}}/${NUM_PS}/g" -e "s/{{server_type}}/${SERVER_TYPE}/g" \
	-e "s/{{zk_servers}}/${ZK_SERVERS}/g" -e "s/{{dense_alone}}/${DENSE_ALONE}/g" -e "s/{{enable_batching}}/${ENABLE_BATCHING}/g" > render_agent.conf

# add other conf parameter here
#echo -e "\ndense_service_num 3" >> render_agent.conf

cd $SHELL_FOLDER/bin

if [ $PYV = '3.8' ]; then
  python run --bin_name="agent" --conf /opt/tiger/monolith_serving/render_agent.conf
else
  python3 run --bin_name="agent" --conf /opt/tiger/monolith_serving/render_agent.conf
fi
