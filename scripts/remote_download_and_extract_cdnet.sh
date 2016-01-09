#!/bin/bash

if [ $# -eq 0 ]
    then
    echo "No arguments supplied."
    exit -1;
elif [ -z "$1" ]
    then
    echo "Bad first arg (remote host name)."
    exit -1;
fi

REMOTE_HOST_NAME="$1.info.polymtl.ca"
REMOTE_SCRIPT_NAME="download_and_extract_cdnet_$1.sh"
REMOTE_LOG_NAME="download_and_extract_cdnet_$1.out"

echo "Writing local download/extract script..."
echo "#!/bin/bash
wget -P /tmp/ http://wordpress-jodoin.dmi.usherb.ca/static/dataset/dataset.7z
chmod 644 /tmp/dataset.7z
7z x /tmp/dataset.7z -o/tmp/
mkdir -p /tmp/datasets/CDNet/
mv /tmp/dataset /tmp/datasets/CDNet/" > /home/perf5/workspace/lbsp/$REMOTE_SCRIPT_NAME

echo "Sending download/extract script to $REMOTE_HOST_NAME..."
scp /home/perf5/workspace/lbsp/$REMOTE_SCRIPT_NAME pistc@$REMOTE_HOST_NAME:
if [ $? -ne 0 ]
    then
    echo "SSH connection failed."
    echo "Removing local files..."
    rm /home/perf5/workspace/lbsp/$REMOTE_SCRIPT_NAME
    exit -1
fi

echo "Starting download/extract script on $REMOTE_HOST_NAME..."
REMOTE_CMD="chmod 755 $REMOTE_SCRIPT_NAME; nohup ./$REMOTE_SCRIPT_NAME \`</dev/null\` >~/$REMOTE_LOG_NAME 2>&1 &"
ssh pistc@$REMOTE_HOST_NAME $REMOTE_CMD

echo "Removing local files..."
rm /home/perf5/workspace/lbsp/$REMOTE_SCRIPT_NAME

