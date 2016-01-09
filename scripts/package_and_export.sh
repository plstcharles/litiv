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
REMOTE_PACKAGE_NAME="lbsp_pack_$1"
REMOTE_SCRIPT_NAME="lbsp_exec_$1.sh"
REMOTE_LOG_NAME="lbsp_exec_$1.out"

echo "Creating local package..."
mkdir -p /home/perf5/workspace/lbsp/Release/$REMOTE_PACKAGE_NAME
cp /home/perf5/workspace/lbsp/Release/lbsp /home/perf5/workspace/lbsp/Release/$REMOTE_PACKAGE_NAME/
find /usr/local/lib -name "libopencv_*" -exec cp {} /home/perf5/workspace/lbsp/Release/$REMOTE_PACKAGE_NAME/ \;
old_wd=$(pwd)
cd /home/perf5/workspace/lbsp/Release
echo "Compressing..."
tar czf /home/perf5/workspace/lbsp/Release/$REMOTE_PACKAGE_NAME.tar.gz $REMOTE_PACKAGE_NAME
cd $old_wd

echo "Sending to $REMOTE_HOST_NAME..."
scp /home/perf5/workspace/lbsp/Release/$REMOTE_PACKAGE_NAME.tar.gz pistc@$REMOTE_HOST_NAME:
if [ $? -ne 0 ]
    then
    echo "SSH connection failed."
    rm -r /home/perf5/workspace/lbsp/Release/$REMOTE_PACKAGE_NAME
    rm -r /home/perf5/workspace/lbsp/Release/$REMOTE_PACKAGE_NAME.tar.gz
    exit -1
fi

echo "Writing local extraction/execution script..."
echo "#!/bin/bash
chmod 600 $REMOTE_PACKAGE_NAME.tar.gz
tar -xzf $REMOTE_PACKAGE_NAME.tar.gz
chmod 700 $REMOTE_PACKAGE_NAME/lbsp
rm $REMOTE_PACKAGE_NAME.tar.gz
./$REMOTE_PACKAGE_NAME/lbsp"  > /home/perf5/workspace/lbsp/$REMOTE_SCRIPT_NAME

echo "Sending extraction/execution script to $REMOTE_HOST_NAME..."
scp /home/perf5/workspace/lbsp/$REMOTE_SCRIPT_NAME pistc@$REMOTE_HOST_NAME:
if [ $? -ne 0 ]
    then
    echo "SSH connection failed."
    echo "Removing local files..."
    rm /home/perf5/workspace/lbsp/$REMOTE_SCRIPT_NAME
    rm -r /home/perf5/workspace/lbsp/Release/$REMOTE_PACKAGE_NAME
    rm -r /home/perf5/workspace/lbsp/Release/$REMOTE_PACKAGE_NAME.tar.gz
    exit -1
fi

echo "Starting download/extract script on $REMOTE_HOST_NAME..."
REMOTE_CMD="chmod 755 $REMOTE_SCRIPT_NAME; nohup ./$REMOTE_SCRIPT_NAME \`</dev/null\` >~/$REMOTE_LOG_NAME 2>&1 &"
ssh pistc@$REMOTE_HOST_NAME $REMOTE_CMD

echo "Removing local files..."
rm /home/perf5/workspace/lbsp/$REMOTE_SCRIPT_NAME
rm -r /home/perf5/workspace/lbsp/Release/$REMOTE_PACKAGE_NAME
rm -r /home/perf5/workspace/lbsp/Release/$REMOTE_PACKAGE_NAME.tar.gz
