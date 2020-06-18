pids=$(ps -aux|grep "chengli"|grep "test"|awk '{print $2}')
for pid in $pids
do
 echo $pid
  kill -9 $pid
done