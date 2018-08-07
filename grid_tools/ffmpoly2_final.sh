train_path=ijcnn/ijcnn1.tr_va.ffm
test_path=ijcnn/ijcnn1.te.ffm
pair_path=ijcnn/ijcnn1.tr_va.ffm.pair

# threshold need to be scale
v=113397
# latent size
k=11
# iteration
t=500
# regularization parameter
l=0
# learning rate
r=0.8

log_path=ffmpoly2_logs/${test_path#*/}.$l.$r.$v.final
cmd="./ffmpoly2-train -s 1 -t ${t} -k ${k} -v ${v} -vp ${test_path} -fl ${l} -pl ${l} -r ${r} -p ${pair_path} ${train_path} >> $log_path"
echo $cmd
echo $cmd > $log_path
eval $cmd &
