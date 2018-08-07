train_path=ijcnn/ijcnn1.tr_va.ffm
test_path=ijcnn/ijcnn1.te.ffm
pair_path=ijcnn/ijcnn1.tr_va.ffm.pair

# threshold need to be scale
#v=0
# latent size
k=11
# iteration
t=500
# regularization parameter
l=0
# learning rate
r=0.8

log_path="ffm_logs/${test_path#*/}.$l.$r.final"
cmd="./ffm-train -s 1 -t ${t} -k ${k} -p ${test_path} -l ${l} -r ${r} ${train_path} >> $log_path"
echo $cmd
echo $cmd > $log_path
eval $cmd &
