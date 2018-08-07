train_path=ijcnn/ijcnn1.tr_va.ffm
test_path=ijcnn/ijcnn1.te.ffm
pair_path=ijcnn/ijcnn1.tr_va.ffm.pair

# threshold need to be scale
#v=0
# latent size
#k=1
# iteration
t=500
# regularization parameter
l=1e-7
# learning rate
r=3.2

log_path=poly2_logs/${test_path#*/}.$l.$r.final
cmd="./poly2-train -s 1 -t ${t} -k 1 -p ${test_path} -l ${l} -r ${r} ${train_path} >> $log_path"
echo $cmd
echo $cmd > $log_path
eval $cmd &
