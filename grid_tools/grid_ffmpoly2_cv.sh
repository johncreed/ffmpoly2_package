# /bin/bash

# Get fold number and check cmd argument
if [ $# -ne 2 ]
    then
        echo "./grid_ffmpoly2_cv num_fold num_core.\n"
        exit 1
    else
        fold=$1
        core_num=$2
        echo "Do fold = $fold with max_core ${core_num}"
fi

# Data settings
train_path=mnist/mnist_ove.scale.tr_va.ffm.$fold.tr
test_path=mnist/mnist_ove.scale.tr_va.ffm.$fold.va
pair_path=mnist/mnist_ove.scale.tr_va.ffm.$fold.tr.pair
#for v in 0 2 8 23 68 198 571 1647 4747 13678 39410 
#1.6 k = 64 t = 200

train_path=ijcnn/ijcnn1.tr_va.ffm.$fold.tr
test_path=ijcnn/ijcnn1.tr_va.ffm.$fold.va
pair_path=ijcnn/ijcnn1.tr_va.ffm.$fold.tr.pair
#for v in 0 3 10 32 105 336 1078 3453 11058 35411 113397 
#1.6 k=11 t=200 l=1e-5 

train_path=news20/news20.tr.ffm.$fold.tr
test_path=news20/news20.tr.ffm.$fold.va
pair_path=new20/news20.tr.ffm.pair
#for v in ??
#3.2 k=64 t=2000 l=0 1e-7 1e-5

train_path=webspam/webspam_wc_normalized_unigram.tr_va.ffm.$fold.tr
test_path=webspam/webspam_wc_normalized_unigram.tr_va.ffm.$fold.va
pair_path=webspam/webspam_wc_normalized_unigram.tr_va.ffm.$fold.tr.pair
#for v in 0 3 12 43 151 529 1856 6506 22809 79962 280319 
#6.4 k = 64 t = 2500

train_path=real-sim/real-sim.ffm.tr.${fold}.tr
test_path=real-sim/real-sim.ffm.tr.${fold}.va
pair_path=real-sim/real-sim.ffm.tr.pair
#for v in 0 6 43 291 1932 12817 
#0.8 k=64 t=2500

train_path=frappe/frappe.ffm.tr_va.${fold}.tr
test_path=frappe/frappe.ffm.tr_va.${fold}.va
pair_path=frappe/frappe.ffm.tr_va.pair
#for v in 0 11 124 1386 15455 172324 
#0.025 k=512 t=2500

train_path=german-numer/german-numer_scale.tr_va.ffm.${fold}.tr
test_path=german-numer/german-numer_scale.tr_va.ffm.${fold}.va
pair_path=german-numer/german-numer_scale.tr_va.ffm.pair
#for v in 0 3 14 55 210 800 
#?? k=12 t=2500

echo "Generate and check bin file and pair file!!!!"
./ffmpoly2-train -s 1 -t 1 -k 1 -vp ${test_path} -p ${pair_path} ${train_path} 
echo "Finish generating and checking!! Start multi core!"
#echo "Pleas generate pair file first!!"

# Generate Parameter Set
grid()
{
  # Iteration
  t=5000
  # Latent vector
  k=6
  # Log path
  log_path=ffmpoly2_logs
  
  for k in 6
  do 
    for v in 0 160 320 480 540 800 
    do
        for r in 0.05 0.0125 0.003125
        do
            for l in 0 1e-5 1e-7 1e-6
            do
             echo "./ffmpoly2-train -s 1 -t ${t} -k ${k} -v ${v} -vp ${test_path} -fl ${l} -pl ${l} -r ${r} -p ${pair_path} ${train_path} > ${log_path}/${test_path}.$k.$l.$r.$v"
            done
        done
    done
  done
}

# Start grid with multi core
grid | xargs -d '\n' -P $core_num -I {} sh -c {} &
