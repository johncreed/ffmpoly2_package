#! /bin/bash

# Get fold number and check cmd argument
if [ $# -ne 2 ]
    then
        echo "./grid_ffmpoly2_cv num_fold num_core.\n"
        exit 1
    else
        fold=$1
        core_num=$2
        echo "Do fold = $fold with max core ${core_num}"
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

echo "Generate bin file and pair file!!!!"
./poly2-train -s 1 -t 1 -k 1 -p ${test_path} ${train_path} 
echo "Finish generating!! Start multi core!"

grid()
{ 
  # Iteration
  t=2500
  # Log path
  log_path=poly2_logs
  for l in 0 1e-7
  do
      for r in 6.4 3.2 1.6 0.8
      do
          echo "./poly2-train -s 1 -t 2500 -k 1 -p ${test_path} -l ${l} -r ${r} ${train_path} > ${log_path}/${test_path}.$l.$r "
      done
  done
}

# Start grid with multi core
grid | xargs -d '\n' -P $core_num -I {} sh -c {} &
