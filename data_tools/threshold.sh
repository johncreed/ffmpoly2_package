#! /bin/bash

# Get fold number and check cmd argument
if [ $# -ne 1 ]
    then
        echo "./grid_ffmpoly2_cv pair_file.\n"
        exit 1
    else
        echo "Start!!"
fi

res_file=news20.tr.ffm.res
python static.py $1 $res_file
tail -n 1 $res_file | cut -d ':' -f1 | xargs -I {} python gen_thresh.py {} > threshold_log.tr

echo "Finish. Please check the threshold_log.tr file"
