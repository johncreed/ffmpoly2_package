
if [ $# -ne 2 ]
then
  echo './smooth_log.sh file smooth_num'
  exit 1
else
  echo 'Start'
  lfile=$1
  snum=$2
fi

#Parse
logtmp=$1.tmp
grep 'iter' ${lfile} > ${logtmp}
lnum="$(wc -l ${logtmp} | cut -d ' ' -f1)"

#Start convert
ofile=$1.smooth
rm -i ${ofile}
for i in $(seq $((${lnum}-${snum}+1)))
do
  if [ $(($i % 100)) -eq 0 ]; then echo '.' ; fi
  avgnum="$(head -n $(($i+${snum}-1)) ${logtmp} | tail -n ${snum} | awk '{ total += $3 } END { print total/ NR }')"
  echo "iter ${i} ${avgnum}" >> ${ofile}
done
rm -if ${logtmp}
