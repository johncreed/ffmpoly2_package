for l in 0 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4
do
    for r in 0.05 0.1 0.2
    do
        ./libffm/ffm-train -r $r -l $l --auto-stop -s 12 -k 4 --no-norm -p data/avazu-app.val.ffm data/avazu-app.tr.ffm > logs/app.$l.$r
    done
done

for l in 0 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4
do
    for r in 0.05 0.1 0.2
    do
    ./libffm/ffm-train -r $r -l $l --auto-stop -s 12 -k 4 --no-norm -p data/avazu-site.val.ffm data/avazu-site.tr.ffm > logs/site.$l.$r
    done
done

for l in 0 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4
do
    for r in 0.05 0.1 0.2
    do
        ./fm/fm-train -r $r -l $l -s 12 -k 100 --no-norm -p data/avazu-app.val.ffm data/avazu-app.tr.ffm > logs/app.$l.$r.fm
    done
done

for l in 0 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4
do
    for r in 0.05 0.1 0.2
    do
    ./fm/fm-train -r $r -l $l -s 12 -k 100 --no-norm -p data/avazu-site.val.ffm data/avazu-site.tr.ffm > logs/site.$l.$r.fm
    done
done

for l in 0 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4
do
    for r in 0.05 0.1 0.2
    do
        ./poly2_w_linear/poly2_w_linear-train -r $r -l $l --auto-stop -s 12 -k 4 --no-norm -p data/avazu-app.val.ffm data/avazu-app.tr.ffm > logs/app.$l.$r.poly2
    done
done

for l in 0 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4
do
    for r in 0.05 0.1 0.2
    do
        ./poly2_w_linear/poly2_w_linear-train -r $r -l $l --auto-stop -s 12 -k 4 --no-norm -p data/avazu-site.val.ffm data/avazu-site.tr.ffm > logs/site.$l.$r.poly2
    done
done

for l in 0 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4
do
    for r in 0.05 0.1 0.2
    do
        ./afm/train -s 6 --freq --eta $r -lu $l -lv $l -lw -1 -c 12 -k 50 -p data/avazu-app.val data/avazu-app.tr > logs/app.$l.$r.afm
    done
done

for l in 0 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4
do
    for r in 0.05 0.1 0.2
    do
        ./afm/train -s 6 --freq --eta $r -lv $l -lu $l -lw -1 -c 12 -k 50 -p data/avazu-site.val data/avazu-site.tr > logs/site.$l.$r.afm
    done
done

