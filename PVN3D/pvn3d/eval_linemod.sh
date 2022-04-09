cls='ape'
#tst_mdl=train_log/linemod/checkpoints/${cls}/ape_pvn3d_best_27.5123.pth.tar
#tst_mdl=train_log/linemod/checkpoints/${cls}/ape_pvn3d_best_0.5243.pth.tar
tst_mdl=train_log/linemod/checkpoints/${cls}/xh.pth.tar
#tst_mdl=train_log/linemod/checkpoints/${cls}/${cls}_pvn3d.pth.tar
#tst_mdl=train_log/linemod/checkpoints/${cls}/${cls}_pvn3d_best.pth.tar
python3 -m train.train_linemod_pvn3d -checkpoint $tst_mdl -eval_net --test --cls $cls


# Eval all object

# cls_lst=('ape' 'benchvise' 'cam' 'can' 'cat' 'driller' 'duck' 'eggbox' 'glue' 'holepuncher' 'iron' 'lamp' 'phone')
# for((i=0;i<13;i++));
# do
# cls=${cls_lst[i]}
# tst_mdl=train_log/linemod/checkpoints/${cls}/${cls}_pvn3d_best.pth.tar
# python3 -m train.train_linemod_pvn3d -checkpoint $tst_mdl -eval_net --test --cls $cls
# done

