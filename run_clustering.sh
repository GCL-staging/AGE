for i in $(seq 10)
do
  python train.py --dataset cora --gnnlayers 8 --upth_st 0.011 --lowth_st 0.001 --upth_ed 0.1 --lowth_ed 0.5 --gpu 5
done

for i in $(seq 10)
do
  python train.py --dataset citeseer --gnnlayers 3 --upth_st 0.0015 --lowth_st 0.001 --upth_ed 0.1 --lowth_ed 0.5 --gpu 5
done

for i in $(seq 10)
do
  python train.py --dataset pubmed --gnnlayers 35 --upth_st 0.0013 --lowth_st 0.001 --upth_ed 0.7 --lowth_ed 0.8 --gpu 5
done
