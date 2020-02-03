python train.py \
    --dataset /run/media/juju/backup_loja/set_004_train.tfrecord \
    --val_dataset /run/media/juju/backup_loja/set_004_val.tfrecord \
    --classes ./data/set_00.names \
    --num_classes 1 \
    --mode fit --transfer darknet \
    --batch_size 16 \
    --epochs 80 \
    --weights /run/media/juju/backup_loja/checkpoints/regular/train/yolov3.tf \
    --checkpoints /run/media/juju/backup_loja/checkpoints/regular/train \
    --weights_num_classes 80

python train.py \
    --dataset /run/media/juju/backup_loja/set_004_train.tfrecord \
    --val_dataset /run/media/juju/backup_loja/set_004_val.tfrecord \
    --classes ./data/set_00.names \
    --num_classes 1 \
    --mode fit --transfer darknet \
    --batch_size 16 \
    --epochs 80 \
    --weights /run/media/juju/backup_loja/checkpoints/tiny/train/yolov3-tiny.tf \
    --checkpoints /run/media/juju/backup_loja/checkpoints/tiny/train \
    --weights_num_classes 80 \
    --tiny