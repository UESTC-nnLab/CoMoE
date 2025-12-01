import torch
from nets.slowfastnet import slowfastnet
from thop import profile, clever_format
from utils.dataloader_for_DAUB import seqDataset, dataset_collate
from torch.utils.data import DataLoader
import time

net = slowfastnet(num_classes=1, num_frame=5)
a = torch.randn(1, 3, 5, 512, 512)
flops, params = profile(net, inputs=(a,))
macs, params = clever_format([flops, params], "%.3f")
print('Total GFLOPS: %s' % macs)
print('Total params: %s' % params)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = slowfastnet(num_classes=1, num_frame=5).to(device)

val_annotation_path = '/home/public/ITSDT/coco_val_ITSDT.txt'
val_dataset = seqDataset(val_annotation_path, 512, 5, 'val')
gen_val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4,
                     pin_memory=True, drop_last=True, collate_fn=dataset_collate)

num_warmup = 5
log_interval = 50
max_iter = 1000

total_time = 0.0
partial_time = 0.0
total_count = 0
device = 'cuda'

print(f"Start benchmark on {device}, total images: {max_iter}")

with torch.no_grad():
    for i, data in enumerate(gen_val):
        if i >= max_iter:
            break

        images = data[0].to(device)

        if i < num_warmup:
            net(images)
            continue

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        net(images)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        total_time += elapsed
        partial_time += elapsed
        total_count += 1

        if (i + 1) % log_interval == 0:
            local_fps = log_interval / partial_time
            print(f"[{i + 1:<3}/ {max_iter}] Local FPS: {local_fps:.3f}")
            partial_time = 0.0

average_fps = total_count / total_time
print(f'Final Average FPS: {average_fps:.3f}')
