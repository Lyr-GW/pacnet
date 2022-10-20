import os
import models, datasets
import config
import torch
from torchvision import transforms

from task_jointUpsampling.main import main

def test_dataset():
    #DDR loader
    #train 训练集
    ch, guide_ch = 1, 3
    # perf_measures = ('rmse',) if not args.measures else args.measures
    train_augs = transforms.Compose([
        transforms.Resize([2848, 4288]),
        transforms.RandomCrop([512,512]),
        transforms.ToTensor(),
    ])
    train_dset = datasets.DDRDataset(
        inputs_root = config.DDR_ROOT_DIR+config.DDR_TRAIN_IMG,
        transform = train_augs,
    ),
    test_dset = datasets.DDRDataset(
        inputs_root = config.DDR_ROOT_DIR+config.DDR_TEST_IMG,
        transform = train_augs,
    ),
    print('Loaded DDR datasets.')

    # data loader
    # train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=8, shuffle=True)
    train_loader=torch.utils.data.DataLoader(
        dataset=datasets.DDRDataset(
            inputs_root=config.DDR_ROOT_DIR+config.DDR_TRAIN_IMG,
            transform=train_augs,
        ),
        batch_size=8,
        shuffle=False,
        num_workers=8,
    )
    # print("len of test_dset"+str(len(test_dset)))
    # test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.test_batch_size, shuffle=True, **dl_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=8, shuffle=True)
    imgs = os.listdir(config.DDR_ROOT_DIR+config.DDR_TRAIN_IMG)
    print('len of files in dir: ' + str(len(imgs)))
    print('len of train_dset: ' + str(len(train_dset)))

    for i, inputs in enumerate(train_loader):
        print(str(i) + "--" + str(inputs))


if __name__ == '__main__':
    test_dataset()