import argparse
import os
import os.path as osp
import tempfile
import zipfile

import mmcv

LINEMOD_LEN = 1236
TRAINING_LEN = 999999


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert LineMod dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='path of LineMod')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'linemod', 'fuse_driller')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'images'))
    # mmcv.mkdir_or_exist(osp.join(out_dir, 'images', 'training'))
    # mmcv.mkdir_or_exist(osp.join(out_dir, 'images', 'validation'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations'))
    # mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations', 'training'))
    # mmcv.mkdir_or_exist(osp.join(out_dir, 'annotations', 'validation'))

    #delete only for fuse#
    # args.tmp_dir = osp.join(dataset_path, 'rgb')

    # # with args.tmp_dir as tmp_dir:
    #     # print('Extracting LineMod.zip...')
    #     # zip_file = zipfile.ZipFile(dataset_path)
    #     # zip_file.extractall(tmp_dir)

    # print('Generating training dataset...')

    # # print(args.tmp_dir, len(os.listdir(tmp_dir)))
    # # assert len(os.listdir(args.tmp_dir)) == LINEMOD_LEN, \
    # #     'len(os.listdir(tmp_dir)) != {}'.format(LINEMOD_LEN)

    # for img_name in sorted(os.listdir(args.tmp_dir))[:TRAINING_LEN]:
    #     img = mmcv.imread(osp.join(args.tmp_dir, img_name))
    #     if osp.splitext(img_name)[1] == '.png':
    #         mmcv.imwrite(
    #             img,
    #             osp.join(out_dir, 'images',
    #                      osp.splitext(img_name)[0] + '.png'))
    #     # else:
    #     #     # The annotation img should be divided by 128, because some of
    #     #     # the annotation imgs are not standard. We should set a
    #     #     # threshold to convert the nonstandard annotation imgs. The
    #     #     # value divided by 128 is equivalent to '1 if value >= 128
    #     #     # else 0'
    #     #     mmcv.imwrite(
    #     #         img[:, :, 0] // 128,
    #     #         osp.join(out_dir, 'annotations', 'training',
    #     #                  osp.splitext(img_name)[0] + '.png'))

    # for img_name in sorted(os.listdir(args.tmp_dir))[TRAINING_LEN:]:
    #     img = mmcv.imread(osp.join(args.tmp_dir, img_name))
    #     if osp.splitext(img_name)[1] == '.png':
    #         mmcv.imwrite(
    #             img,
    #             osp.join(out_dir, 'images',
    #                      osp.splitext(img_name)[0] + '.png'))
    #     # else:
    #     #     mmcv.imwrite(
    #     #         img[:, :, 0] // 128,
    #     #         osp.join(out_dir, 'annotations', 'validation',
    #     #                  osp.splitext(img_name)[0] + '.png'))

    #delete only for fuse#
    # args.tmp_dir = osp.join(dataset_path, 'mask')
    args.tmp_dir = dataset_path
    
    # with args.tmp_dir as tmp_dir:
        # print('Extracting LineMod.zip...')
        # zip_file = zipfile.ZipFile(dataset_path)
        # zip_file.extractall(tmp_dir)

    print('Generating training dataset...')

    # assert len(os.listdir(args.tmp_dir)) == LINEMOD_LEN, \
    #     'len(os.listdir(tmp_dir)) != {}'.format(LINEMOD_LEN)

    for img_name in sorted(os.listdir(args.tmp_dir))[:TRAINING_LEN]:
        # img = mmcv.imread(osp.join(args.tmp_dir, img_name))
        if osp.splitext(img_name)[1] == '.jpg':
            # if img_name > '9845_rgb.jpg':
            #     img = mmcv.imread(osp.join(args.tmp_dir, img_name))
            #     mmcv.imwrite(
            #         img,
            #         osp.join(out_dir, 'images',
            #                 osp.splitext(img_name)[0][:-4] + '.png'))
            pass
        elif "_mask" in img_name:
            # pass
            img = mmcv.imread(osp.join(args.tmp_dir, img_name))
            # The annotation img should be divided by 128, because some of
            # the annotation imgs are not standard. We should set a
            # threshold to convert the nonstandard annotation imgs. The
            # value divided by 128 is equivalent to '1 if value >= 128
            # else 0'
            img_ = img[:, :, 0].copy()
            for i in range(480):
                for j in range(640):
                    #1 for ape, 8 for driller
                    if(img[i, j, 0] == 8):
                        img_[i, j] = 1
                    else:
                        img_[i, j] = 0
            mmcv.imwrite(
                # img[:, :, 0] // 128,
                img_,
                osp.join(out_dir, 'annotations',
                         osp.splitext(img_name)[0][:-5] + '.png'))

    for img_name in sorted(os.listdir(args.tmp_dir))[TRAINING_LEN:]:
        # img = mmcv.imread(osp.join(args.tmp_dir, img_name))
        if osp.splitext(img_name)[1] == '.jpg':
            # mmcv.imwrite(
            #     img,
            #     osp.join(out_dir, 'images',
            #              osp.splitext(img_name)[0][:-4] + '.png'))
            pass
        elif "_mask" in img_name:
            img = mmcv.imread(osp.join(args.tmp_dir, img_name))
            img_ = img[:, :, 0].copy()
            for i in range(480):
                for j in range(640):
                    if(img[i, j, 0] == 10):
                        img_[i, j] = 1
                    else:
                        img_[i, j] = 0
            mmcv.imwrite(
                # img[:, :, 0] // 128,
                img_,
                osp.join(out_dir, 'annotations',
                         osp.splitext(img_name)[0][:-5] + '.png'))

        # print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()