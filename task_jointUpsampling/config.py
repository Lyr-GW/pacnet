#IDRiD数据集根目录
# ROOT_DIR = 'D:/迅雷下载/idrid/A. Segmentation'
# ROOT_DIR = '/home/unet/data/A. Segmentation'
ROOT_DIR = '../data/A. Segmentation'

#训练集原始图片
ORIGIN_TRAIN_IMG = '/1. Original Images/a. Training Set'
TRAINING_PATH = '/home/unet/data/A. Segmentation/1. Original Images/a. Training Set'
#训练集视GroudTruth
ORIGIN_TRAIN_OD_GT = '/2. All Segmentation Groundtruths/a. Training Set/4. Soft Exudates'
TRAINING_GT_PATH = '/home/unet/data/A. Segmentation/1. Original Images/a. Training Set/2. All Segmentation ' \
                   'Groundtruths/a. Training Set/4. Soft Exudates '


# DDR_ROOT_DIR = '/home/linwei/UNet/data/lesion_segmentation'
DDR_ROOT_DIR = 'D:/Study/GithubReps/pacnet/lesion_segmentation'

DDR_TRAIN_IMG = '/train/image'
DDR_TRAIN_VSL = '/train/vessels'
DDR_TRAIN_GT = '/train/label/SE'

DDR_VALID_IMG = '/valid/image'
DDR_VALID_VSL = '/valid/vessels'
DDR_VALID_GT = '/valid/segmentation label/SE'

DDR_TEST_IMG = '/test/image'
DDR_TEST_VSL = '/test/vessels'
DDR_TEST_GT = '/test/label/SE'