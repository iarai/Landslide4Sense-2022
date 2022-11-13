import argparse
import os
import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from dataset.landslide_dataset import LandslideDataSet
from model.Networks import unet
import h5py

name_classes = ['Non-Landslide', 'Landslide']
epsilon = 1e-14


def importName(modulename, name):
    """ Import a named object from a module in the context of this function.
    """
    try:
        module = __import__(modulename, globals(), locals(), [name])
    except ImportError:
        return None
    return vars(module)[name]


def get_arguments():
    parser = argparse.ArgumentParser(description="Baseline method for Land4Seen")

    parser.add_argument("--data_dir", type=str, default='/scratch/Land4Sense_Competition_h5/',
                        help="dataset path.")
    parser.add_argument("--model_module", type=str, default='model.Networks',
                        help='model module to import')
    parser.add_argument("--model_name", type=str, default='unet',
                        help='modle name in given module')
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt',
                        help="test list file.")
    parser.add_argument("--input_size", type=str, default='128,128',
                        help="width and height of input images.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="gpu id in the training.")
    parser.add_argument("--snapshot_dir", type=str, default='./test_map/',
                        help="where to save predicted maps.")
    parser.add_argument("--restore_from", type=str, default='./exp/batch3500_F1_7396.pth',
                        help="trained model.")

    return parser.parse_args()


def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    snapshot_dir = args.snapshot_dir
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    cudnn.enabled = True
    cudnn.benchmark = True

    # Create network   
    model = unet(n_classes=args.num_classes)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model = model.cuda()

    test_loader = data.DataLoader(
        LandslideDataSet(args.data_dir, args.test_list, set='unlabeled'),
        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    print('Testing..........')
    model.eval()

    for index, batch in enumerate(test_loader):
        image, _, name = batch
        image = image.float().cuda()
        name = name[0].split('.')[0].split('/')[-1].replace('image', 'mask')
        print(index + 1, '/', len(test_loader), ': Testing ', name)

        with torch.no_grad():
            pred = model(image)

        _, pred = torch.max(interp(nn.functional.softmax(pred, dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy().astype('uint8')
        with h5py.File(snapshot_dir + name + '.h5', 'w') as hf:
            hf.create_dataset('mask', data=pred)


if __name__ == '__main__':
    main()
