class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # global settings
        parser.add_argument('--train-dir', type=str, default ='./datasets/train',  help='dir of train data')
        parser.add_argument('--val-dir', type=str, default ='./datasets/val',  help='dir of validation data')
        parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', help='contrastive model architecture backbone.')
        parser.add_argument('--dataset', type=str, default ='CholecSeg8k')
        parser.add_argument('--save-dir',  type=str, default='./workdir/', help='save dir.')
        parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)')
        parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256), this is the total '
                                                                                    'batch size of all GPUs on the current node when '
                                                                                    'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
        parser.add_argument('--contrastive-weights', default='', type=str, metavar='PATH', help='path to contrastive pre-training weights (default: none)')
        parser.add_argument('--generative-weights', default='', type=str, metavar='PATH', help='path to generative pre-training weights (default: none)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
        parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training.')
        parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
        parser.add_argument('--num_classes', type=int, default=2, help='number of classes for segmentation')

        return parser
