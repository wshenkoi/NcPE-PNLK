"""
    Example for testing PointNet-LK.

    No-noise version.
"""

import argparse
import os
import sys
import logging
import numpy
import torch
import torch.utils.data
import torchvision
from scipy.spatial.transform import Rotation

# addpath('../')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # required.
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='FILENAME', help='output filename (.csv)')
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', required=True, type=str,
                        metavar='PATH', help='path to the categories to be tested') # eg. './sampledata/modelnet40_half1.txt'
    parser.add_argument('-p', '--perturbations', default='./perturbations.csv', type=str,
                        metavar='PATH', help='path to the perturbation file') # see. generate_perturbations.py

    # settings for input data
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--format', default='wv', choices=['wv', 'wt'],
                        help='perturbation format (default: wv (twist)) (wt: rotation and translation)') # the output is always in twist format
    parser.add_argument('--crop', default=1.0, type=float,
                        metavar='T', help='')

    # settings for PointNet-LK
    parser.add_argument('--max-iter', default=20, type=int,
                        metavar='N', help='max-iter on LK. (default: 20)')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to trained model file (default: null (no-use))')
    parser.add_argument('--transfer-from', default='', type=str,
                        metavar='PATH', help='path to classifier feature (default: null (no-use))')
    parser.add_argument('--dim-k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')
    parser.add_argument('--delta', default=1.0e-2, type=float,
                        metavar='D', help='step size for approx. Jacobian (default: 1.0e-2)')

    # settings for on testing
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile (default: null (no logging))')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--device', default='cpu', type=str,
                        metavar='DEVICE', help='use CUDA if available (default: cpu)')

    args = parser.parse_args(argv)
    return args
    
def dcm2euler(mats, seq, degrees=True):
    mats = mats.cpu().numpy()
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=degrees))
    return torch.tensor(numpy.stack(eulers))

def compute_metrics(est_g, igt_g):
    metrics = {}
    
    gt_g = ptlk.se3.inverse(igt_g)

    # Euler angles, Individual translation errors (Deep Closest Point convention)
    r_gt_euler_deg = dcm2euler(gt_g[:, :3, :3], seq="xyz")
    r_pred_euler_deg = dcm2euler(est_g[:, :3, :3], seq="xyz")
    
    t_gt = gt_g[:, :3, 3]
    t_pred = est_g[:, :3, 3]

    r_mse = torch.mean((r_gt_euler_deg - r_pred_euler_deg)**2, dim=1)
    r_mae = torch.mean(torch.abs(r_gt_euler_deg - r_pred_euler_deg), dim=1)
    t_mse = torch.mean((t_gt - t_pred)**2, dim=1)
    t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

    r_rmse = torch.sqrt(torch.mean(r_mse))
    t_rmse = torch.sqrt(torch.mean(t_mse))
    r_mae = torch.mean(r_mae)
    t_mae = torch.mean(t_mae)

    # Rotation, translation errors (isotropic, i.e. doesn"t depend on error
    # direction, which is more representative of the actual error)
    concatenated = ptlk.se3.concatenate(igt_g, est_g)
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), -1.0, 1.0)) * 180.0 / numpy.pi
    residual_transmag = torch.norm(concatenated[:, :, 3], dim=-1)
    err_r = torch.mean(residual_rotdeg)
    err_t = torch.mean(residual_transmag)

    metrics = [r_rmse, r_mae, t_rmse, t_mae, err_r, err_t]

    return metrics

def main(args):
    # dataset
    testset = get_datasets(args)

    # testing
    act = Action(args)
    run(args, testset, act)


def run(args, testset, action):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

#    LOGGER.debug('Testing (PID=%d), %s', os.getpid(), args)

    model = action.create_model()
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)

    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1, shuffle=False, num_workers=args.workers)

    # testing
    LOGGER.logger.debug('tests, begin')
    action.eval_1(model, testloader, args.device)
    LOGGER.logger.debug('tests, end')


class Action:
    def __init__(self, args):
        self.filename = '/content/drive/MyDrive/ori_PointNetLK-master/logf/test/' + args.outfile
        # PointNet
        self.transfer_from = args.transfer_from
        self.dim_k = args.dim_k
        self.sym_fn = None
        if args.symfn == 'max':
            self.sym_fn = ptlk.pointnet.symfn_max
        elif args.symfn == 'avg':
            self.sym_fn = ptlk.pointnet.symfn_avg
        # LK
        self.delta = args.delta
        self.max_iter = args.max_iter
        self.xtol = 1.0e-7
        self.p0_zero_mean = True
        self.p1_zero_mean = True

    def create_model(self):
        ptnet = self.create_pointnet_features()
        return self.create_from_pointnet_features(ptnet)

    def create_pointnet_features(self):
        ptnet = ptlk.pointnet.PointNet_features(self.dim_k, use_tnet=False, sym_fn=self.sym_fn)
        if self.transfer_from and os.path.isfile(self.transfer_from):
            ptnet.load_state_dict(torch.load(self.transfer_from, map_location='cpu'))
        return ptnet

    def create_from_pointnet_features(self, ptnet):
        return ptlk.pointlk.PointLK(ptnet, self.delta)
        
    def eval_1__header(self, fout):
        cols = ['overlap', 'g_rmse', 'r_rmse', 'r_mae', 't_rmse', 't_mae', 'err_r', 'err_t']
        print(','.join(map(str, cols)), file=fout)
        fout.flush()

    def eval_1(self, model, testloader, device):
        model.eval()
        
        all_dms = []
        all_est_g = []
        all_igt_g = []
        count = 0
        overlaps = 0.
        
        with open(self.filename, 'a') as fout:
#            self.eval_1__header(fout)
            with torch.no_grad():
                for i, data in enumerate(testloader):
#                    p0, p1, igt = data
                    p0, p1, igt, overlap = data
                    res = self.do_estimate(p0, p1, model, device) # --> [1, 4, 4]
                    ig_gt = igt.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]
                    g_hat = res.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]

                    dg = g_hat.bmm(ig_gt) # if correct, dg == identity matrix.
                    dx = ptlk.se3.log(dg) # --> [1, 6] (if corerct, dx == zero vector)
                    dn = dx.norm(p=2, dim=1) # --> [1]
                    dm = dn.mean()
                    all_dms.append(dm)
                    all_est_g.append(g_hat)
                    all_igt_g.append(ig_gt)
                    
                    overlaps += overlap
                    count += 1
                    
                    LOGGER.logger.info('test, %d/%d, %f', i, len(testloader), dm)
                    
            g_rmse = torch.sqrt(torch.mean(torch.tensor(all_dms)))
            avg_overlap = overlaps/count
            
            est_gs = torch.cat(all_est_g, 0)   # [..., 4, 4]
            igt_gs = torch.cat(all_igt_g, 0)   # [..., 4, 4]
            metrics = compute_metrics(est_gs, igt_gs)
            
            vals = torch.tensor([avg_overlap, g_rmse]+metrics)
            valn = vals.cpu().numpy().tolist()
            print(','.join(map(str, valn)), file=fout)


    def do_estimate(self, p0, p1, model, device):
        p0 = p0.to(device) # template (1, N, 3)
        p1 = p1.to(device) # source (1, M, 3)
        r = ptlk.pointlk.PointLK.do_forward(model, p0, p1, self.max_iter, self.xtol,\
                                            self.p0_zero_mean, self.p1_zero_mean)
        #r = model(p0, p1, self.max_iter)
        est_g = model.g # (1, 4, 4)

        return est_g


def get_datasets(args):

    cinfo = None
    if args.categoryfile:
        #categories = numpy.loadtxt(args.categoryfile, dtype=str, delimiter="\n").tolist()
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)

    perturbations = None
    fmt_trans = False
    if args.perturbations:
        perturbations = numpy.loadtxt(args.perturbations, delimiter=',')
    if args.format == 'wt':
        fmt_trans = True

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
#                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
            ])

        testdata = ptlk.data.getdata.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo, crop = args.crop)

        testset = ptlk.data.getdata.CADset4tracking_fixed_perturbation(testdata,\
                        perturbations, fmt_trans=fmt_trans)

    return testset

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',fmt='%(levelname)s:%(name)s, %(asctime)s, %(message)s'):
        
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        
        #th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        #往文件里写入#指定间隔时间自动生成文件的处理器
        th = logging.FileHandler(filename=filename,mode='w')
        th.setFormatter(format_str)#设置文件里写入的格式
        
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)



if __name__ == '__main__':
    ARGS = options()

    LOGGER = Logger(ARGS.logfile,level='debug')
    LOGGER.logger.debug('Testing (PID=%d), %s', os.getpid(), ARGS)

    main(ARGS)
    LOGGER.logger.debug('done (PID=%d)', os.getpid())

#EOF