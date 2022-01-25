import glob
import os
import time

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from quicktorch.utils import train, evaluate, get_splits
from quicktorch.writers import LabScribeWriter
from myvis.vis import visualise_attention
from experiment_utils import ExperimentParser, calculate_error
from segmentation import create_model, load
from cirrus.training_utils import construct_dataset, load_config, get_loss, get_metrics, create_attention_model


def write_results(**kwargs):
    sorted_keys = sorted([key for key in kwargs.keys()])
    result = '\n' + '-'.join([key + '=' + str(kwargs[key]) for key in sorted_keys])

    f = open("cirrus_results.txt", "a+")
    f.write(result)
    f.close()


def run_cirrus_split(net_args, args, exp_config, model_config, save_dir, writer=None,
                     device='cuda:0', split=None):
    dataset_train = construct_dataset(dataset=exp_config['dataset'], idxs=split[0], class_map=exp_config['class_map'], transform=exp_config['transforms'], aug_mult=exp_config['aug_mult'], padding=exp_config['padding'])
    dataset_val = construct_dataset(dataset=exp_config['dataset'], idxs=split[1], class_map=exp_config['class_map'], transform=exp_config['transforms'], padding=exp_config['padding'])
    train_data = DataLoader(dataset_train, exp_config['batch_size'], True, pin_memory=True)
    val_data = DataLoader(dataset_val, exp_config['batch_size'], True, pin_memory=True)
    num_classes = dataset_train.num_classes

    if model_config['model_variant'] == 'Attention':
        model = create_attention_model(
            len(exp_config['bands']),
            num_classes,
            model_config,
            pad_to_remove=exp_config['padding'],
            pretrain_path=args.model_path
        )
        model.save_dir = save_dir
    else:
        model = create_model(
            save_dir,
            variant=args.model_variant,
            n_channels=len(exp_config['bands']),
            n_classes=num_classes,
            bands=exp_config['bands'],
            padding=exp_config['padding'],
            class_map=exp_config['class_map'],
            name_params={
                'consensus': os.path.split(args.mask_dir)[-1],
                'loss': exp_config['loss_type'],
            },
            **vars(net_args)
        )
    model = model.to(device)

    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    total_params = total_params / 1000000
    print("Total # parameter: " + str(total_params) + "M")

    optimizer = optim.Adam(
        model.parameters(),
        lr=exp_config['lr'],
        weight_decay=exp_config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        exp_config['lr_decay']
    )
    class_balances = dataset_train.class_balances
    pos_weight = torch.sqrt(torch.tensor(class_balances, device=device))
    pos_weight = pos_weight.view(pos_weight.shape[0], 1, 1)
    criterion = get_loss(
        exp_config['seg_loss'],
        exp_config['consensus_loss'],
        exp_config['aux_loss'],
        pos_weight=pos_weight
    )

    metrics_class = get_metrics(dataset_train.num_classes, model_config['model_variant'])
    metrics_class.Writer = writer

    if os.path.exists(os.path.join(save_dir, 'current.pt')):
        checkpoint = torch.load(os.path.join(save_dir, 'current.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(checkpoint.keys())
        metrics_class.best_metrics.update(checkpoint['best_metrics'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    start = time.time()
    m = train(
        model,
        [train_data, val_data],
        criterion=criterion,
        save_best=True,
        epochs=exp_config['epochs'],
        start_epoch=start_epoch,
        opt=optimizer,
        device=device,
        sch=scheduler,
        metrics=metrics_class,
        val_epochs=15
    )

    time_taken = time.time() - start
    mins = int(time_taken // 60)
    secs = int(time_taken % 60)
    del(model)
    torch.cuda.empty_cache()
    stats = {
        'mins': mins,
        'secs': secs,
        'total_params': total_params
    }
    return m, stats


def run_evaluation_split(net_args, args, exp_config, model_config, model_path, test_idxs,
                         device='cuda:0', figs_dir=None):
    dataset_test = construct_dataset(
        dataset=exp_config['dataset'],
        idxs=test_idxs,
        class_map=exp_config['class_map'],
        transform={
            'crop': [3000, 3000],
            'resize': [1024, 1024],
            'pad': [1024 + exp_config['padding'], 1024 + exp_config['padding']]
        },
        padding=exp_config['padding']
    )
    test_data = DataLoader(dataset_test, 1)
    num_classes = dataset_test.num_classes

    if model_config['model_variant'] == 'Attention':
        model = create_attention_model(
            len(exp_config['bands']),
            num_classes,
            model_config,
            pad_to_remove=exp_config['padding'],
        )
    else:
        model = create_model(
            args.save_dir,
            variant=args.model_variant,
            n_channels=len(exp_config['bands']),
            n_classes=num_classes,
            bands=exp_config['bands'],
            padding=exp_config['padding'],
            class_map=args.class_map,
            **vars(net_args),
        )
    model = model.to(device)
    load(model, model_path, legacy=False, pretrain=False)

    metrics_class = get_metrics(num_classes, model_config['model_variant'])

    figs_dir = os.path.join(figs_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    eval_m = evaluate(
        model,
        test_data,
        device=device,
        metrics=metrics_class,
        figs_dir=figs_dir,
        figs_labels=dataset_test.galaxies
    )

    return eval_m


def visualise(net_args, args, exp_config, model_config, model_path,
              device='cuda:0'):
    dataset = construct_dataset(
        exp_config['dataset'],
        {
            'crop': [3000, 3000],
            'resize': [1024, 1024],
            'pad': [1024 + exp_config['padding'], 1024 + exp_config['padding']]
        },
        padding=exp_config['padding'],
        class_map=exp_config['class_map']
    )

    if model_config['model_variant'] == 'Attention':
        model = create_attention_model(
            len(exp_config['bands']),
            dataset.num_classes,
            model_config,
            pad_to_remove=exp_config['padding'],
        )
    else:
        model = create_model(
            args.save_dir,
            variant=args.model_variant,
            n_channels=len(exp_config['bands']),
            n_classes=dataset.num_classes,
            bands=exp_config['bands'],
            padding=exp_config['padding'],
            class_map=args.class_map,
            **vars(net_args),
        ).to(device)

    load(model, model_path, legacy=False, pretrain=False)

    img, mask = dataset.get_galaxy(args.galaxy)
    img = img.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    if 'DAF' in args.model_variant:
        visualise_attention(model, img, mask, torch.tensor([512, 512]))


def generate_splits(N, nsplits=1, val_ratio=0.2, test_ratio=0.15, force_test_idxs=[]):
    test_N = int(N * test_ratio)
    test_idxs = list(range(N - test_N, N))

    splits = get_splits(N - test_N, max(int(1 / val_ratio), nsplits))  # Divide into 6 or more blocks

    if force_test_idxs:
        to_be_swapped = test_idxs[:len(force_test_idxs)]
        for i, (idx, sw_idx) in enumerate(zip(force_test_idxs, to_be_swapped)):
            test_idxs[i] = idx
            for split in splits:
                for part in split:
                    part[part == idx] = sw_idx

    return splits, test_idxs


def get_test_idxs(dataset, exp_config):
    if 'test_galaxies' not in exp_config:
        return []
    else:
        return [i for i, gal in enumerate(dataset.galaxies) if gal in exp_config['test_galaxies']]


def main():
    parser = ExperimentParser(description='Runs a segmentation model')
    parser.n_parser.set_defaults(dataset='cirrus')
    parser.add_argument('--save_dir',
                        default='models/seg/cirrus', type=str,
                        help='Directory to save models to. (default: %(default)s)')
    parser.add_argument('--model_path',
                        default='', type=str,
                        help='Path to model, enabling pretraining/evaluation. (default: %(default)s)')
    parser.add_argument('--checkpoint_dir',
                        default='', type=str,
                        help='Path to checkpoint directory. (default: %(default)s)')
    parser.add_argument('--auto_checkpoint',
                        default=False, action='store_true',
                        help='Automatically uses last version directory as checkpoint dir.')
    parser.add_argument('--experiment_config',
                        default='./configs/experiments/cirrus/standard.yaml', type=str,
                        help='Experiment configuration. (default: %(default)s)')
    parser.add_argument('--default_experiment_config',
                        default='./configs/experiments/default.yaml', type=str,
                        help='Experiment configuration. (default: %(default)s)')
    parser.add_argument('--evaluation_config',
                        default='./configs/experiments/cirrus/evaluation.yaml', type=str,
                        help='Experiment configuration. (default: %(default)s)')
    parser.add_argument('--model_config',
                        default='./configs/models/triplemsguided.yaml', type=str,
                        help='Model configuration. (default: %(default)s)')
    parser.add_argument('--model_variant',
                        default='SFC', type=str,
                        choices=[
                            'SFC', 'SFCT', 'SFCP', 'SFCReal',
                            'DAF', 'DAFT', 'DAFP',
                            'DAFMS', 'DAFMST', 'DAFMSP',
                            'DAFMSPlain', 'DAFMSPlainT', 'DAFMSPlainP',
                            'Standard', 'StandardT', 'StandardP',
                        ], help='Model variant. (default: %(default)s)')
    parser.add_argument('--evaluate',
                        default=False, action='store_true',
                        help='Evaluates given model path.')
    parser.add_argument('--visualise',
                        default=False, action='store_true',
                        help='Visualises network outputs for given galaxy.')
    parser.add_argument('--galaxy',
                        default='NGC1121', type=str,
                        help='Which galaxy to visualise. (default: %(default)s)')

    net_args, _ = parser.parse_group_args()
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_config = load_config(args.experiment_config, args.default_experiment_config)
    eval_config = load_config(args.evaluation_config, args.default_experiment_config)
    model_config = load_config(args.model_config)

    if args.checkpoint_dir:
        args.save_dir = args.checkpoint_dir
        ver_dir = os.path.split(args.save_dir)[-1]
    else:
        args.save_dir = os.path.join(args.save_dir, exp_config['dataset'], f'{model_config["name"]}_{exp_config["name"]}')
        ver_no = len(glob.glob(args.save_dir + '/*/'))
        if args.auto_checkpoint:
            ver_no = max(0, ver_no - 1)
        ver_dir = 'ver' + str(ver_no)
        args.save_dir = os.path.join(args.save_dir, ver_dir)
        os.makedirs(args.save_dir, exist_ok=True)

    metrics = []
    save_paths = []
    dataset = construct_dataset(dataset=exp_config['dataset'], class_map=exp_config['class_map'], padding=exp_config['padding'])
    N = len(dataset)
    force_test_idxs = get_test_idxs(dataset, exp_config)
    splits, test_idxs = generate_splits(N, exp_config['nsplits'], force_test_idxs=force_test_idxs)
    num_classes = dataset.num_classes
    del dataset

    if args.evaluate:
        eval_m = run_evaluation_split(
            net_args,
            args,
            eval_config,
            model_config,
            args.model_path,
            test_idxs,
            device='cuda:0',
        )
        return

    if args.visualise:
        visualise(
            net_args,
            args,
            exp_config,
            model_config,
            args.model_path,
            device='cuda:0',
        )
        return

    exp_name = '-'.join([
        f'class_map={exp_config["class_map"]}',
        f'exp_config={exp_config["name"]}',
        f'model_config={model_config["name"]}',
        f'version={ver_dir}',
    ])
    writer = LabScribeWriter(
        'Results',
        exp_name=exp_name,
        exp_worksheet_name='CirrusSeg',
        metrics_worksheet_name='CirrusSegMetrics',
        nsplits=exp_config['nsplits']
    )
    writer.begin_experiment(exp_name)
    for split_no, split in zip(range(exp_config['nsplits']), splits):
        print('Beginning split #{}/{}'.format(split_no + 1, exp_config['nsplits']))
        save_dir = os.path.join(args.save_dir, f'split{split_no + 1}')
        os.makedirs(save_dir, exist_ok=True)
        m, stats = run_cirrus_split(
            net_args,
            args,
            exp_config,
            model_config,
            save_dir,
            writer=writer,
            device=device,
            split=split
        )
        save_paths.append(m.pop('save_path'))

        if not exp_config['eval_best'] and exp_config['eval']:
            m = run_evaluation_split(
                net_args,
                args,
                eval_config,
                model_config,
                save_paths[-1],
                test_idxs,
                device='cuda:0',
                figs_dir=save_dir
            )

        metrics.append(m)
        writer.upload_split({k: str(m[k]) for k in ('IoU',)})

    # mean_m = {key: sum(mi[key] for mi in metrics) / exp_config['nsplits'] for key in m.keys()}
    def avg(x, n_classes):
        if n_classes > 1:
            return sum(x) / n_classes
        else:
            return x

    mean_m = {key: sum([avg(mi[key], num_classes) for mi in metrics]) / exp_config['nsplits']
        for key in ('IoU',)}
    best_iou = max([avg(mi['IoU'], num_classes) for mi in metrics])
    best_split = [avg(mi['IoU'], num_classes) for mi in metrics].index(best_iou) + 1
    # best_psnr = metrics[[mi['IoU'] for mi in metrics].index(best_iou)]['PSNR']
    # mean_m['epoch'] = metrics[best_split-1]['epoch']

    error_m = {'e_psnr': 0}
    if exp_config['nsplits'] > 1:
        error_m = {f'e_{key}': calculate_error([avg(mi[key], num_classes) for mi in metrics])
                for key in ('IoU',)}

    if exp_config['eval_best'] and exp_config['eval']:
        eval_m = run_evaluation_split(
            net_args,
            args,
            eval_config,
            model_config,
            save_paths[best_split-1],
            test_idxs,
            device='cuda:0',
        )
    else:
        eval_m = metrics[best_split-1]
    print(metrics)
    print(eval_m)

    total_params = stats['total_params']
    mins = stats['mins']
    secs = stats['secs']
    writer.upload_best_split(
        {
            'test_iou': str(eval_m['IoU']),
            'mean_iou': str(mean_m['IoU']),
            'best_split': best_split,
        },
        best_split
    )

    exp_config['exp_name'] = exp_config['name']
    model_config['model_name'] = model_config['name']
    del exp_config['name']
    del model_config['name']
    write_results(
        **{
            'params': total_params,
            # 'm_psnr': mean_m['PSNR'],
            'm_iou': mean_m['IoU'],
            'best_iou': best_iou,
            # 'best_psnr': best_psnr,
            'best_split': best_split,
            'time_split': "{:3d}m{:2d}s".format(mins, secs),
            'zpaths': save_paths,
        },
        **error_m,
        **eval_m,
        **exp_config,
        **model_config,
    )


if __name__ == '__main__':
    main()
