import argparse
import os
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    

    parser.add_argument('--dataset', type=str, default='DE')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu_ids', nargs="+", default=['5', '6', '7'])
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--port', type=int, default=2023)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--dist_url', type=str)

    return parser.parse_args()



def init_for_distributed(opts):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opts.rank = int(os.environ["RANK"])
        opts.world_size = len(opts.gpu_ids)
        opts.gpu = int(opts.gpu_ids[opts.rank])
        torch.cuda.set_device(opts.gpu)
        # opts.world_size = int(os.environ['WORLD_SIZE'])
        # opts.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        opts.distributed = False
        return

    print('| distributed init (rank {}): {}'.format(
        opts.rank, 'env://'), flush=True)

    torch.distributed.init_process_group(backend='nccl', init_method=opts.dist_url,
                                         world_size=opts.world_size, rank=opts.rank)
    # torch.distributed.barrier()
    setup_for_distributed(opts.rank==0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def calculate_loss(y_pred, y_true):
    """Calculates the loss between true and predicted using cross entropy

    Args:
        y_pred (T.tensor): Prediction from the model
        y_true (T.tensor): Ground Truth
        mask (T.tensor): Boolean tensor with True for masked tokens and False
        for unmasked

    Returns:
        T.tensor: Total loss
    """
    loss = F.cross_entropy(y_pred, y_true, ignore_index=0)
    # loss = loss * mask
    # loss = loss.sum() / (mask.sum() + 1e-8)
    
    return loss

