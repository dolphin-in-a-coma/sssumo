"""Reading and writing training checkpoints.

Checkpoints used to be a bare ``model.state_dict()``. That is enough to *evaluate*
a run but not to *continue* one: the optimiser's Adam moments, the learning-rate
scheduler's step count, and the reconstructor's shape parameters all live outside
the detector, so a resumed run silently restarted its optimiser from zero and lost
whatever shape it had learned. Runs resumed a different number of times were then
not comparable with each other, which is a confound rather than a nuisance when the
arms of a study are being compared to one another.

Format 2 keeps all of it in one file. `load_model_state` reads either format, so
every checkpoint written before this module -- including the released
``config-0423-ModGaussian_ampl_24.pth`` -- still loads unchanged.
"""

import torch

FORMAT = 2


def save_checkpoint(path, model, epoch, optimizer=None, scheduler=None, primitive=None):
    """Write a format-2 checkpoint holding everything needed to resume."""
    payload = {
        '_format': FORMAT,
        'epoch': epoch,
        'model': model.state_dict(),
    }
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        payload['scheduler'] = scheduler.state_dict()
    if primitive is not None:
        payload['primitive'] = primitive.state_dict()
        payload['primitive_family'] = primitive.family
    torch.save(payload, path)


def load_checkpoint(path, map_location=None):
    """Load a checkpoint of either format, always as a format-2 style dict.

    A format-1 file (a bare state_dict) comes back as ``{'model': ...}`` with no
    optimiser, scheduler or primitive, which is exactly what it contains.
    """
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and obj.get('_format'):
        return obj
    return {'_format': 1, 'model': obj}


def load_model_state(path, map_location=None):
    """The detector weights from a checkpoint of either format."""
    return load_checkpoint(path, map_location)['model']


def restore_training_state(checkpoint, optimizer=None, scheduler=None, primitive=None):
    """Restore optimiser, scheduler and shape state in place from a loaded checkpoint.

    Returns the names of the pieces restored, so the caller can say plainly what
    carried over and what did not. A format-1 checkpoint restores nothing, which is
    the honest outcome -- it never held any of it.
    """
    restored = []
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        restored.append('optimizer')
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        restored.append('scheduler')
    if primitive is not None and 'primitive' in checkpoint:
        saved_family = checkpoint.get('primitive_family')
        if saved_family is not None and saved_family != primitive.family:
            raise ValueError(
                f'checkpoint carries a {saved_family} primitive but the config asks for '
                f'{primitive.family}; refusing to load one family\'s shape into another')
        primitive.load_state_dict(checkpoint['primitive'])
        restored.append('primitive')
    return restored
