from . import ppat

def make(config):
    print("Model name: ", config.model.name)
    if config.model.name == "PointBERT":
        model = ppat.make(config)
    elif config.model.name == "DGCNN":
        from . import dgcnn
        model = dgcnn.make(config)
    elif config.model.name == "PointNeXt":
        from . import pointnext
        model = pointnext.make(config)
    elif config.model.name == "PointMLP":
        from . import pointmlp
        model = pointmlp.make(config)
    elif config.model.name == "PointNet":
        from . import pointnet
        model = pointnet.make(config)
    else:
        raise NotImplementedError("Model %s not supported." % config.model.name)
    return model

def make_no_classification(config):
    print("Model name: ", config.model.name)
    if config.model.name == "PointBERT":
        model = ppat.make_no_classification(config)
    else:
        raise NotImplementedError("Model %s not supported." % config.model.name)
    return model
