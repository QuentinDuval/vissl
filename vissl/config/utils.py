# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from vissl.config.attr_dict import AttrDict


def is_feature_extractor_model(model_config: AttrDict):
    """
    If the model is a feature extractor model:
        - evaluation model is on
        - trunk is frozen
        - number of features specified for features extraction > 0
    """
    if (
        model_config.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON
        and model_config.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY
        and len(model_config.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP) > 0
    ):
        return True
    return False


def get_trunk_output_feature_names(model_config):
    """
    Get the feature names which we will use to associate the features witl.
    If Feature eval mode is set, we get feature names from
    config.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP.
    """
    feature_names = []
    if is_feature_extractor_model(model_config):
        feat_ops_map = model_config.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP
        feature_names = [item[0] for item in feat_ops_map]
    return feature_names
