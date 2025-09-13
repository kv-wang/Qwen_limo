# 这是一个补丁文件，用于修改 accelerate 库的 accelerator.py
# 使用方法：将此文件复制到 accelerate 库的安装目录并替换原文件

def _prepare_fsdp(self, *args):
    result = []
    for obj in args:
        if isinstance(obj, torch.nn.Module):
            model = obj
            break
    optimizers = []

    self._schedulers = []
    self._models = []
    intermediate_result = []
    for obj in args:
        if isinstance(obj, torch.optim.Optimizer):
            
            if len(obj.param_groups) > 1:
                logger.warning(
                    "FSDP Warning: When using FSDP, several parameter groups will be conflated into "
                    "a single one due to nested module wrapping and parameter flattening."
                )
            
            # 特殊处理 Muon 优化器
            if hasattr(obj.optimizer, '__class__') and obj.optimizer.__class__.__name__ == 'Muon':
                try:
                    # 对于 Muon，我们需要重新构建参数组
                    from muon import Muon
                    
                    # 获取原始参数组配置
                    original_groups = obj.optimizer.param_groups
                    
                    # 重新构建参数组，使用新的参数
                    new_param_groups = []
                    for group in original_groups:
                        new_group = group.copy()
                        new_group['params'] = list(model.parameters())  # 使用新的参数
                        new_param_groups.append(new_group)
                    
                    optimizer = Muon(new_param_groups)
                except Exception as e:
                    logger.warning(f"Failed to recreate Muon optimizer with param groups: {e}")
                    # 回退到原始方法
                    try:
                        optimizer = obj.optimizer.__class__(model.parameters(), **obj.optimizer.defaults)
                    except TypeError:
                        if "differentiable" in obj.optimizer.defaults:
                            defaults = {k: v for k, v in obj.optimizer.defaults.items() if k != "differentiable"}
                            optimizer = obj.optimizer.__class__(model.parameters(), **defaults)
                        else:
                            raise
            else:
                # 原始逻辑：处理其他优化器
                try:
                    optimizer = obj.optimizer.__class__(model.parameters(), **obj.optimizer.defaults)
                except TypeError:
                    if "differentiable" in obj.optimizer.defaults:
                        # https://github.com/huggingface/accelerate/issues/801
                        defaults = {k: v for k, v in obj.optimizer.defaults.items() if k != "differentiable"}
                        optimizer = obj.optimizer.__class__(model.parameters(), **defaults)
                    else:
                        raise
            obj = self.prepare_optimizer(optimizer)
            optimizers.append(obj)
        elif isinstance(obj, torch.nn.Module):
            self._models.append(obj)
        intermediate_result.append(obj)

    for obj in intermediate_result:
        if isinstance(obj, AcceleratedScheduler):
            obj.optimizer = optimizers
            for i, opt in enumerate(self._optimizers):
                if getattr(obj.scheduler, "optimizer", None) == opt.optimizer:
                    obj.scheduler.optimizer = optimizers[i]
                    obj.optimizers = [optimizers[i]]
                    break
            self._schedulers.append(obj)
        result.append(obj)
    self._optimizers = optimizers
    return tuple(result)

