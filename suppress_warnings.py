# coding=utf-8
"""
PyTorch Backward Hook 警告抑制工具
解决 "Using a non-full backward hook" 等警告信息
"""

import warnings
import torch

def suppress_pytorch_warnings():
    """抑制PyTorch相关的警告信息"""
    
    # 抑制PyTorch backward hook相关警告
    warnings.filterwarnings("ignore", message="Using a non-full backward hook*")
    warnings.filterwarnings("ignore", message="Using non-full backward hooks*")
    
    # 抑制transformers相关警告
    warnings.filterwarnings("ignore", message="Some weights of*")
    warnings.filterwarnings("ignore", message="The following parameters*")
    
    # 抑制其他常见警告
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    print("✅ PyTorch 警告已被抑制")

def apply_full_backward_hooks():
    """为模型应用完整的backward hooks以避免警告"""
    
    def register_full_hooks(module):
        """递归为模块注册完整的backward hooks"""
        for child in module.children():
            register_full_hooks(child)
        
        # 如果模块有现有的hooks，替换为full hooks
        if hasattr(module, '_backward_hooks') and len(module._backward_hooks) > 0:
            hooks_to_replace = list(module._backward_hooks.items())
            for handle_id, hook_fn in hooks_to_replace:
                # 移除旧hook
                module._backward_hooks.pop(handle_id)
                # 注册新的full hook
                module.register_full_backward_hook(hook_fn)
    
    return register_full_hooks

# 使用示例
if __name__ == "__main__":
    # 在训练脚本开头调用
    suppress_pytorch_warnings()
    
    # 或者对特定模型应用
    import torch.nn as nn
    model = nn.Linear(10, 1)
    apply_fn = apply_full_backward_hooks()
    apply_fn(model)
    
    print("警告抑制设置完成！") 