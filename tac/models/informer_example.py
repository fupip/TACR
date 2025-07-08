# coding=utf-8
"""
Informer模型使用示例
演示如何使用Informer模型进行时间序列预测
"""

import torch
import torch.nn as nn
import numpy as np
from informer import InformerConfig, InformerModel, InformerForPrediction


def create_sample_data(batch_size=32, seq_len=96, pred_len=24, features=7):
    """创建示例时间序列数据"""
    # 编码器输入: [batch_size, seq_len, features]
    x_enc = torch.randn(batch_size, seq_len, features)
    
    # 编码器时间标记: [batch_size, seq_len, 4] (月,日,星期,小时)
    x_mark_enc = torch.randint(0, 12, (batch_size, seq_len, 4)).float()
    
    # 解码器输入: [batch_size, label_len + pred_len, features]
    # label_len是用于解码器的历史长度，pred_len是预测长度
    label_len = seq_len // 2
    x_dec = torch.randn(batch_size, label_len + pred_len, features)
    
    # 解码器时间标记
    x_mark_dec = torch.randint(0, 12, (batch_size, label_len + pred_len, 4)).float()
    
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def create_informer_model():
    """创建Informer模型配置和实例"""
    # 配置Informer模型
    config = InformerConfig(
        seq_len=96,          # 输入序列长度
        label_len=48,        # 标签长度
        pred_len=24,         # 预测长度
        d_model=512,         # 模型维度
        n_head=8,            # 注意力头数
        n_layer=6,           # 编码器层数
        n_embd=7,            # 输入特征数
        d_ff=2048,           # Feed Forward层维度
        factor=5,            # ProbSparse注意力采样因子
        dropout=0.05,        # Dropout率
        distil=True,         # 是否使用注意力蒸馏
        activation_function="gelu"
    )
    
    # 创建模型实例
    model = InformerForPrediction(config)
    return model, config


def train_step_example():
    """展示训练步骤的示例"""
    print("=== Informer模型训练示例 ===")
    
    # 创建模型
    model, config = create_informer_model()
    
    # 创建示例数据
    batch_size = 32
    x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_data(
        batch_size=batch_size,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        features=config.n_embd
    )
    
    print(f"输入数据形状:")
    print(f"  编码器输入: {x_enc.shape}")
    print(f"  编码器时间标记: {x_mark_enc.shape}")
    print(f"  解码器输入: {x_dec.shape}")
    print(f"  解码器时间标记: {x_mark_dec.shape}")
    
    # 前向传播
    model.train()
    try:
        outputs = model(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec
        )
        
        print(f"\n模型输出:")
        print(f"  预测结果形状: {outputs.prediction.shape}")
        print(f"  编码器输出形状: {outputs.last_hidden_state.shape}")
        
        # 假设我们有真实标签用于计算损失
        # 通常预测目标是解码器输入的后pred_len部分
        pred_len = config.pred_len
        target = x_dec[:, -pred_len:, :]  # [batch_size, pred_len, features]
        prediction = outputs.prediction[:, -pred_len:, :]  # [batch_size, pred_len, features]
        
        # 计算MSE损失
        criterion = nn.MSELoss()
        loss = criterion(prediction, target)
        
        print(f"  训练损失: {loss.item():.6f}")
        
        # 反向传播
        loss.backward()
        print("反向传播完成!")
        
    except Exception as e:
        print(f"模型前向传播出错: {e}")
        return False
    
    return True


def inference_example():
    """展示推理的示例"""
    print("\n=== Informer模型推理示例 ===")
    
    # 创建模型
    model, config = create_informer_model()
    
    # 创建单个样本数据用于推理
    batch_size = 1
    x_enc, x_mark_enc, x_dec, x_mark_dec = create_sample_data(
        batch_size=batch_size,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        features=config.n_embd
    )
    
    # 推理模式
    model.eval()
    with torch.no_grad():
        outputs = model(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec
        )
        
        prediction = outputs.prediction
        print(f"推理结果形状: {prediction.shape}")
        print(f"预测值范围: [{prediction.min().item():.4f}, {prediction.max().item():.4f}]")


def model_architecture_info():
    """显示模型架构信息"""
    print("\n=== Informer模型架构信息 ===")
    
    model, config = create_informer_model()
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型配置:")
    print(f"  序列长度: {config.seq_len}")
    print(f"  预测长度: {config.pred_len}")
    print(f"  模型维度: {config.d_model}")
    print(f"  注意力头数: {config.n_head}")
    print(f"  编码器层数: {config.n_layer}")
    print(f"  特征数: {config.n_embd}")
    print(f"  ProbSparse因子: {config.factor}")
    
    print(f"\n参数统计:")
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数数: {trainable_params:,}")
    
    # 显示模型主要组件
    print(f"\n主要组件:")
    print(f"  编码器层数: {len(model.informer.encoder.attn_layers)}")
    print(f"  解码器层数: {len(model.informer.decoder.layers)}")
    print(f"  是否使用蒸馏: {config.distil}")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 显示模型架构信息
    model_architecture_info()
    
    # 运行训练示例
    success = train_step_example()
    
    if success:
        # 运行推理示例
        inference_example()
        
        print(f"\n=== Informer模型特点 ===")
        print("1. ProbSparse自注意力: 降低计算复杂度O(L log L)")
        print("2. 自注意力蒸馏: 通过卷积层减少网络冗余")
        print("3. 生成式解码器: 一次性生成长序列预测")
        print("4. 多种时间嵌入: 支持位置、时间特征嵌入")
        print("5. 编码器-解码器架构: 适合长序列时间序列预测")
        
    else:
        print("模型运行失败，请检查配置和依赖") 