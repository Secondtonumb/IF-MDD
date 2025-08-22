# 预训练组件加载功能说明

## 功能概述
这个功能允许你从一个已训练的模型中加载特定的组件（如SSL模型、encoder、decoder等），并选择性地冻结这些组件，只训练剩余的部分。

## 主要特性
1. **选择性加载**: 可以只加载模型的某些部分
2. **自动冻结**: 加载的组件可以自动冻结，不参与训练
3. **灵活配置**: 支持配置文件和代码两种方式
4. **状态监控**: 可以查看哪些参数被冻结了

## 支持的组件
- `ssl`: 感知SSL模型 (如WavLM, Wav2Vec2等)
- `encoder`: Transformer编码器部分
- `enc_projection`: 编码器投影层
- `ctc_head`: CTC分类头
- `decoder`: Transformer解码器和输出层

## 使用方法

### 方法1: 配置文件方式（推荐）
在`transformer.yaml`中添加：
```yaml
load_pretrained_components: true
pretrained_model_path: "/path/to/your/checkpoint/save/"
components_to_load: ["ssl", "encoder"]
freeze_loaded_components: true
```

### 方法2: 代码方式
```python
# 创建模型后手动加载
model.load_pretrained_components(
    checkpoint_path="/path/to/checkpoint",
    components_to_load=["ssl", "encoder"],
    freeze_loaded=True
)

# 或使用简化接口
model.load_from_checkpoint_manual(
    checkpoint_path="/path/to/checkpoint",
    freeze_ssl=True,
    freeze_encoder=True
)
```

### 方法3: 查看参数状态
```python
# 查看哪些参数被冻结了
model.print_parameter_status()
```

## 常用场景

### 1. 迁移学习
```yaml
components_to_load: ["ssl", "encoder"]
freeze_loaded_components: true
```
冻结特征提取部分，只训练分类头

### 2. 消融实验  
```yaml
components_to_load: ["ssl"]
freeze_loaded_components: true
```
固定SSL特征，测试不同encoder架构

### 3. 计算受限训练
```yaml
components_to_load: ["ssl", "encoder"]
freeze_loaded_components: true
```
冻结大部分参数，只训练轻量级组件

### 4. 分阶段训练
```yaml
components_to_load: ["ssl", "encoder", "ctc_head"]
freeze_loaded_components: true
```
固定已训练的部分，专注训练decoder

## 文件说明
- `models/Transformer.py`: 主要实现代码
- `hparams/transformer.yaml`: 配置文件
- `usage_examples.py`: 详细使用示例
- `quick_config_template.yaml`: 快速配置模板
- `load_pretrained_example.py`: 命令行工具

## 注意事项
1. 确保预训练模型的架构与当前模型兼容
2. 检查日志确认组件加载成功
3. 使用`print_parameter_status()`验证冻结状态
4. 冻结的参数不会在训练中更新
5. 可以随时调用`unfreeze_encoder_ssl()`来解冻

## 示例输出
```
🔄 Loading pretrained components from: /path/to/checkpoint
   Components to load: ['ssl', 'encoder']
   ✅ Loaded 768 parameters for perceived_ssl
   ✅ Loaded 2304 parameters for TransASR.encoder
   🔒 SSL model frozen
   🔒 Encoder frozen
   
📊 Model Parameter Status:
   perceived_ssl: 0/12,345,678 params 🔒 FROZEN
   TransASR: 1,234,567/5,678,901 params 🔓 TRAINABLE
   ...
   
📈 Summary:
   Trainable parameters: 2,345,678
   Frozen parameters: 15,678,901
   Frozen ratio: 87.0%
```

这样你就可以高效地进行迁移学习和模型实验了！
