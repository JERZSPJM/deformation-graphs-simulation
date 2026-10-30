# 材料形变与能量转化模拟

用 NumPy 和 Matplotlib 模拟一张由弹簧连接点组成的二维材料片：左边固定，右边平滑拉伸并保持。程序把“形变图”、拉力响应、功率变化，以及输入功如何转成弹性势能、动能和阻尼耗散画在一张图里。

## 运行

需要 Python 3.10 或更高版本。安装依赖并运行：

```bash
python -m pip install -r requirements.txt
python deformation_graphs.py
```

默认生成：

- `deformation_simulation.png`：四幅图组成的结果图
- `deformation_simulation_timeseries.csv`：每个时间步的位移、应变、拉力、功率和能量数据

例如，把目标应变改为 15%，并把图保存到其他路径：

```bash
python deformation_graphs.py --strain 0.15 --output results/tension.png
```

添加 `--show` 可以在保存后打开图窗。更多选项可用 `python deformation_graphs.py --help` 查看。

## 模型说明

- 网格节点通过水平、竖直和对角弹簧连接；每条弹簧遵循胡克定律，弹簧力为 `k × (当前长度 − 原长)`。
- 内部节点按牛顿第二定律运动，并受到与速度成正比的粘性阻尼。
- 左边界固定，右边界按平滑的三次曲线移动；拉伸结束后保持位置。
- 输入功率由夹具反力乘以夹具速度得到；功率对时间积分得到输入功。能量图同时显示弹簧势能、内部节点动能、累计阻尼耗散和能量平衡残差。
- 形变图用颜色表示最终各连接边的轴向应变，灰色虚线是初始网格。

这是用于教学演示的离散弹簧网络，参数为模型单位，不能直接当作某种真实材料的实验预测。模型不含塑性、断裂、温度效应或经过标定的连续介质本构关系。
