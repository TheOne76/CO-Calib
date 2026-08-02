# Kalibr 多相机标定可视化工具

[English](README.md) | 简体中文

一个无需后端依赖的 Kalibr 多相机外参可视化工具。YAML 文件仅在浏览器本地解析，不会上传。

## 环境与运行

```bash
conda env create -f environment.yml
conda activate kalibr-visualizer
python -m http.server 8765
```

在浏览器中打开 `http://127.0.0.1:8765/`。

Python 只用于启动静态文件服务器，程序没有第三方 Python 依赖。

## 输入

选择或拖入 Kalibr 生成的 `camchain.yaml`。文件需要包含连续的 `cam0`、`cam1` 等节点，并且 `cam1` 之后的每个相机均包含 4x4 `T_cn_cnm1`。支持 YAML 的行内矩阵和块序列矩阵写法。

## 坐标变换

矩阵名遵循 `T_A_B`，即将 B 坐标系中的点变换到 A 坐标系。

```text
T_ci_c0 = T_ci_ci-1 * ... * T_c1_c0
T_c0_ci = inverse(T_ci_c0)
T_W_ci  = T_W_c0 * T_c0_ci

T_W_c0 = [0 0 1 0]
         [1 0 0 0]
         [0 1 0 0]
         [0 0 0 1]
```

世界坐标为 X 前、Y 左、Z 上。所有相机转换到世界系后，程序统一平移其 XY 坐标，使世界坐标系原点位于所有相机中心的 XY 平均点。Z 坐标、姿态和相机间相对位置保持不变。

相机视锥使用原始光学轴。仅坐标轴箭头使用以下显示映射：

```text
X_display = -X_optical
Y_display = -Y_optical
Z_display =  Z_optical
```

## 操作

- 左键拖动：旋转
- 右键或 Alt + 拖动：平移
- 鼠标滚轮：缩放
