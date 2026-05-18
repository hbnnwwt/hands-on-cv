# Hands-On Computer Vision: From Pixels to Applications — 配套代码仓库

本仓库包含教材 *Hands-On Computer Vision: From Pixels to Applications* 的所有可运行代码示例，按章节组织。

面向人工智能专业和计算机相关专业，非相关专业学生也可通过项目驱动与 AI 辅助方式学习。

## 目录结构

```
教材代码/
├── README.md                   # 本文件
├── requirements.txt            # 依赖清单
├── setup.py                    # 安装脚本
├── common/                     # 跨章节共享工具
│   ├── utils.py
│   ├── visualization.py
│   └── test_images.py          # 生成测试图像
├── data/                       # 示例数据（小图像）
│   ├── sample_paper.jpg
│   ├── handwriting.jpg
│   └── README.md
├── chapter01/                  # 第1章：导论
├── chapter02/                  # 第2章：图像数字化
├── chapter03/                  # 第3章：NumPy基础
├── chapter04/                  # 第4章：AI辅助编程
├── chapter05/                  # 第5章：图像预处理
├── chapter06/                  # 第6章：几何变换
├── chapter07/                  # 第7章：边缘检测
├── chapter08/                  # 第8章：版面分析
├── chapter09/                  # 第9章：OMR
├── chapter10/                  # 第10章：模板匹配
├── chapter11/                  # 第11章：深度学习基础
├── chapter12/                  # 第12章：CNN
├── chapter13/                  # 第13章：OCR
├── chapter14/                  # 第14章：手写识别
├── chapter15/                  # 第15章：目标检测
├── chapter16/                  # 第16章：生成模型
├── chapter17/                  # 第17章：项目实战
├── chapter18/                  # 第18章：测试部署
└── tests/                      # 跨章节集成测试
```

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate    # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行示例

每个章节的代码都可以独立运行：

```bash
# 第2章 — 读取并显示图像
python chapter02/read_show.py data/sample_paper.jpg

# 第5章 — 高斯滤波去噪
python chapter05/gaussian_filter.py data/noisy_image.jpg

# 第9章 — OMR 识别完整流程
python chapter09/omr_pipeline.py data/answer_sheet.jpg

# 第17章 — 启动智能阅卷系统
python chapter17/main.py --mode single --image data/answer_sheet.jpg
```

### 3. 测试

```bash
# 运行所有单元测试
pytest tests/

# 运行特定章节的测试
pytest tests/test_chapter09.py -v

# 生成覆盖率报告
pytest --cov=. --cov-report=html
```

## 章节代码索引

| 章节 | 关键代码文件 | 主题 |
|------|------------|------|
| chapter02 | read_show.py, color_space.py | 图像基础操作 |
| chapter03 | array_ops.py, broadcasting.py | NumPy 实战 |
| chapter05 | gaussian_filter.py, otsu_threshold.py | 预处理 |
| chapter06 | perspective_correction.py | 透视矫正 |
| chapter07 | canny_demo.py, contour_analysis.py | 边缘和轮廓 |
| chapter08 | layout_analyzer.py | 版面分析 |
| chapter09 | omr_pipeline.py | OMR 识别 |
| chapter10 | template_matching.py, hu_moments.py | 形状识别 |
| chapter11 | mlp_numpy.py, mlp_pytorch.py | MLP 实现 |
| chapter12 | lenet.py, transfer_learning.py | CNN |
| chapter13 | paddleocr_demo.py | OCR |
| chapter14 | trocr_inference.py | 手写识别 |
| chapter15 | yolov8_demo.py, iou.py | 目标检测 |
| chapter16 | gan_mnist.py, clip_demo.py | 生成模型 |
| chapter17 | auto_grading/ | 完整阅卷系统 |
| chapter18 | benchmark.py, onnx_export.py | 测试与部署 |

## 数据集说明

`data/` 目录包含本仓库自带的示例图像（< 1MB 总大小）。完整的训练数据集需要从公开来源下载，参见教材附录C。

## 开发者指南

### 代码规范

- 所有公开函数必须有 docstring
- 使用类型注解（type hints）
- 文件名使用 snake_case
- 类名使用 PascalCase

### 添加新示例

新增代码示例时：
1. 放在对应章节目录下
2. 文件顶部说明对应教材小节
3. 提供 `if __name__ == "__main__"` 入口
4. 添加 README 索引

## 许可证

MIT License. 教学用途，引用请注明出处。
