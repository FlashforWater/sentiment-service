# 项目当前状态

## 完成度
- ✅ 阶段 1: 项目骨架 + Git
- ✅ 阶段 2: 环境隔离 + 依赖锁定
- ✅ 阶段 3: 训练 pipeline + MLflow(tag v0.3)
- ⏳ 阶段 4: 多次实验对比 ← 下次从这里开始
- ⬜ 阶段 5: FastAPI 推理服务
- ⬜ 阶段 6: Docker 化
- ⬜ 阶段 7: 文档收尾

## 下次开始时怎么恢复环境

```bash
cd /Users/sfffpanda/Project/Engineer_experience/sentiment-service
source .venv/bin/activate

# 如果要 push 到 GitHub,可能需要代理
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890

# 验证环境
which python      # 应该指向 .venv/bin/python
git status        # 应该是 clean
```

## 阶段 4 的目标
跑 3 个对比实验(改配置即可,不动代码):
1. run1: baseline(已有,1 epoch lr 2e-5)
2. run-lr5e5: lr 改 5e-5,看学习率敏感性
3. run-data5000: train_size 改 5000,看数据量影响

每次实验后在 MLflow UI 用 Compare 功能对比。
最终在 notes.md 里写结论:哪个超参影响最大?

## 已知问题
- HuggingFace 下载需要代理或镜像
- MLflow UI 在 http://localhost:5000(需要在新终端 mlflow ui 启动)
