# GameAgent: 服务器操作指南

## 连接服务器

```bash
sshpass -p '123456' ssh -p 30022 wujn@root@ssh-362.default@222.223.106.147
```

## 一键启动实验

登录后执行以下命令（已全部准备好）：

```bash
# 进入项目
cd /gfs/space/private/wujn/Research/nips-gameagent

# 设置 conda 环境（首次运行）
export PATH=/home/nwh/anaconda3/bin:$PATH
eval "$(/home/nwh/anaconda3/bin/conda shell.bash hook)"

# 首次：创建环境
conda create -y -n gameagent python=3.10
conda activate gameagent
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install sentence-transformers

# 后续：激活环境
conda activate gameagent

# 启动快速验证（~1小时）
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$(pwd):$PYTHONPATH
nohup bash scripts/run_all_experiments.sh --quick > logs/pipeline_quick.log 2>&1 &
echo "PID=$!"

# 启动完整实验（~24小时）
nohup bash scripts/run_all_experiments.sh > logs/pipeline_full.log 2>&1 &
echo "PID=$!"
```

## 监控进度

```bash
# 查看实时日志
tail -f logs/pipeline_quick.log

# 查看 GPU 使用
nvidia-smi

# 查看各阶段日志
ls logs/*.log

# 检查结果
ls -la results/
```

## 收集结果

```bash
# 运行可视化
conda activate gameagent
python scripts/collect_and_visualize.py --results_dir results --output_dir to_human

# 打包结果下载
bash collect_results.sh
```

## 同步代码（从本地推送到服务器）

本地执行：
```bash
cd nips-gameagent
tar czf /tmp/ga.tar.gz --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='results' --exclude='data' --exclude='logs' --exclude='checkpoints' .
sshpass -p '123456' scp -P 30022 /tmp/ga.tar.gz wujn@root@ssh-362.default@222.223.106.147:/gfs/space/private/wujn/Research/nips-gameagent/
# 在服务器上解压
sshpass -p '123456' ssh -p 30022 wujn@root@ssh-362.default@222.223.106.147 "cd /gfs/space/private/wujn/Research/nips-gameagent && tar xzf ga.tar.gz && rm ga.tar.gz"
```
