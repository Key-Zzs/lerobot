# Le-nero

## TODOs

- [ ] waiting for dataset to validate act_dagger
- [ ] refactor dagger to online training

## Config

```bash
git clone --recurse-submodules https://github.com/Key-Zzs/Le-nero.git
```

```bash
# branch checkout: main/test_chunk-wise/worked_base_db44758
git checkout test_chunk-wise
git submodule foreach 'git checkout main' # submodule only have branch 'main' when Repo is on brach 'test_chunk-wise' 

# branch push
git submodule foreach 'git add . ;git commit -m "<commit_message>" || true'
git submodule foreach 'git push origin <branch_name>'
git push origin <branch_name>
```

如果忘记添加 `--recursive` 选项，需要手动克隆子模块：

```bash
cd agilex_ws/agilex_teleop

# 1. 初始化 submodule 配置
git submodule init

# 2. 拉取所有 submodule 的实际代码（递归，如果子模块还有子模块）
git submodule update --recursive
```

```bash
conda create -n dual_arm_teleop python=3.10 -y
conda activate dual_arm_teleop
```

### agilex_teleop

```bash
cd dual_arm_data_collection/agilex_teleop/
# git submodule init
# git submodule update --recursive
pip install -e .
pip install -r requirements.txt

```

```bash
sudo apt update && sudo apt install ethtool can-utils
bash pyAgxArm/scripts/ubuntu/find_all_can_port.sh
```

假设上面记录的 `USB port` 数值分别为 `3-1.4:1.0` 和 `3-1.1:1.0`，则将 [agilex_ws/agilex_teleop/pyAgxArm/scripts/ubuntu/can_muti_activate.sh](./dual_arm_data_collection/agilex_teleop/pyAgxArm/scripts/ubuntu/can_muti_activate.sh) 中的参数修改为：

```bash
USB_PORTS["3-1.4:1.0"]="can_left:1000000"
USB_PORTS["3-1.1:1.0"]="can_right:1000000"
```

含义：`3-1.4:1.0` 端口的 CAN 设备重命名为 `can_left`，波特率 `1000000`，并激活。

激活多个 CAN 模块

执行：

```bash
bash pyAgxArm/scripts/ubuntu/can_muti_activate.sh
```

运行 nero 测试脚本（最好运行，否则后续机械臂可能会处于未使能状态）

**注意**：[reset.py](./nero/tests/reset.py) 和 [test_pos_flw_ik.py](./nero/tests/test_pos_flw_ik.py) 均为单臂测试脚本，请运行单臂后修改文件中的 can 设备名，如 `can_left` 或 `can_right`，再运行下一个。
**注意**：
请保证 `bash pyAgxArm/scripts/ubuntu/find_all_can_port.sh` 输出有 `can_left` 和 `can_right` 两个 can 设备！！

```bash
# nero 关节重置脚本
python nero/tests/reset.py
```

```bash
# 启动 Server 服务
python nero/teleop/interface/nero_interface_server.py --ip 0.0.0.0 --port 4242

# 开放端口 4242（若 Server 端 PC 默认开放端口，无需此步）
udo iptables -I INPUT -p tcp --dport 4242 -j ACCEPT # iptables 方式
```

### lerobot_dual_arm_teleop

```bash
cd dual_arm_data_collection/lerobot_dual_arm_teleop/
pip install -e .

cd teleoperators/oculus_teleoperator/oculus
git clone https://github.com/rail-berkeley/oculus_reader.git
cd oculus_reader
pip install -e .
```

1. 安装 ADB（Android 调试桥）：Oculus Quest 与计算机之间通信必需的工具

   ```bash
   # 在 Ubuntu 上
   sudo apt install android-tools-adb

   # 验证安装
   adb version
   ```

2. 在 Oculus Quest 上启用开发者模式

   1. 在 [Meta for Developers](https://developer.oculus.com/manage/organizations/create/) 创建或加入开发者组织
   2. 在手机上打开 Meta Quest 应用
   3. 进入 **设置** → 选择您的设备 → **更多设置** → **开发者模式**
   4. 启用 **开发者模式** 开关

3. 连接 Oculus Quest 到计算机

   方式 1：USB 连接（推荐用于初始设置，或对实时性要求高的场景）

   1. 使用 USB-C 线缆将 Oculus Quest 连接到计算机
   2. 佩戴头显并在提示时允许 USB 调试
   3. 勾选 `始终允许来自此计算机`
   4. 验证连接：
   
      ```bash
      adb devices
      # 预期输出：
      # List of devices attached
      # <device_id>    device
      adb shell ip route
      # 查找 "src" 后面的 IP 地址，例如 192.168.110.62
      ```

   方式 2：无线连接（操作更便捷）

   1. 首先通过 USB 线缆连接 Oculus Quest 到计算机执行方案 1
   2. 确保 Oculus Quest 和计算机连接到同一网络
   3. 验证连接：
   
      ```bash
      adb connect <获取到的IP地址>:5555
      adb shell ip route
      # 查找 "src" 后面的 IP 地址，例如 192.168.110.62
      ```

   4. 在 `record_cfg.yaml` 中配置 IP：
   
      ```yaml
      teleop:
         oculus_config:
            ip: "192.168.110.62"  # 您的 Oculus Quest IP 地址
      ```

启动遥操作 Client 端服务

**注意**：
1. 启动前请 `adb devices` 检查 Oculus Quest 是否连接成功
2. 每次修改项目中的 python 文件后，需在项目根目录 `agilex_ws/dual_arm_teleop`  下执行 `pip install -e .` 更新依赖

```bash
# 重置机械臂
robot-reset
# 开始遥操作及数据采集
robot-record
# 右箭头：停止采集数据
# enter：继续遥操作
```

