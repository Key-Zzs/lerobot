#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train ACT with DAgger on real-robot rollouts.

Example:

```shell
lerobot-train-dagger \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodemXYZ \
  --robot.id=follower \
  --expert.mode=teleop \
  --expert.teleop.type=so100_leader \
  --expert.teleop.port=/dev/tty.usbmodemABC \
  --expert.teleop.id=leader \
  --dataset.seed_repo_id=my_user/my_bc_dataset \
  --dataset.seed_root=/data/my_bc_dataset \
  --policy.path=/path/to/act_bc_checkpoint \
  --rollout.single_task="pick-and-place" \
  --training.rounds=8 \
  --training.steps_per_round=2000
```
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.policies.dagger.configuration_dagger import DAggerPipelineConfig
from lerobot.policies.dagger.dagger_trainer import train_dagger
from lerobot.robots import (  # noqa: F401
    bi_so100_follower,
    hope_jr,
    koch_follower,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    so100_leader,
    so101_leader,
)
from lerobot.utils.import_utils import register_third_party_devices


@parser.wrap()
def run(cfg: DAggerPipelineConfig):
    train_dagger(cfg)


def main():
    register_third_party_devices()
    run()


if __name__ == "__main__":
    main()
