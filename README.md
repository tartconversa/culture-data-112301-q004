# 文化旅游引流的增量测算与预约承接

用于本地业务开发的合成经营资料，所有参与方均使用虚构标识。

data保存UTF-8 JSONL记录及catalog.json格式登记，docs/资料契约.md说明来源口径、字段、单位及直接引用。不同来源、业务时期与后续到达的记录各自保留身份，不能仅按文件行数解释业务数量。

Python需要3.11至3.14。依赖版本在requirements.txt中固定，可在独立虚拟环境安装。

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 tools/check_materials.py
python3 -m unittest discover -s tests -v
```

Windows PowerShell中可用`.venv\Scripts\Activate.ps1`激活虚拟环境。

资料检查入口输出各表行数，验证内容指纹、字段类型及直接外键。现有测试检查资料完整性和损坏检测，其通过只说明资料可读取。data及docs中的记录不包含经营分析结果。应用产生的本地状态可放入state，导出内容可放入outputs，两者已从Git排除。

## 应用：tourflow

`tourflow` 包在合成资料上实现完整业务路径：增量测算 → 需求预测 → 承诺试算与接受 → 履约核销 → 溯源审计。业务口径与使用说明见 docs/使用路径.md。

```sh
python3 -m tourflow.cli validate      # 资料校验
python3 -m tourflow.cli analyze       # 各推广线路的新增到访与不确定性
python3 -m tourflow.cli forecast      # 客源地×线路×日需求范围与回测
python3 -m tourflow.cli plan          # 承诺试算（--commit 落账）
python3 -m tourflow.cli fulfill       # 关闭替代与核销分类
python3 -m tourflow.cli audit         # 承诺溯源与可承诺量变化
python3 -m tourflow.cli demo          # 依次跑通整条路径
```

分析纪律：无法支持归因的增量不计入已证实成果；同一人跨场馆到访不加总为新增人数；售罄日按右截尾处理；外溢客源不充当干净对照。承诺纪律：整团全有或全无、重复接受幂等、已接受团体不被牺牲、证据不足只小规模试行。
