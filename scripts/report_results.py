"""Render only measured result files into the delivery report."""
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
def read(name): return json.loads((ROOT/name).read_text(encoding="utf-8"))
def mean(rows,key): return statistics.mean(row[key] for row in rows)


def main():
    final=read("results/prism-benchmark-20261016.json")
    legacy=read("results/legacy-benchmark-20261016.json")
    scale=read("results/scale-20261016.json")
    visual=read("results/query-reranker-final-20261016.json")
    docker=read("results/docker-release-final-20261016.json")
    testlog=(ROOT/"results/tests-release-20261016.txt").read_text()
    counts=re.search(r"Ran (\d+) tests in ([0-9.]+)s\s+OK",testlog)
    if not counts: raise ValueError("Final tests are incomplete or failed")
    lines=["# Prism-Swift：实际性能与验证报告", "", "运行日期：2026-10-15–16；交付整理：2026-10-16。所有数据均为自建合成数据。", "",
           "## 1. 已完成与边界", "",
           f"- 最终完整单元/真实本机接口回归：**{counts[1]} 项通过，{counts[2]} 秒，exit 0**。",
           "- 真正运行了默认 Docker 服务：UID10001、WAL/FULL、鉴权、立即返回原图、幂等、并发2、重启和SIGKILL后稳定ID/原图保持。",
           f"- Docker报告状态：`{docker['status']}`；实际测试镜像 `{docker.get('image_id')}`。",
           "- 默认镜像使用已缓存的官方Python slim基础镜像和公开TUNA包镜像；不是声称默认DockerHub网络已通。可选视觉Docker镜像未构建。",
           "- 本环境SSH认证失败（publickey,password），因此未在47.111.109.198部署或验证新8003公网接口；GitHub公开发布、官方Smoke/Full均未运行。",
           "- 记忆服务测试没有远程LLM/Embedding API调用；TinyCLIP实测是本地CPU推理，架构图另使用了ImageGen。", "",
           "## 2. HTTP写入负载", "",
           "Linux WSL、2 logical CPU affinity、持久ext4、SQLite FULL；没有施加云服务器RAM限额。真实本机HTTP含服务端JSON/图片/索引/提交，源生成不计入。",
           "每个普通profile只有12次请求；p99是小样本插值，不能当SLA。持续profile运行30秒。机器不是专用隔离主机，物理磁盘/CPU后台负载不能与云服务器等同。", "",
           "| 负载 | 请求数 | 请求wire最大MiB | Add p50/p95/p99 ms | 成功req/s | msg/s | img/s | RSS峰值MiB | 错误率 |",
           "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in final["performance"]:
        if row["strategy"]!="adaptive": continue
        lines.append(f"| {row['profile']} | {row['requests']} | {row['request_wire_bytes_max']/1024**2:.2f} | {row['add_p50_ms']:.1f}/{row['add_p95_ms']:.1f}/{row['add_p99_ms']:.1f} | {row['requests_per_second']:.2f} | {row['messages_per_second']:.2f} | {row['images_per_second']:.2f} | {row['rss_peak_bytes']/1024**2:.1f} | {row['error_rate']:.3f} |")
    lines += ["", "三种完整无模型HTTP策略的全部24个profile run和18个检索/消融run见[原始JSON](../results/prism-benchmark-20261016.json)。",
              "其中`source_bytes/s`是持久化源的wire等价值，不是物理磁盘写入量。实际OS write bytes/s、唯一图片新增BLOB bytes、SQLite/WAL字节和CPU秒均分开记录；不把重复引用当唯一图片存储。", "",
              "## 3. 可获得旧代码的同负载下界", "",
              "旧代码固定ZIP SHA256为`f838f314aab67e03a2a27411a3a9912dd55e566d8c2116eaf9ba27d6c82acc0e`；**模型设为none**。因此不是Quanzil线上旧服务，也不是官网2%/天的复现。旧/新运行不是严格交替的配对试验，倍率只描述本机观察值。", "",
              "| 负载 | 旧代码none req/s | Prism-Swift req/s | 本机观察倍率 |",
              "|---|---:|---:|---:|"]
    for old in legacy["performance"]:
        new=next(row for row in final["performance"] if row["strategy"]=="adaptive" and row["profile"]==old["profile"])
        lines.append(f"| {old['profile']} | {old['requests_per_second']:.2f} | {new['requests_per_second']:.2f} | {new['requests_per_second']/old['requests_per_second']:.2f}× |")
    lines += ["", "可证的优化机制是减少重复Base64存储、重复图片处理、无远程模型热路径，以及保持服务连接避免每请求最后连接关闭时的WAL收尾。",
              "旧全局锁/三次解析已在此前fast版部分修复，不能重复当作当前线上未修问题。具体源码证据和未确认根因见[bottleneck_analysis.md](bottleneck_analysis.md)。", "",
              "## 4. 大scope负载", "",
              f"- {scale['sources']}条源消息，{scale['image_occurrences']}个图片出现位置；每Add100条消息，共{scale['add_requests']}次真实FULL提交。",
              f"- 写入{scale['add_seconds']:.2f}秒，**{scale['messages_per_second']:.1f}消息/秒**；重开连接后Search p50/p95={scale['search_p50_ms']:.1f}/{scale['search_p95_ms']:.1f}ms，错误{scale['errors']}。",
              f"- 只有{scale['unique_blob_rows']}个唯一图片对象：这是重复图引用与大索引scope测试，**不是2500张独特自然图的吞吐测试**。",
              "- 直接Store调用，不含HTTP、上网、Answer/Eval；不能拿批量消息/秒冒充单消息请求/秒。[原始数据](../results/scale-20261016.json)。", "",
              "## 5. A/B/C与视觉质量", "",
              "完整同环境三seed A/B/C写入和原图保留对照见[visual_pipeline_comparison.md](visual_pipeline_comparison.md)。B写入TinyCLIP持久向量约17.42 req/s，C廉价索引约241.51 req/s，但C在无caption文字→视觉形状题没有语义收益，负结果已保留。",
              "下面是额外的**查询时**TinyCLIP：每seed144源、2用户，按固定随机种子从每类选择12题；3个独立合成seed。基础/重排使用相同源和同一批query，没有oracle过滤。", "",
              "| Query模式 | 文字→视觉Hit@1均值 | Search p95均值ms | 附图Hit@1 | 精确文字Hit@1 |",
              "|---|---:|---:|---:|---:|"]
    for count,enabled,label in ((32,False,"无重排配对基线"),(32,True,"查询重排32候选"),(64,True,"查询重排64候选")):
        values=[summary for run in visual["runs"] if run["candidates"]==count for summary in run["summaries"] if summary["reranker"]==enabled]
        lines.append(f"| {label} | {statistics.mean(row['quality']['visual_text']['hit_at_1'] for row in values):.4f} | {statistics.mean(row['quality']['visual_text']['search_p95_ms'] for row in values):.1f} | {statistics.mean(row['quality']['image_identity']['hit_at_1'] for row in values):.4f} | {statistics.mean(row['quality']['lexical']['hit_at_1'] for row in values):.4f} |")
    lines += ["", "这是容易的颜色/形状开发诊断，不是自然图片、OCR、复杂关系或AML分数。路由首次漏识别photograph而没有触发模型，初轮失败结果保留在`results/query-reranker-routing-initial.json`，通用词形修复后重跑；不把它说成独立held-out测试。",
              "[完整查询结果](../results/query-reranker-final-20261016.json)包含源哈希、每题指标、三seed、延迟与内存。候选缺失无法靠重排补救；本实验未直接测candidate recall oracle上界，不冒称100%候选覆盖。",
              "查询时模型保留Add的低开销，但会增加每个视觉Search约秒级CPU工作。若N张写入图、Q次视觉query、每次b张候选，编码次数近似B=N，查询模式C=Qb；只有Qb<N时C的总编码量才可能更低。它解决写入阶段依赖，不保证总计算更少。", "",
              "## 6. 消融、来源与范围", "",
              "主JSON包含raw、sparse、adaptive、no_visual、no_neighbours、no_coverage，每个检索条件3个生成seed。没有激活某模块的题型上零差异不能证明模块无用；coverage/邻居的研究收益未由独立自然数据建立。没有训练或私有评测调参。",
              "原始JSON保留各测量时源码hash。后续更名、视觉路由及回填历史的时间邻居修复由最终回归/Docker校验；较早JSON不冒充最终源码逐字节相同。部分旧JSON date字段是启动时写死的协议日期10月15日，实际交付测量跨10月15–16日；数值与原文件未改。",
              "新/旧各profile没有3轮独立性能重复，不能把单次倍率当置信区间。独立三seed仅用于生成数据的检索诊断；不伪造确定性重复的准确率方差。", "",
              "## 7. 截止时间与部署判断", "",
              "官方近5000评测实例不是Add请求数，完整图像字节/重试/队列未公开。使用`estimate_completion.py`代入实际Add/Search数量、带宽与Answer/Eval/排队时间；未提供的总耗时保持null。",
              "建议先部署无模型稳定模式并验证目标云盘；视觉profile使用独立配置/卷并提前测RAM和Search预算。部署/资格/材料建议在10月20日前确认，10月21–23日完成公开接口和官方Smoke，尽早完成有权运行的原/首次Full，给11月4日留故障余量。这是实施计划，不是评测完成承诺。",
              "最终官方Smoke/Full、排名、云端吞吐和自然图像QA结果均为null。"]
    (ROOT/"docs/performance_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Wrote measured performance report")


if __name__=="__main__": main()
