# -*- coding: utf-8 -*-
# @File    : app.py
"""
MEG-mod 交互式预测界面 (Streamlit)

输入 chemically modified siRNA 的 sense / antisense 序列与化学修饰信息，
调用 MEG_mod_predictor 返回敲低效率（knockdown efficiency）预测，
并可查看双链共折叠（RNAcofold）结构与碱基配对。

运行：
    cd MEG-mod 项目根目录
    streamlit run app.py

依赖：
    streamlit, torch, torch_geometric(PyG), multimolecule
    可选（在线计算共折叠结构时需要）：ViennaRNA 工具包中的 RNAcofold
    预计算文件：data_pre/*.pkl 与 Saved_Best_Models/best_model.pt
                （见 README 的 Zenodo / HuggingFace 下载说明）
"""

import os
import sys
import pickle
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from BAN_graph import MEG_mod_predictor
from utils import run_rnacofold, dotbracket_to_pairs, parse_dotplot_ps

try:
    from multimolecule import RnaTokenizer, RnaErnieModel
    HAS_RNAERNIE = True
except Exception:
    HAS_RNAERNIE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Cfg:
    ckpt: str = "Saved_Best_Models/best_model.pt"
    data_pre_dir: str = "data_pre"
    rnaernie_dir: str = "rnaernie"
    use_prob: bool = True
    prob_threshold: float = 0.2
    include_intra_mfe: bool = False
    batch_size: int = 64


# ----------------------------------------------------------------------------
# 模型与特征加载（尽量复用仓库内已有函数，避免重复实现）
# ----------------------------------------------------------------------------
@st.cache_resource
def load_rnaernie_local(rnaernie_dir):
    if not HAS_RNAERNIE:
        raise RuntimeError("未安装 multimolecule，无法在线计算 RNAErnie 嵌入，请预先准备 rnaernie_base_emb_fixed.pkl")
    tokenizer = RnaTokenizer.from_pretrained(rnaernie_dir)
    model = RnaErnieModel.from_pretrained(rnaernie_dir).to(DEVICE)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def compute_base_emb_local(seqs, ids, tokenizer, model, max_len=29):
    out = {}
    for sid, seq in zip(ids, seqs):
        sid = str(sid)
        token = tokenizer(
            seq, return_tensors="pt", truncation=True,
            padding="max_length", max_length=max_len,
        )
        token = {k: v.to(DEVICE) for k, v in token.items()}
        h = model(**token, output_hidden_states=True).last_hidden_state[:, 1:-1, :]
        out[sid] = h[0].cpu()
    return out


def ensure_base_embeddings(df, base_dict, rnaernie_dir):
    """base_dict 中缺失的序列，若环境支持则在线用 RNAErnie 计算。"""
    need_ids, need_seqs = [], []
    for _, row in df.iterrows():
        for k, s in [("sense_id", "sense"), ("anti_id", "antisense")]:
            sid = str(row[k])
            if sid not in base_dict:
                need_ids.append(sid)
                need_seqs.append(str(row[s]))
    if need_ids:
        if not HAS_RNAERNIE:
            raise RuntimeError(
                f"以下序列在预计算文件中缺失且无 multimolecule，无法计算嵌入：{sorted(set(need_ids))}"
            )
        tok, mdl = load_rnaernie_local(rnaernie_dir)
        new = compute_base_emb_local(need_seqs, need_ids, tok, mdl)
        base_dict.update(new)
        st.info(f"[RNAErnie] 在线计算 {len(new)} 条新嵌入")
    return base_dict


def ensure_cofold(df, cofold_dict, use_prob):
    """cofold_dict 中缺失的序列对，若环境支持则在线用 RNAcofold 计算。"""
    for _, row in df.iterrows():
        key = f"{row['sense_id']}|{row['anti_id']}"
        if key not in cofold_dict:
            try:
                co = run_rnacofold(row["sense"], row["antisense"], generate_prob=use_prob)
            except Exception as e:
                raise RuntimeError(
                    f"缺失共折叠预计算且 RNAcofold 调用失败（{e}）。\n"
                    f"请在 data_pre/cofold_results.pkl 中补充 {key}，或安装 ViennaRNA 的 RNAcofold。"
                )
            pairs = dotbracket_to_pairs(co.dot_bracket)
            prob_map = parse_dotplot_ps(co.dotplot_path) if use_prob else {}
            cofold_dict[key] = {
                "dot_bracket": co.dot_bracket,
                "pairs_mfe": pairs,
                "prob_map": prob_map,
            }
            st.info(f"[cofold] 在线计算 {key}")
    return cofold_dict


@st.cache_resource
def build_model(cfg: Cfg):
    data_pre_dir = cfg.data_pre_dir if os.path.isabs(cfg.data_pre_dir) else os.path.join(ROOT, cfg.data_pre_dir)
    model = MEG_mod_predictor(
        device=DEVICE, combine_1_dim=512, rnaernie_dim=768, pc_dim=10,
        use_prob=cfg.use_prob, prob_threshold=cfg.prob_threshold,
        include_intra_mfe_pairs=cfg.include_intra_mfe, data_pre_dir=data_pre_dir,
    ).to(DEVICE)
    ckpt = cfg.ckpt if os.path.isabs(cfg.ckpt) else os.path.join(ROOT, cfg.ckpt)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def run_prediction(df: pd.DataFrame, model, cfg: Cfg):
    base_dict = dict(model.base_embeddings) if isinstance(model.base_embeddings, dict) else {}
    cofold_dict = dict(model.cofold_dict) if isinstance(model.cofold_dict, dict) else {}
    base_dict = ensure_base_embeddings(df, base_dict, cfg.rnaernie_dir)
    cofold_dict = ensure_cofold(df, cofold_dict, cfg.use_prob)
    model.base_embeddings = base_dict
    model.cofold_dict = cofold_dict

    cols = ["sense", "antisense"]
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype(str).str.lower()

    preds = []
    with torch.no_grad():
        for i in range(0, len(df), cfg.batch_size):
            sub = df.iloc[i:i + cfg.batch_size]
            out = model(
                sub["sense_id"].astype(str).tolist(),
                sub["anti_id"].astype(str).tolist(),
                sub["sense"].astype(str).tolist(),
                sub["antisense"].astype(str).tolist(),
                sub["sense_mod_types"].astype(str).tolist(),
                sub["sense_mod_positions"].astype(str).tolist(),
                sub["anti_mod_types"].astype(str).tolist(),
                sub["anti_mod_positions"].astype(str).tolist(),
                sub["concentration"].tolist(),
            )
            preds.extend(out.view(-1).cpu().numpy().tolist())

    res = df.copy()
    res["pred"] = [round(float(p), 4) for p in preds]
    res["pred_pct"] = [round(float(p) * 100, 2) for p in preds]
    return res


def cofold_of(sense, antisense, use_prob):
    try:
        co = run_rnacofold(sense, antisense, generate_prob=use_prob)
        return co.dot_bracket, co.mfe_energy
    except Exception as e:
        return None, f"RNAcofold 不可用：{e}"


# ----------------------------------------------------------------------------
# 界面
# ----------------------------------------------------------------------------
st.set_page_config(page_title="MEG-mod 预测界面", layout="wide")
st.title("MEG-mod · 化学修饰 siRNA 敲低效率预测")
st.caption(f"设备: {DEVICE} ｜ RNAErnie 在线编码: {'可用' if HAS_RNAERNIE else '不可用（需预计算 pkl）'}")


# 侧边栏配置
with st.sidebar:
    st.header("配置")
    cfg = Cfg()
    cfg.ckpt = st.text_input("模型检查点路径", value=cfg.ckpt)
    cfg.data_pre_dir = st.text_input("预计算目录 data_pre", value=cfg.data_pre_dir)
    cfg.rnaernie_dir = st.text_input("RNAErnie 模型目录", value=cfg.rnaernie_dir)
    cfg.use_prob = st.checkbox("使用共折叠配对概率", value=cfg.use_prob)
    cfg.prob_threshold = st.number_input("配对概率阈值", value=cfg.prob_threshold, step=0.05, min_value=0.0, max_value=1.0)
    cfg.include_intra_mfe = st.checkbox("保留链内 MFE 配对", value=cfg.include_intra_mfe)
    cfg.batch_size = st.number_input("批大小", value=cfg.batch_size, step=16, min_value=1)

    data_pre_abs = cfg.data_pre_dir if os.path.isabs(cfg.data_pre_dir) else os.path.join(ROOT, cfg.data_pre_dir)
    ckpt_abs = cfg.ckpt if os.path.isabs(cfg.ckpt) else os.path.join(ROOT, cfg.ckpt)
    with st.expander("环境与文件检查"):
        st.write("data_pre 目录:", "✅ 存在" if os.path.isdir(data_pre_abs) else "❌ 缺失")
        st.write("模型检查点:", "✅ 存在" if os.path.isfile(ckpt_abs) else "❌ 缺失")
        st.write("RNAcofold:", "✅ 在 PATH 中" if shutil.which("RNAcofold") else "❌ 缺失（批量新序列将无法在线共折叠）")

# 载入模型
try:
    model = build_model(cfg)
    st.sidebar.success("模型已加载")
except Exception as e:
    st.sidebar.error(f"模型加载失败：{e}")
    st.stop()


tab_single, tab_batch = st.tabs(["单条预测", "批量预测（CSV / Excel）"])

with tab_single:
    st.subheader("单条 siRNA 预测")
    st.markdown(
        "修饰格式示例：类型用 `*` 分隔，位置用 `*` 分隔、逗号列举。"
        "例如 `sense_mod_types = 2'-O-Me*PS`，`sense_mod_positions = 1,2*3,4` 表示第 1、2 位为 2'-O-Me，第 3、4 位为 PS。"
    )
    with st.form("single_form"):
        c1, c2 = st.columns(2)
        sense = c1.text_input("Sense 序列 (5'->3')", value="gcuacguacguacguacgua")
        antisense = c2.text_input("Antisense 序列 (5'->3')", value="uacguacguacguacguagcu")
        c3, c4 = st.columns(2)
        s_mod = c3.text_input("sense 修饰类型", value="")
        s_pos = c4.text_input("sense 修饰位置", value="")
        c5, c6 = st.columns(2)
        a_mod = c5.text_input("antisense 修饰类型", value="")
        a_pos = c6.text_input("antisense 修饰位置", value="")
        conc = st.number_input("浓度 (concentration)", value=1.0, step=0.5, min_value=0.0)
        submitted = st.form_submit_button("预测")

    if submitted:
        row = {
            "sense_id": "sense_0", "anti_id": "anti_0",
            "sense": sense.strip(), "antisense": antisense.strip(),
            "sense_mod_types": s_mod, "sense_mod_positions": s_pos,
            "anti_mod_types": a_mod, "anti_mod_positions": a_pos,
            "concentration": float(conc),
        }
        if not row["sense"] or not row["antisense"]:
            st.error("Sense 与 Antisense 序列均不能为空")
        else:
            with st.spinner("预测中…"):
                try:
                    df_single = pd.DataFrame([row])
                    res = run_prediction(df_single, model, cfg)
                    pred = float(res.iloc[0]["pred"])
                    pred_pct = float(res.iloc[0]["pred_pct"])
                    st.metric("预测敲低效率 (0–1)", f"{pred:.4f}", help="模型输出为 0–1 之间的连续值")
                    st.metric("预测敲低效率 (%)", f"{pred_pct:.2f}%")
                except Exception as e:
                    st.error(f"预测失败：{e}")

            # 共折叠结构（独立展示，不阻塞预测）
            with st.spinner("计算共折叠结构…"):
                db, mfe = cofold_of(sense.strip().lower(), antisense.strip().lower(), cfg.use_prob)
            if db:
                st.subheader("双链共折叠结构 (RNAcofold)")
                st.code(db, language="text")
                if isinstance(mfe, (int, float)):
                    st.caption(f"MFE 自由能: {mfe:.2f} kcal/mol")
                pairs = dotbracket_to_pairs(db)
                paired = len(pairs)
                st.caption(f"最小自由能配对碱基对数: {paired}")
            else:
                st.warning(f"无法展示共折叠结构：{mfe}")


with tab_batch:
    st.subheader("批量预测")
    st.markdown(
        "上传 CSV / Excel，列名需包含：`sense`, `antisense`, `concentration`，"
        "以及可选修饰列 `modification_sense`, `sense_position`, `modification_antisense`, `antisense_position`，"
        "可选 `sense_id`, `anti_id`。缺少 id 时会自动生成。"
    )
    uploaded = st.file_uploader("选择文件", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"读取文件失败：{e}")
            df_up = None

        if df_up is not None:
            # 列名归一化
            rename = {
                "modification_sense": "sense_mod_types",
                "modification_antisense": "anti_mod_types",
                "sense_position": "sense_mod_positions",
                "antisense_position": "anti_mod_positions",
            }
            df_up = df_up.rename(columns=rename)
            for c in ["sense", "antisense", "sense_mod_types", "sense_mod_positions",
                      "anti_mod_types", "anti_mod_positions"]:
                if c not in df_up.columns:
                    df_up[c] = ""
            if "concentration" not in df_up.columns:
                df_up["concentration"] = 1.0
                st.warning("未找到 concentration 列，已默认填 1.0")
            df_up["concentration"] = pd.to_numeric(df_up["concentration"], errors="coerce").fillna(1.0)
            if "sense_id" not in df_up.columns:
                df_up["sense_id"] = [f"sense_{i}" for i in range(len(df_up))]
            if "anti_id" not in df_up.columns:
                df_up["anti_id"] = [f"anti_{i}" for i in range(len(df_up))]

            st.write(f"共 {len(df_up)} 条记录，预览前 5 行：")
            st.dataframe(df_up.head())

            if st.button("运行批量预测"):
                with st.spinner("批量预测中…"):
                    try:
                        res_all = run_prediction(df_up, model, cfg)
                        st.success("预测完成")
                        st.dataframe(res_all[["sense_id", "anti_id", "sense", "antisense",
                                              "concentration", "pred", "pred_pct"]])
                        csv = res_all.to_csv(index=False).encode("utf-8")
                        st.download_button("下载结果 CSV", csv, "meg_mod_predictions.csv", "text/csv")
                    except Exception as e:
                        st.error(f"批量预测失败：{e}")
