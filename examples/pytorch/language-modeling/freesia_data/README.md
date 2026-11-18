# freesia_data

Freesia v2 のデータ作成・検証・アップロードに使ったスクリプト。もとは `~/freesia-work/scripts/` で実行したもので、パスは `~/freesia-work/` 配下の構成を前提にしている。

- 段階 0（コーパス・事前学習データ）: `top_boiler.py`, `inspect_contam.py`, `inspect_head.py`, `verify_pool_v2.py`, `verify_pt_v2.py`, `check_tail_j.py`, `rebuild_test_v2.py`, `fix_readmes.py`, `upload_tok_pt.py`, `upload_pt_v2.py`, `upload_test_v2.py`（コーパスの構築本体は `../build_freesia_corpus_v2.py`）
- 段階 2（因果 LM）: `eval_test_lm.py`, `upload_lm.py`, `upd_readme.py`
- 段階 3（freesia-weak-cl）: `extract_weakcl_passages.py`, `weakcl_rules.py`, `weakcl_embed.py`, `weakcl_rank.py`, `weakcl_build.py`, `weakcl_trace.py`, `weakcl_b_prepare.py`, `gen_queries_b.py`, `weakcl_b_embed.py`, `weakcl_b_build.py`, `run_b_post.sh`, `after_gen_b.sh`, `upload_weakcl.py`, `upload_weak_model.py`
- 段階 4（freesia-supervised-cl）: `supcl_labels.py`, `supcl_prepare.py`, `supcl_gen.py`, `supcl_embed.py`, `supcl_mine.py`, `supcl_judge.py`, `supcl_build.py`, `run_supcl_pilot.sh`, `run_supcl_full.sh`, `run_supcl_full_ws2.sh`, `supcl_switch_ws2.sh`, `upload_supcl.py`

HF のトークンは実行時に読み込む（スクリプトには含まない）。
