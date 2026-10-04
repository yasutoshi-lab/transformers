"""工学系教科書 OCR から軽量 CPT コーパスを作る前処理パッケージ.

ステージごとにモジュールを分けている。各モジュールは単独で import・試験できる。

    books        対象書籍の一覧（カテゴリ別）
    glyphs       S1 文字正規化（NFKC・簡体字→日本字体・OCR 誤字補正）
    lines        S2 行クリーニング（柱・ページ番号行・段組み改行の接合）
    page_filter  S3 ページ除去ルールと閾値
    dedup        S4 重複除去（完全一致 + MinHash）
    corpus       S5/S6 分割と文書化（train / qa_eval / holdout_ppl）
    stats        統計の集計と書き出し
"""
