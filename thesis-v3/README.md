# thesis-v2

基于 [hithesis-dev](https://github.com/dustincys/hithesis) 模板重新组织的中期报告工程。

## 目录结构

```
thesis-v2/midterm/
├── report.tex              # 主文件（哈尔滨硕士中期）
├── front/coverart.tex      # 封面信息（含 ctype 实践成果/学位论文）
├── body/                   # 正文（同步自 thesis/midterm）
├── figures/                # 插图
├── reference.bib           # 参考文献（同步自 thesis/references.bib）
├── hithesisart.cls/.cfg    # hithesis-dev 生成的报告类
├── hithesis.bst
├── latexmkrc
└── Makefile
```

## 编译

```bash
cd thesis-v2/midterm
latexmk -xelatex report.tex
```

或使用 `make report`。

## 与 thesis/midterm 的差异

| 项目 | thesis/midterm | thesis-v2/midterm |
|------|----------------|-------------------|
| 模板来源 | 旧版 hithesisart（无 ctype） | hithesis-dev 最新 hithesisart |
| 封面 ctype | 不支持 | 支持「实践成果/学位论文」 |
| 参考文献 | `../references.bib` | 本地 `reference.bib` |
| 工程位置 | `thesis/midterm/` | 独立 `thesis-v2/midterm/` |

正文内容与 `thesis/midterm/body/report_harbin_master_midterm.tex` 保持一致；后续可在 v2 中独立迭代。
