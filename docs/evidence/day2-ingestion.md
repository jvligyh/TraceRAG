# Day 2 文档接入证据

- 日期：2026-09-14（Asia/Shanghai）
- Markdown 冒烟测试：2个文件成功解析，在120字符上限下生成9个chunk。
- PDF 冒烟测试：1份用户提供的单页PDF成功解析为1个文档、4个chunk。
- PDF document id：`pdf-919bcd2f8f93d11d-p0001`
- 同一PDF重复解析：document id 与4个chunk id全部一致。
- 提取文本中 Unicode 替换字符 `�` 数量：0。
- 单元测试：16项通过。
- 密钥和PDF原文均未写入此证据或Git。

PDF字体映射存在 `pypdf` 底层警告，但本次页面文本提取和切块完成。后续需要在评测集中单独检查乱码、扫描件和无文本PDF。
