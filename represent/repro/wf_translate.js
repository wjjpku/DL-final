export const meta = {
  name: 'translate-reports-zh',
  description: 'Translate the 3 English sub-reports into Chinese in place',
  phases: [{ title: 'Translate', detail: '3 agents, one report each' }],
}
const ROOT = 'c:/Users/21100/Desktop/represent'
const GLOSSARY = `
术语表(务必一致):adiabatic=绝热, non-adiabatic lag=非绝热滞后, residual=残差,
relaxation time=弛豫时间, spectral mixture=谱混合, noise floor=噪声地板,
preconditioned curvature=预条件曲率, edge of stability=失稳边缘, weight decay=权重衰减,
momentum=动量, cumulative-drop=累积-drop, adiabatic baseline=绝热基线, held-out=留出,
circularity=循环论证, artifact=伪影, claim=主张, verdict=裁决, schedule=调度,
learning rate=学习率, loss curve=损失曲线, amplitude=幅度, spectrum=谱, scale=尺度,
scale-invariant=尺度不变, falsifiable=可证伪, leakage=泄漏, sweep=扫描.
`
const RULES = `
规则(严格遵守):
- 把英文 markdown 翻译成流畅、专业的中文,直接 OVERWRITE 同一个文件(用 Write 写回原路径)。
- 完整保留:所有数字、公式、代码标识符(函数名/文件路径/变量名如 lambda_slow, tau, eta, R², p, beta2, DropRelaxS, MPL, NQM, wsdcon, wsd_sharp 等)、markdown 表格、链接、标题层级、列表结构。
- 只翻译散文与表头/说明文字;不要增删内容、不要改动任何数值或结论。
- 表格里的英文术语按术语表译,但保留数字与符号不变。
- 输出后用 Read 确认文件已是中文。
`
phase('Translate')
const files = ['results/NQM_REPORT.md', 'results/EXTENSIONS_REPORT.md', 'results/AUDIT_PARTC.md']
const r = await parallel(files.map(f => () =>
  agent(`把 ${ROOT}/${f} 翻译成中文。先 Read 该文件,然后 Write 回同一路径(中文版)。\n${GLOSSARY}\n${RULES}\n完成后返回一句话:文件名 + 已翻译的行数。`,
    { label: 'tr:' + f.split('/').pop(), phase: 'Translate' })))
return { translated: files, notes: r }
