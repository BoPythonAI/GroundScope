import { useMemo, useState } from 'react'
import type { Artifact, JobRecord } from '../types'

interface ResultsWorkspaceProps {
  job: JobRecord
  correctionBusy: boolean
  onCorrect: (
    frameSlot: number,
    targetIndex: number,
    operation: 'add' | 'erase',
    point: { x: number; y: number },
    radius: number,
  ) => Promise<void>
}

const targetColors = ['#23d3ee', '#ffc72c', '#ff5b70', '#7ee787']

const viewLabels = {
  overlay: '分割叠加',
  source: '原视频',
  mask: '掩码',
} as const

const artifactLabels: Record<string, string> = {
  'Overlay video': '关键帧叠加视频',
  'Binary-mask video': '关键帧掩码视频',
  'Selected frames': '关键帧原视频',
  'Mask PNG archive': '关键帧掩码压缩包',
  'Frame metrics': '关键帧指标',
  'Dense source video': '完整原视频',
  'Dense overlay video': '完整分割视频',
  'Dense mask video': '完整掩码视频',
  'Dense target masks': '逐帧目标掩码',
  'Dense frame metrics': '逐帧指标表',
  'Dense metrics summary': '指标汇总',
  'Audit report': '运行报告',
  Corrections: '人工修正记录',
}

const artifactKindLabels: Record<string, string> = {
  video: '视频',
  archive: '压缩包',
  table: '数据表',
  report: '报告',
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function artifactTitle(artifact: Artifact) {
  return artifactLabels[artifact.name] ?? artifact.name
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="metric-cell">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  )
}

export function ResultsWorkspace({ job, correctionBusy, onCorrect }: ResultsWorkspaceProps) {
  const [view, setView] = useState<'overlay' | 'source' | 'mask'>('overlay')
  const [selectedSlot, setSelectedSlot] = useState(0)
  const [selectedTargetIndex, setSelectedTargetIndex] = useState(0)
  const [correctionMode, setCorrectionMode] = useState<'add' | 'erase'>('add')
  const [brushRadius, setBrushRadius] = useState(0.02)
  const activeFrame = job.frames[Math.min(selectedSlot, Math.max(0, job.frames.length - 1))]
  const activeTarget = job.targets[selectedTargetIndex] ?? job.targets[0] ?? '目标'
  const version = encodeURIComponent(job.updated_at)
  const dense = job.tracking_mode === 'dense' && job.tracked_frames > job.frames.length

  const summary = useMemo(() => {
    if (dense && job.dense_summary) {
      return {
        coverage: job.dense_summary.mean_coverage,
        temporal: job.dense_summary.mean_temporal_iou,
        empty: job.dense_summary.empty_frame_ratio,
      }
    }
    if (!job.frames.length) return { coverage: 0, temporal: 0, empty: 1 }
    let coverage = 0
    let temporal = 0
    let temporalCount = 0
    let empty = 0
    for (const frame of job.frames) {
      coverage += frame.coverage
      if (frame.temporal_iou !== null) {
        temporal += frame.temporal_iou
        temporalCount += 1
      }
      if (frame.coverage === 0) empty += 1
    }
    return {
      coverage: coverage / job.frames.length,
      temporal: temporalCount ? temporal / temporalCount : 0,
      empty: empty / job.frames.length,
    }
  }, [dense, job.dense_summary, job.frames])

  const targetAverages = useMemo(() => {
    const values = new Map<string, number>()
    if (dense && job.dense_summary) {
      for (const target of job.targets) values.set(target, job.dense_summary.target_mean_coverage[target] ?? 0)
      return values
    }
    for (const target of job.targets) {
      let total = 0
      for (const frame of job.frames) total += frame.target_coverages?.[target] ?? 0
      values.set(target, job.frames.length ? total / job.frames.length : 0)
    }
    return values
  }, [dense, job.dense_summary, job.frames, job.targets])

  const videoUrl = view === 'overlay'
    ? `/artifacts/${job.id}/${dense ? 'dense_overlay' : 'overlay'}.mp4?v=${version}`
    : view === 'source'
      ? `/artifacts/${job.id}/${dense ? 'dense_source' : 'selected_frames'}.mp4?v=${version}`
      : `/artifacts/${job.id}/${dense ? 'dense_mask' : 'mask'}.mp4?v=${version}`

  const primaryDownload = job.artifacts.find((artifact) => artifact.url.endsWith(dense ? 'dense_overlay.mp4' : 'overlay.mp4'))
  const presenceRatios = Object.values(job.dense_summary?.target_presence_ratio ?? {})
  const meanPresence = presenceRatios.length
    ? presenceRatios.reduce((total, value) => total + value, 0) / presenceRatios.length
    : 1 - summary.empty
  const filterSummaries = Object.values(job.dense_summary?.target_filter_summary ?? {})
  const discardedComponents = filterSummaries.reduce(
    (total, item) => total + Number(item.discarded_components ?? 0),
    0,
  )
  const bootstrapRejections = filterSummaries.reduce(
    (total, item) => total + Number(item.bootstrap_rejections ?? 0),
    0,
  )
  const totalSeconds = (job.inference_seconds ?? 0) + (job.identity_filter_seconds ?? 0)

  return (
    <section className="results-workspace" aria-labelledby="results-heading">
      <div className="results-heading-row">
        <div>
          <span className="section-kicker">分析结果</span>
          <h2 id="results-heading">{dense ? (job.dense_summary?.filter_version === 'mask-refine-v3' ? '可靠掩码反向补全追踪' : job.dense_summary?.filter_version === 'component-reid-v2' ? '单目标约束逐帧追踪' : job.dense_summary?.filter_version ? '离场感知逐帧追踪' : '逐帧目标追踪') : '关键帧目标分割'}</h2>
          {job.dense_summary?.filter_version ? <span className="filter-badge">组件级身份校验已启用</span> : null}
        </div>
        <div className="result-actions">
          {primaryDownload ? <a className="download-primary" href={primaryDownload.url} download>下载结果</a> : null}
          <div className="view-switch" role="group" aria-label="视频视图">
            {(['overlay', 'source', 'mask'] as const).map((item) => (
              <button className={view === item ? 'active' : ''} type="button" key={item} onClick={() => setView(item)}>
                {viewLabels[item]}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="target-legend" aria-label="追踪目标">
        <span>追踪目标</span>
        {job.targets.map((target, index) => (
          <button
            type="button"
            className={selectedTargetIndex === index ? 'active' : ''}
            key={target}
            onClick={() => setSelectedTargetIndex(index)}
          >
            <i style={{ background: targetColors[index % targetColors.length] }} />
            <strong>{target}</strong>
            <small>平均占画面 {((targetAverages.get(target) ?? 0) * 100).toFixed(1)}%</small>
          </button>
        ))}
      </div>

      <div className="video-stage">
        <video key={videoUrl} controls playsInline preload="metadata" src={videoUrl}>
          当前浏览器不支持视频播放。
        </video>
        <div className="stage-meta">
          <span>{dense ? `完整视频 · ${job.source_fps?.toFixed(1) ?? '—'} FPS` : '关键帧视频 · 4 FPS'}</span>
          <span>已处理 {dense ? job.tracked_frames : job.frames.length} / {job.source_frames ?? '—'} 帧</span>
        </div>
      </div>

      <div className="quick-summary" aria-label="结果概览">
        <div><strong>{dense ? job.tracked_frames : job.frames.length}</strong><span>已处理帧</span></div>
        <div><strong>{job.targets.length}</strong><span>追踪目标</span></div>
        <div><strong>{totalSeconds ? totalSeconds.toFixed(1) : '—'} 秒</strong><span>处理耗时</span></div>
        <div><strong>{job.peak_vram_gib?.toFixed(1) ?? '—'} GiB</strong><span>峰值显存</span></div>
      </div>

      <details className="secondary-panel research-details">
        <summary><span>研究指标与模型输出</span><small>在场状态、身份一致性和原始响应</small></summary>
        <div className="metrics-board">
          <Metric label="目标覆盖率" value={`${(summary.coverage * 100).toFixed(1)}%`} note="所有目标掩码并集的平均面积" />
          <Metric label="时间一致性 IoU" value={summary.temporal.toFixed(3)} note={dense ? '相邻完整帧掩码的一致性' : '关键帧之间的未配准估计'} />
          <Metric label="有效在场率" value={`${(meanPresence * 100).toFixed(0)}%`} note="通过外观与语义校验的帧" />
          <Metric label="抑制身份跳变" value={`${job.dense_summary?.identity_rejections ?? 0}`} note="被拒绝的疑似错误掩码" />
          <Metric label="拆除联合区域" value={`${discardedComponents}`} note="被逐区域评分后丢弃的组件" />
          <Metric label="清除错误启动" value={`${bootstrapRejections}`} note="与可靠原型不连通的开头帧" />
          <Metric label="反向补全" value={`${job.dense_summary?.backward_refined_frames ?? 0}`} note="由可靠目标掩码向开头重播的帧" />
          <Metric label="重新识别" value={`${job.dense_summary?.reacquisitions ?? 0}`} note="离场后确认恢复的次数" />
          <Metric label="推理与传播" value={`${job.inference_seconds?.toFixed(2) ?? '—'}s`} note={`${job.identity_filter_seconds?.toFixed(2) ?? '—'}s 身份校验`} />
          <div className="model-response">
            <span>模型原始输出</span>
            {Object.entries(job.predictions).length ? Object.entries(job.predictions).map(([target, response], index) => (
              <p key={target}><i style={{ background: targetColors[index % targetColors.length] }} /> <b>{target}</b> — {response}</p>
            )) : <p>{job.prediction ?? '没有文本输出'}</p>}
            <small>{job.dense_summary?.reid_backend ?? job.selector_backend} · {job.selector.toUpperCase()}</small>
          </div>
        </div>
      </details>

      <details className="secondary-panel keyframe-details">
        <summary><span>关键帧与人工修正</span><small>查看分析帧，点击画面可补充或擦除掩码</small></summary>
        <div className="timeline-head">
          <span>关键帧时间轴</span>
          <span>黄色表示定位分数，彩色表示目标面积</span>
        </div>
        <div className="timeline" role="list" aria-label="关键帧时间轴">
          {job.frames.map((frame) => (
            <button
              type="button"
              role="listitem"
              className={`timeline-frame ${activeFrame?.slot === frame.slot ? 'active' : ''}`}
              key={frame.slot}
              onClick={() => setSelectedSlot(frame.slot)}
            >
              <div className="thumb-wrap">
                <img src={`${frame.overlay_url}?v=${version}`} alt={`${frame.timestamp.toFixed(2)} 秒的分割结果`} loading="lazy" />
                <i className="score-line" style={{ height: `${Math.max(4, frame.selection_score * 100)}%` }} />
                <i
                  className="coverage-line"
                  style={{
                    width: `${(frame.target_coverages?.[activeTarget] ?? frame.coverage) * 100}%`,
                    background: targetColors[selectedTargetIndex % targetColors.length],
                  }}
                />
              </div>
              <span>{frame.timestamp.toFixed(2)} 秒</span>
              <small>第 {frame.source_index} 帧</small>
            </button>
          ))}
        </div>

        {activeFrame ? (
          <div className="correction-panel">
            <div className="correction-copy">
              <span className="section-kicker">人工修正</span>
              <h3>修正“{activeTarget}” · 关键帧 {activeFrame.slot + 1}</h3>
              <p>选择目标与画笔操作，再点击右侧画面。当前修正仅更新关键帧，完整追踪视频仍保留原始结果。</p>
              <div className="correction-targets" role="group" aria-label="修正目标">
                {job.targets.map((target, index) => (
                  <button
                    type="button"
                    className={selectedTargetIndex === index ? 'active' : ''}
                    key={target}
                    onClick={() => setSelectedTargetIndex(index)}
                    style={{ borderColor: targetColors[index % targetColors.length] }}
                  >
                    {target}
                  </button>
                ))}
              </div>
              <div className="correction-controls">
                <button type="button" className={correctionMode === 'add' ? 'active' : ''} onClick={() => setCorrectionMode('add')}>＋ 补充</button>
                <button type="button" className={correctionMode === 'erase' ? 'active' : ''} onClick={() => setCorrectionMode('erase')}>－ 擦除</button>
                <label>画笔大小 <input type="range" min="0.005" max="0.1" step="0.005" value={brushRadius} onChange={(event) => setBrushRadius(Number(event.target.value))} /></label>
              </div>
            </div>
            <button
              type="button"
              className={`correction-canvas ${correctionBusy ? 'busy' : ''}`}
              disabled={correctionBusy}
              aria-label={`在关键帧 ${activeFrame.slot + 1} 上${correctionMode === 'add' ? '补充' : '擦除'}掩码`}
              onClick={async (event) => {
                const bounds = event.currentTarget.getBoundingClientRect()
                await onCorrect(
                  activeFrame.slot,
                  selectedTargetIndex,
                  correctionMode,
                  { x: (event.clientX - bounds.left) / bounds.width, y: (event.clientY - bounds.top) / bounds.height },
                  brushRadius,
                )
              }}
            >
              <img src={`${activeFrame.overlay_url}?v=${version}`} alt="当前关键帧修正画面" />
              <span>{correctionBusy ? '正在生成修正结果…' : '点击画面修正掩码'}</span>
            </button>
          </div>
        ) : null}
      </details>

      <details className="secondary-panel artifact-section">
        <summary><span>下载全部文件</span><small>{job.artifacts.length} 个视频、掩码和指标文件</small></summary>
        <div className="artifact-list">
          {job.artifacts.map((artifact) => (
            <a href={artifact.url} download key={artifact.url}>
              <span>{artifactKindLabels[artifact.kind] ?? artifact.kind}</span>
              <strong>{artifactTitle(artifact)}</strong>
              <small>{formatBytes(artifact.bytes)} ↓</small>
            </a>
          ))}
        </div>
      </details>
    </section>
  )
}
