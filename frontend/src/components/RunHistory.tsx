import type { JobRecord } from '../types'

const statusLabel = {
  queued: '等待中',
  decoding: '读取视频',
  selecting: '定位目标',
  loading_model: '加载模型',
  inferencing: '正在追踪',
  rendering: '生成结果',
  complete: '查看',
  failed: '失败',
} as const

interface RunHistoryProps {
  jobs: JobRecord[]
  activeId?: string
  onSelect: (job: JobRecord) => void
}

export function RunHistory({ jobs, activeId, onSelect }: RunHistoryProps) {
  return (
    <aside className="run-history" aria-label="运行记录">
      <div className="section-kicker">运行记录 · {jobs.length}</div>
      {jobs.length === 0 ? <p className="empty-copy">还没有运行记录</p> : null}
      {jobs.map((job) => (
        <button
          type="button"
          className={`run-row ${activeId === job.id ? 'selected' : ''}`}
          key={job.id}
          onClick={() => onSelect(job)}
        >
          <span className={`run-state ${job.status}`} />
          <span>
            <strong>{job.filename}</strong>
            <small>
              {job.targets.length || 1} 个目标 · {job.tracking_mode === 'dense' ? `${job.tracked_frames || job.source_frames || '—'} 帧` : `${job.frame_count} 个关键帧`}
            </small>
          </span>
          <b>{job.status === 'complete' || job.status === 'failed' ? statusLabel[job.status] : `${job.progress}%`}</b>
        </button>
      ))}
    </aside>
  )
}
