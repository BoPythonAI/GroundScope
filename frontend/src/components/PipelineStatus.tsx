import type { JobRecord, JobStatus } from '../types'

const steps: Array<{ status: JobStatus; label: string }> = [
  { status: 'decoding', label: '读取视频' },
  { status: 'selecting', label: '定位目标' },
  { status: 'loading_model', label: '加载模型' },
  { status: 'inferencing', label: '逐帧追踪' },
  { status: 'rendering', label: '生成结果' },
  { status: 'complete', label: '已完成' },
]

const stageLabel: Partial<Record<JobStatus, string>> = {
  queued: '任务已加入队列',
  decoding: '正在读取视频',
  selecting: '正在定位目标',
  loading_model: '正在加载模型',
  inferencing: '正在逐帧追踪',
  rendering: '正在生成结果',
  complete: '处理完成',
  failed: '处理失败',
}

const order: Record<JobStatus, number> = {
  queued: -1,
  decoding: 0,
  selecting: 1,
  loading_model: 2,
  inferencing: 3,
  rendering: 4,
  complete: 5,
  failed: -1,
}

export function PipelineStatus({ job }: { job: JobRecord }) {
  const current = order[job.status]
  return (
    <section className="pipeline-panel" aria-live="polite">
      <div className="pipeline-head">
        <div>
          <span className="section-kicker">处理进度</span>
          <strong>{stageLabel[job.status] ?? job.stage}</strong>
        </div>
        <span className="progress-number">{job.progress.toString().padStart(3, '0')}%</span>
      </div>
      <div className="progress-track"><i style={{ width: `${job.progress}%` }} /></div>
      <ol className="pipeline-steps">
        {steps.map((step, index) => (
          <li className={index < current ? 'done' : index === current ? 'active' : ''} key={step.status}>
            <span>{String(index + 1).padStart(2, '0')}</span>{step.label}
          </li>
        ))}
      </ol>
      {job.error ? <div className="error-block">{job.error}</div> : null}
    </section>
  )
}
