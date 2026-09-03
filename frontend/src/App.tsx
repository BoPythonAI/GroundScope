import { useCallback, useEffect, useState } from 'react'
import { ExperimentForm } from './components/ExperimentForm'
import { PipelineStatus } from './components/PipelineStatus'
import { ResultsWorkspace } from './components/ResultsWorkspace'
import { RunHistory } from './components/RunHistory'
import { SystemHeader } from './components/SystemHeader'
import { correctMask, fetchHealth, fetchJobs, submitJob } from './lib/api'
import type { Health, JobRecord, SubmitJob } from './types'

function mergeJob(jobs: JobRecord[], updated: JobRecord): JobRecord[] {
  const exists = jobs.some((job) => job.id === updated.id)
  const next = exists ? jobs.map((job) => job.id === updated.id ? updated : job) : [updated, ...jobs]
  return [...next].sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at))
}

export function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [jobs, setJobs] = useState<JobRecord[]>([])
  const [activeJob, setActiveJob] = useState<JobRecord | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [correctionBusy, setCorrectionBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    Promise.all([fetchHealth(), fetchJobs()])
      .then(([nextHealth, nextJobs]) => {
        if (cancelled) return
        setHealth(nextHealth)
        setJobs(nextJobs)
        setActiveJob(nextJobs[0] ?? null)
      })
      .catch((cause: unknown) => !cancelled && setError(cause instanceof Error ? cause.message : String(cause)))
    const interval = window.setInterval(() => {
      fetchHealth().then((value) => !cancelled && setHealth(value)).catch(() => undefined)
    }, 5000)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [])

  useEffect(() => {
    if (!activeJob || activeJob.status === 'complete' || activeJob.status === 'failed') return
    const events = new EventSource(`/api/jobs/${activeJob.id}/events`)
    events.addEventListener('job', (event) => {
      const updated = JSON.parse((event as MessageEvent<string>).data) as JobRecord
      setActiveJob(updated)
      setJobs((current) => mergeJob(current, updated))
      if (updated.status === 'complete' || updated.status === 'failed') events.close()
    })
    events.onerror = () => events.close()
    return () => events.close()
  }, [activeJob?.id, activeJob?.status])

  const handleSubmit = useCallback(async (input: SubmitJob) => {
    setSubmitting(true)
    setError(null)
    try {
      const job = await submitJob(input)
      setActiveJob(job)
      setJobs((current) => mergeJob(current, job))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSubmitting(false)
    }
  }, [])

  const handleCorrection = useCallback(async (
    frameSlot: number,
    targetIndex: number,
    operation: 'add' | 'erase',
    point: { x: number; y: number },
    radius: number,
  ) => {
    if (!activeJob) return
    setCorrectionBusy(true)
    setError(null)
    try {
      const updated = await correctMask(activeJob.id, frameSlot, targetIndex, operation, point, radius)
      setActiveJob(updated)
      setJobs((current) => mergeJob(current, updated))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setCorrectionBusy(false)
    }
  }, [activeJob])

  return (
    <div className="app-shell">
      <SystemHeader health={health} />
      <main>
        <div className="control-deck">
          <ExperimentForm disabled={submitting} onSubmit={handleSubmit} />
          <RunHistory jobs={jobs} activeId={activeJob?.id} onSelect={setActiveJob} />
        </div>
        {error ? <div className="global-error" role="alert"><strong>运行提示</strong>{error}</div> : null}
        {activeJob ? <PipelineStatus job={activeJob} /> : null}
        {activeJob?.status === 'complete' ? (
          <ResultsWorkspace job={activeJob} correctionBusy={correctionBusy} onCorrect={handleCorrection} />
        ) : null}
        {!activeJob ? (
          <section className="empty-state">
            <span>GPU 已就绪</span>
            <strong>上传视频，开始第一次目标分割。</strong>
          </section>
        ) : null}
      </main>
      <footer>
        <span>GroundScope v0.6 · 可信掩码纯视觉反向补全</span>
        <span>模型版本 82faf06 · BF16 / CUDA</span>
      </footer>
    </div>
  )
}
