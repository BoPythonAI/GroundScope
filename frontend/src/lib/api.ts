import type { Health, JobRecord, SubmitJob } from '../types'

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const data = (await response.json().catch(() => ({ detail: response.statusText }))) as { detail?: string }
    throw new Error(data.detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export async function fetchHealth(): Promise<Health> {
  return readJson<Health>(await fetch('/api/health'))
}

export async function fetchJobs(): Promise<JobRecord[]> {
  return readJson<JobRecord[]>(await fetch('/api/jobs'))
}

export async function submitJob(input: SubmitJob): Promise<JobRecord> {
  const form = new FormData()
  form.append('video', input.file)
  form.append('prompt', input.prompt)
  form.append('selector', input.selector)
  form.append('tracking_mode', input.trackingMode)
  form.append('frame_count', String(input.frameCount))
  return readJson<JobRecord>(await fetch('/api/jobs', { method: 'POST', body: form }))
}

export async function correctMask(
  jobId: string,
  frameSlot: number,
  targetIndex: number,
  operation: 'add' | 'erase',
  point: { x: number; y: number },
  brushRadius: number,
): Promise<JobRecord> {
  return readJson<JobRecord>(
    await fetch(`/api/jobs/${jobId}/corrections`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frame_slot: frameSlot,
        target_index: targetIndex,
        operation,
        shape: 'brush',
        points: [point],
        brush_radius: brushRadius,
      }),
    }),
  )
}
