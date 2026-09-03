export type JobStatus =
  | 'queued'
  | 'decoding'
  | 'selecting'
  | 'loading_model'
  | 'inferencing'
  | 'rendering'
  | 'complete'
  | 'failed'

export type SelectorName = 'uniform' | 'motion' | 'query' | 'hybrid'
export type TrackingMode = 'analysis' | 'dense'

export interface Artifact {
  name: string
  kind: string
  url: string
  bytes: number
}

export interface FrameResult {
  slot: number
  source_index: number
  timestamp: number
  thumbnail_url: string
  overlay_url: string | null
  mask_url: string | null
  selection_score: number
  relevance_score: number
  motion_score: number
  coverage: number
  target_coverages: Record<string, number>
  temporal_iou: number | null
}

export interface DenseSummary {
  frame_count: number
  fps: number
  mean_coverage: number
  empty_frame_ratio: number
  mean_temporal_iou: number
  target_mean_coverage: Record<string, number>
  target_presence_ratio: Record<string, number>
  identity_rejections: number
  reacquisitions: number
  filter_version: string | null
  reid_backend: string | null
  backward_refined_frames: number
  backward_refinement_seeds: Record<string, number>
  target_filter_summary: Record<string, Record<string, number | string>>
}

export interface JobRecord {
  id: string
  filename: string
  prompt: string
  targets: string[]
  selector: SelectorName
  tracking_mode: TrackingMode
  frame_count: number
  status: JobStatus
  progress: number
  stage: string
  created_at: string
  updated_at: string
  error: string | null
  prediction: string | null
  predictions: Record<string, string>
  duration_seconds: number | null
  source_fps: number | null
  source_frames: number | null
  inference_seconds: number | null
  peak_vram_gib: number | null
  tracked_frames: number
  propagation_seconds: number | null
  identity_filter_seconds: number | null
  dense_summary: DenseSummary | null
  selector_backend: string | null
  frames: FrameResult[]
  artifacts: Artifact[]
}

export interface Health {
  status: string
  gpu: string | null
  cuda_available: boolean
  model_ready: boolean
  model_path: string
  queue_depth: number
  disk_free_gib: number
}

export interface SubmitJob {
  file: File
  prompt: string
  selector: SelectorName
  trackingMode: TrackingMode
  frameCount: number
}
